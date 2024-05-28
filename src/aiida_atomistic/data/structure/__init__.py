###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""This module defines the classes for structures and all related
functions to operate on them.
"""

import copy
import functools
import itertools
import json
import typing as t

from aiida_atomistic.data.structure.utils import (
    _get_valid_cell,
    get_valid_pbc,
    calc_cell_volume,
    get_formula,
    get_kinds
)
from .site import Site

from aiida.common.constants import elements
from aiida.common.exceptions import UnsupportedSpeciesError

from aiida.orm.nodes.data import Data, Dict

try:
    import ase  # noqa: F401
    from ase import io as ase_io
    has_ase=True
except ImportError:
    has_ase=False

try:
    import pymatgen.core  as core # noqa: F401
    has_pymatgen=True
except ImportError:
    has_pymatgen=False

from aiida.common.constants import elements

_MASS_THRESHOLD = 1.0e-3
# Threshold to check if the sum is one or not
_SUM_THRESHOLD = 1.0e-6
# Default cell
_DEFAULT_CELL = ((0, 0, 0), (0, 0, 0), (0, 0, 0))

_valid_symbols = tuple(i['symbol'] for i in elements.values())
_atomic_masses = {el['symbol']: el['mass'] for el in elements.values()}
_atomic_numbers = {data['symbol']: num for num, data in elements.items()}

__all__ = ('StructureData', 'Site')

class StructureData(Data):
    """Data class that represents an atomic structure.

    The data is organized as a collection of sites together with a cell, the boundary conditions (whether they are
    periodic or not) and other related useful information.
    """

    _dimensionality_label = {0: '', 1: 'length', 2: 'surface', 3: 'volume'}
    _internal_kind_tags = None
    _global_properties = [
        "cell",
        "pbc",
    ]


    def __init__(
        self,
        cell: t.List[t.List[float]] = _DEFAULT_CELL,
        pbc: t.List[float] = [True, True, True],
        pbc1 = None,
        pbc2 = None,
        pbc3 = None,
        #kinds = None,
        #positions = None,
        #masses = None,
        #charges = None,
        **kwargs,
    ):
        """
        This is the constructor of StructureData. It should be possible to provide 
        all the necessary informations needed to completely define a StructureData 
        instance.    
        """
        if pbc1 is not None and pbc2 is not None and pbc3 is not None:
            pbc = [pbc1, pbc2, pbc3]

        super().__init__(**kwargs)

        self.set_cell(cell)
        self.set_pbc(pbc)

        #if kinds is not None:
        #    self.base.attributes.set('kinds', kinds)

        #if positions is not None:
        #    self.base.attributes.set('sites', sites)
    
           
    @classmethod
    def from_ase(cls, aseatoms: ase.Atoms):
        """Load the structure from a ASE object"""
        
        if not has_ase:
            raise ImportError("The ASE package cannot be imported.")
        
        # Read the ase structure
        cell = aseatoms.cell
        pbc = aseatoms.pbc
        #self.clear_kinds()  # This also calls clear_sites
        
        structure = cls(cell=cell,pbc=pbc)
        for atom in aseatoms:
            structure.append_atom(ase=atom)
            
        return structure
    
    @classmethod
    def from_file(cls, filename, format="cif", **kwargs):
        """Load the structure from a file"""
        
        ase_read = ase_io.read(filename, format=format, **kwargs)
            
        return StructureData.from_ase(aseatoms=ase_read)

    @classmethod
    def from_pymatgen(cls, pymatgen_obj: t.Union[core.structure.Molecule,core.structure.Structure], **kwargs):
        """Load the structure from a pymatgen object.

        .. note:: Requires the pymatgen module (version >= 3.0.13, usage
            of earlier versions may cause errors).
        """
        if not has_pymatgen:
            raise ImportError("The pymatgen package cannot be imported.")
            
        if isinstance(pymatgen_obj, core.structure.Molecule):
            structure = cls.from_pymatgen_molecule(pymatgen_obj)
        else:
            structure = cls.from_pymatgen_structure(pymatgen_obj)
        
        return structure
        
    @staticmethod   
    def from_pymatgen_molecule(mol: core.structure.Molecule, margin=5):
        """Load the structure from a pymatgen Molecule object.

        :param margin: the margin to be added in all directions of the
            bounding box of the molecule.

        .. note:: Requires the pymatgen module (version >= 3.0.13, usage
            of earlier versions may cause errors).
        """
        box = [
            max(x.coords.tolist()[0] for x in mol.sites) - min(x.coords.tolist()[0] for x in mol.sites) + 2 * margin,
            max(x.coords.tolist()[1] for x in mol.sites) - min(x.coords.tolist()[1] for x in mol.sites) + 2 * margin,
            max(x.coords.tolist()[2] for x in mol.sites) - min(x.coords.tolist()[2] for x in mol.sites) + 2 * margin,
        ]
        structure = StructureData.from_pymatgen_structure(mol.get_boxed_structure(*box))
        structure.pbc = [False, False, False]
        
        return structure

    @staticmethod
    def from_pymatgen_structure(struct: core.structure.Structure):
        """Load the structure from a pymatgen Structure object.

        .. note:: periodic boundary conditions are set to True in all
            three directions.
        .. note:: Requires the pymatgen module (version >= 3.3.5, usage
            of earlier versions may cause errors).

        :raise ValueError: if there are partial occupancies together with spins.
        """

        def build_kind_name(species_and_occu):
            """Build a kind name from a pymatgen Composition, including an additional ordinal if spin is included,
            e.g. it returns '<specie>1' for an atom with spin < 0 and '<specie>2' for an atom with spin > 0,
            otherwise (no spin) it returns None

            :param species_and_occu: a pymatgen species and occupations dictionary
            :return: a string representing the kind name or None
            """
            species = list(species_and_occu.keys())
            occupations = list(species_and_occu.values())

            # As of v2023.9.2, the ``properties`` argument is removed and the ``spin`` argument should be used.
            # See: https://github.com/materialsproject/pymatgen/commit/118c245d6082fe0b13e19d348fc1db9c0d512019
            # The ``spin`` argument was introduced in v2023.6.28.
            # See: https://github.com/materialsproject/pymatgen/commit/9f2b3939af45d5129e0778d371d814811924aeb6
            has_spin_attribute = hasattr(species[0], '_spin')

            if has_spin_attribute:
                has_spin = any(specie.spin != 0 for specie in species)
            else:
                has_spin = any(specie.as_dict().get('properties', {}).get('spin', 0) != 0 for specie in species)

            has_partial_occupancies = len(occupations) != 1 or occupations[0] != 1.0

            if has_partial_occupancies and has_spin:
                raise ValueError('Cannot set partial occupancies and spins at the same time')

            if has_spin:
                symbols = [specie.symbol for specie in species]
                kind_name = create_automatic_kind_name(symbols, occupations)

                # If there is spin, we can only have a single specie, otherwise we would have raised above
                specie = species[0]
                if has_spin_attribute:
                    spin = specie.spin
                else:
                    spin = specie.as_dict().get('properties', {}).get('spin', 0)

                if spin < 0:
                    kind_name += '1'
                else:
                    kind_name += '2'

                return kind_name

            return None

        cell = struct.lattice.matrix.tolist()
        pbc = [True, True, True]
        #self.clear_kinds()
        
        structure = StructureData(cell=cell, pbc=pbc)

        for site in struct.sites:
            symbols = site.species

            if 'kind_name' in site.properties:
                kind_name = site.properties['kind_name']
            else:
                kind_name = site.label

            inputs = {
                'symbol': site.specie.symbol,
                'weights': site.species.weight,
                'position': site.coords.tolist(),
            }

            if kind_name is not None:
                inputs['name'] = kind_name

            structure.append_atom(**inputs)

        return structure
    
    def get_property_names(self,domain=None):
        """get a list of properties

        Args:
            domain (str, optional): restrict the domain of the printed property names. Defaults to None.
        """
        if not domain:
            return self._global_properties + self.sites[0]._site_properties
        elif domain == "site":
            return self.sites[0]._site_properties
    
    def get_site_property(self,property):
        """Return a list with length equal to the number of sites of this structure,
        where each element of the list is the property of the corresponding site.

        :return: a list of floats
        """
        return [getattr(this_site,property) for this_site in self.sites]
    
    
    def _validate(self):
        """Performs some standard validation tests."""
        from aiida.common.exceptions import ValidationError

        super()._validate()

        try:
            _get_valid_cell(self.cell)
        except ValueError as exc:
            raise ValidationError(f'Invalid cell: {exc}')

        try:
            get_valid_pbc(self.pbc)
        except ValueError as exc:
            raise ValidationError(f'Invalid periodic boundary conditions: {exc}')

        self._validate_dimensionality()

        try:
            # This will try to create the kinds objects
            kinds = set(self.get_site_property("kind_name"))
        except ValueError as exc:
            raise ValidationError(f'Unable to validate the kinds: {exc}')

        from collections import Counter

        counts = Counter([k for k in kinds])
        for count in counts:
            if counts[count] != 1:
                raise ValidationError(f"Kind with name '{count}' appears {counts[count]} times instead of only one")

        try:
            # This will try to create the sites objects
            sites = self.sites
        except ValueError as exc:
            raise ValidationError(f'Unable to validate the sites: {exc}')

        for site in sites:
            if site.kind_name not in kinds:
                raise ValidationError(f'A site has kind {site.kind_name}, but no specie with that name exists')

        kinds_without_sites = set(k for k in kinds) - set(s.kind_name for s in sites)
        if kinds_without_sites:
            raise ValidationError(
                f'The following kinds are defined, but there are no sites with that kind: {list(kinds_without_sites)}'
            )

    def _prepare_xsf(self, main_file_name=''):
        """Write the given structure to a string of format XSF (for XCrySDen)."""
        if self.is_alloy or self.has_vacancies:
            raise NotImplementedError('XSF for alloys or systems with vacancies not implemented.')

        sites = self.sites

        return_string = 'CRYSTAL\nPRIMVEC 1\n'
        for cell_vector in self.cell:
            return_string += ' '.join([f'{i:18.10f}' for i in cell_vector])
            return_string += '\n'
        return_string += 'PRIMCOORD 1\n'
        return_string += f'{int(len(sites))} 1\n'
        for site in sites:
            # I checked above that it is not an alloy, therefore I take the
            # first symbol
            return_string += f'{_atomic_numbers[self.get_kind(site.kind_name).symbols[0]]} '
            return_string += '%18.10f %18.10f %18.10f\n' % tuple(site.position)
        return return_string.encode('utf-8'), {}

    def _prepare_cif(self, main_file_name=''):
        """Write the given structure to a string of format CIF."""
        from aiida.orm import CifData

        cif = CifData(ase=self.to_ase())
        return cif._prepare_cif()

    def _prepare_chemdoodle(self, main_file_name=''):
        """Write the given structure to a string of format required by ChemDoodle."""
        from itertools import product

        import numpy as np

        supercell_factors = [1, 1, 1]

        # Get cell vectors and atomic position
        lattice_vectors = np.array(self.base.attributes.get('cell'))
        base_sites = self.base.attributes.get('sites')

        start1 = -int(supercell_factors[0] / 2)
        start2 = -int(supercell_factors[1] / 2)
        start3 = -int(supercell_factors[2] / 2)

        stop1 = start1 + supercell_factors[0]
        stop2 = start2 + supercell_factors[1]
        stop3 = start3 + supercell_factors[2]

        grid1 = range(start1, stop1)
        grid2 = range(start2, stop2)
        grid3 = range(start3, stop3)

        atoms_json = []

        # Manual recenter of the structure
        center = (lattice_vectors[0] + lattice_vectors[1] + lattice_vectors[2]) / 2.0

        for ix, iy, iz in product(grid1, grid2, grid3):
            for base_site in base_sites:
                shift = (ix * lattice_vectors[0] + iy * lattice_vectors[1] + iz * lattice_vectors[2] - center).tolist()

                kind_name = base_site['kind_name']
                kind_string = self.get_kind(kind_name).get_symbols_string()

                atoms_json.append(
                    {
                        'l': kind_string,
                        'x': base_site['position'][0] + shift[0],
                        'y': base_site['position'][1] + shift[1],
                        'z': base_site['position'][2] + shift[2],
                        'atomic_elements_html': atom_kinds_to_html(kind_string),
                    }
                )

        cell_json = {
            't': 'UnitCell',
            'i': 's0',
            'o': (-center).tolist(),
            'x': (lattice_vectors[0] - center).tolist(),
            'y': (lattice_vectors[1] - center).tolist(),
            'z': (lattice_vectors[2] - center).tolist(),
            'xy': (lattice_vectors[0] + lattice_vectors[1] - center).tolist(),
            'xz': (lattice_vectors[0] + lattice_vectors[2] - center).tolist(),
            'yz': (lattice_vectors[1] + lattice_vectors[2] - center).tolist(),
            'xyz': (lattice_vectors[0] + lattice_vectors[1] + lattice_vectors[2] - center).tolist(),
        }

        return_dict = {'s': [cell_json], 'm': [{'a': atoms_json}], 'units': '&Aring;'}

        return json.dumps(return_dict).encode('utf-8'), {}

    def _prepare_xyz(self, main_file_name=''):
        """Write the given structure to a string of format XYZ."""
        if self.is_alloy or self.has_vacancies:
            raise NotImplementedError('XYZ for alloys or systems with vacancies not implemented.')

        sites = self.sites
        cell = self.cell

        return_list = [f'{len(sites)}']
        return_list.append(
            'Lattice="{} {} {} {} {} {} {} {} {}" pbc="{} {} {}"'.format(
                cell[0][0],
                cell[0][1],
                cell[0][2],
                cell[1][0],
                cell[1][1],
                cell[1][2],
                cell[2][0],
                cell[2][1],
                cell[2][2],
                self.pbc[0],
                self.pbc[1],
                self.pbc[2],
            )
        )
        for site in sites:
            # I checked above that it is not an alloy, therefore I take the
            # first symbol
            return_list.append(
                '{:6s} {:18.10f} {:18.10f} {:18.10f}'.format(
                    self.get_kind(site.kind_name).symbols[0], site.position[0], site.position[1], site.position[2]
                )
            )

        return_string = '\n'.join(return_list)
        return return_string.encode('utf-8'), {}

    def _parse_xyz(self, inputstring):
        """Read the structure from a string of format XYZ."""
        from aiida.tools.data.structure import xyz_parser_iterator

        # idiom to get to the last block
        atoms = None
        for _, _, atoms in xyz_parser_iterator(inputstring):
            pass

        if atoms is None:
            raise TypeError('The data does not contain any XYZ data')

        self.clear_kinds()
        self.pbc = (False, False, False)

        for sym, position in atoms:
            self.append_atom(symbols=sym, position=position)

    def _adjust_default_cell(self, vacuum_factor=1.0, vacuum_addition=10.0, pbc=(False, False, False)):
        """If the structure was imported from an xyz file, it lacks a cell.
        This method will adjust the cell
        """
        import numpy as np

        def get_extremas_from_positions(positions):
            """Returns the minimum and maximum value for each dimension in the positions given"""
            return list(zip(*[(min(values), max(values)) for values in zip(*positions)]))

        # Calculating the minimal cell:
        positions = np.array([site.position for site in self.sites])
        position_min, _ = get_extremas_from_positions(positions)

        # Translate the structure to the origin, such that the minimal values in each dimension
        # amount to (0,0,0)
        positions -= position_min
        for index, site in enumerate(self.base.attributes.get('sites')):
            site['position'] = list(positions[index])

        # The orthorhombic cell that (just) accomodates the whole structure is now given by the
        # extremas of position in each dimension:
        minimal_orthorhombic_cell_dimensions = np.array(get_extremas_from_positions(positions)[1])
        minimal_orthorhombic_cell_dimensions = np.dot(vacuum_factor, minimal_orthorhombic_cell_dimensions)
        minimal_orthorhombic_cell_dimensions += vacuum_addition

        # Transform the vector (a, b, c ) to [[a,0,0], [0,b,0], [0,0,c]]
        newcell = np.diag(minimal_orthorhombic_cell_dimensions)
        self.set_cell(newcell.tolist())

        # Now set PBC (checks are done in set_pbc, no need to check anything here)
        self.set_pbc(pbc)

        return self

    def get_description(self):
        """Returns a string with infos retrieved from StructureData node's properties

        :param self: the StructureData node
        :return: retsrt: the description string
        """
        return self.get_formula(mode='hill_compact')
        
    def get_formula(self, mode='hill', separator=''):
        """Return a string with the chemical formula.

        :param mode: a string to specify how to generate the formula, can
            assume one of the following values:

            * 'hill' (default): count the number of atoms of each species,
              then use Hill notation, i.e. alphabetical order with C and H
              first if one or several C atom(s) is (are) present, e.g.
              ``['C','H','H','H','O','C','H','H','H']`` will return ``'C2H6O'``
              ``['S','O','O','H','O','H','O']``  will return ``'H2O4S'``
              From E. A. Hill, J. Am. Chem. Soc., 22 (8), pp 478-494 (1900)

            * 'hill_compact': same as hill but the number of atoms for each
              species is divided by the greatest common divisor of all of them, e.g.
              ``['C','H','H','H','O','C','H','H','H','O','O','O']``
              will return ``'CH3O2'``

            * 'reduce': group repeated symbols e.g.
              ``['Ba', 'Ti', 'O', 'O', 'O', 'Ba', 'Ti', 'O', 'O', 'O',
              'Ba', 'Ti', 'Ti', 'O', 'O', 'O']`` will return ``'BaTiO3BaTiO3BaTi2O3'``

            * 'group': will try to group as much as possible parts of the formula
              e.g.
              ``['Ba', 'Ti', 'O', 'O', 'O', 'Ba', 'Ti', 'O', 'O', 'O',
              'Ba', 'Ti', 'Ti', 'O', 'O', 'O']`` will return ``'(BaTiO3)2BaTi2O3'``

            * 'count': same as hill (i.e. one just counts the number
              of atoms of each species) without the re-ordering (take the
              order of the atomic sites), e.g.
              ``['Ba', 'Ti', 'O', 'O', 'O','Ba', 'Ti', 'O', 'O', 'O']``
              will return ``'Ba2Ti2O6'``

            * 'count_compact': same as count but the number of atoms
              for each species is divided by the greatest common divisor of
              all of them, e.g.
              ``['Ba', 'Ti', 'O', 'O', 'O','Ba', 'Ti', 'O', 'O', 'O']``
              will return ``'BaTiO3'``

        :param separator: a string used to concatenate symbols. Default empty.

        :return: a string with the formula

        .. note:: in modes reduce, group, count and count_compact, the
            initial order in which the atoms were appended by the user is
            used to group and/or order the symbols in the formula
        """
        symbol_list = [s.symbol for s in self.sites]

        return get_formula(symbol_list, mode=mode, separator=separator)

    def get_composition(self, mode='full'):
        """Returns the chemical composition of this structure as a dictionary,
        where each key is the kind symbol (e.g. H, Li, Ba),
        and each value is the number of occurences of that element in this
        structure.

        :param mode: Specify the mode of the composition to return. Choose from ``full``, ``reduced`` or ``fractional``.
            For example, given the structure with formula Ba2Zr2O6, the various modes operate as follows.
            ``full``: The default, the counts are left unnnormalized.
            ``reduced``: The counts are renormalized to the greatest common denominator.
            ``fractional``: The counts are renormalized such that the sum equals 1.

        :returns: a dictionary with the composition
        """
        import numpy as np

        symbols_list = [self.get_kind(s.kind_name).get_symbols_string() for s in self.sites]
        symbols_set = set(symbols_list)

        if mode == 'full':
            return {symbol: symbols_list.count(symbol) for symbol in symbols_set}

        if mode == 'reduced':
            gcd = np.gcd.reduce([symbols_list.count(symbol) for symbol in symbols_set])
            return {symbol: (symbols_list.count(symbol) / gcd) for symbol in symbols_set}

        if mode == 'fractional':
            sum_comp = sum(symbols_list.count(symbol) for symbol in symbols_set)
            return {symbol: symbols_list.count(symbol) / sum_comp for symbol in symbols_set}

        raise ValueError(f'mode `{mode}` is invalid, choose from `full`, `reduced` or `fractional`.')

    def to_ase(self):
        """Get the ASE object.
        Requires to be able to import ase.

        :return: an ASE object corresponding to this
          :py:class:`StructureData <aiida.orm.nodes.data.structure.StructureData>`
          object.

        .. note:: If any site is an alloy or has vacancies, a ValueError
            is raised (from the site.to_ase() routine).
        """
        if not has_ase:
            raise ImportError("The ASE package cannot be imported.")
        
        return self._get_object_ase()

    def to_pymatgen(self, **kwargs):
        """Get pymatgen object. Returns pymatgen Structure for structures with periodic boundary conditions
        (in 1D, 2D, 3D) and Molecule otherwise.
        :param add_spin: True to add the spins to the pymatgen structure.
        Default is False (no spin added).

        .. note:: The spins are set according to the following rule:

            * if the kind name ends with 1 -> spin=+1

            * if the kind name ends with 2 -> spin=-1

        .. note:: Requires the pymatgen module (version >= 3.0.13, usage
            of earlier versions may cause errors).
        """
        if not has_pymatgen:
            raise ImportError("The pymatgen package cannot be imported.")
        
        return self._get_object_pymatgen(**kwargs)
    
    def to_file(self, filename=None, format="cif"):
        
        """Writes the structure to a file.

        Args:
            filename (_type_, optional): defaults to None.
            format (str, optional): defaults to "cif".

        Raises:
            ValueError: should provide a filename different from None.
        """
        if not has_ase:
            raise ImportError("The ASE package cannot be imported.")
        
        if not filename:
            raise ValueError("Please provide a valid filename.")
        
        aseatoms = self.to_ase()
        ase_io.write(filename, aseatoms, format=format)
        
        return 

    def get_pymatgen_structure(self, **kwargs):
        """Get the pymatgen Structure object with any PBC, provided the cell is not singular.
        :param add_spin: True to add the spins to the pymatgen structure.
        Default is False (no spin added).

        .. note:: The spins are set according to the following rule:

            * if the kind name ends with 1 -> spin=+1

            * if the kind name ends with 2 -> spin=-1

        .. note:: Requires the pymatgen module (version >= 3.0.13, usage
            of earlier versions may cause errors).

        :return: a pymatgen Structure object corresponding to this
          :py:class:`StructureData <aiida.orm.nodes.data.structure.StructureData>`
          object.
        :raise ValueError: if the cell is singular, e.g. when it has not been set.
            Use `get_pymatgen_molecule` instead, or set a proper cell.
        """
        return self._get_object_pymatgen_structure(**kwargs)

    def get_pymatgen_molecule(self):
        """Get the pymatgen Molecule object.

        .. note:: Requires the pymatgen module (version >= 3.0.13, usage
            of earlier versions may cause errors).

        :return: a pymatgen Molecule object corresponding to this
          :py:class:`StructureData <aiida.orm.nodes.data.structure.StructureData>`
          object.
        """
        return self._get_object_pymatgen_molecule()


    def append_site(self, site):
        """Append a site to the
        :py:class:`StructureData <aiida.orm.nodes.data.structure.StructureData>`.
        It makes a copy of the site.

        :param site: the site to append. It must be a Site object.
        """
        from aiida.common.exceptions import ModificationNotAllowed

        if self.is_stored:
            raise ModificationNotAllowed('The StructureData object cannot be modified, it has already been stored')

        new_site = site  # So we make a copy

        """if new_site.kind_name not in [_.kind_name for _ in self.sites]:
            raise ValueError(
                f"No kind with name '{site.kind_name}', available kinds are: {[_.kind_name for _ in self.sites]}"
            )
        """

        # If here, no exceptions have been raised, so I add the site.
        self.base.attributes.all.setdefault('sites', []).append(new_site.get_raw())

    def append_atom(self, **kwargs):
        """Append an atom to the Structure, taking care of creating the
        corresponding kind.

        :param ase: the ase Atom object from which we want to create a new atom
                (if present, this must be the only parameter)
        :param position: the position of the atom (three numbers in angstrom)
        :param symbols: passed to the constructor of the Kind object.
        :param weights: passed to the constructor of the Kind object.
        :param charge: passed to the constructor of the charges property
        :param magnetization: passed to the constructor of the magnetization property
        :param name: passed to the constructor of the Kind object. See also the note below.

        .. note :: Note on the 'name' parameter (that is, the name of the kind):

            * if specified, no checks are done on existing species. Simply,
              a new kind with that name is created. If there is a name
              clash, a check is done: if the kinds are identical, no error
              is issued; otherwise, an error is issued because you are trying
              to store two different kinds with the same name.

            * if not specified, the name is automatically generated. Before
              adding the kind, a check is done. If other species with the
              same properties already exist, no new kinds are created, but
              the site is added to the existing (identical) kind.
              (Actually, the first kind that is encountered).
              Otherwise, the name is made unique first, by adding to the string
              containing the list of chemical symbols a number starting from 1,
              until an unique name is found

        .. note :: checks of equality of species are done using
          the :py:meth:`~aiida.orm.nodes.data.structure.Kind.compare_with` method.
        """
                
        aseatom = kwargs.pop('ase', None)
        if aseatom is not None:
            if kwargs:
                raise ValueError(
                    "If you pass 'ase' as a parameter to " 'append_atom, you cannot pass any further' 'parameter'
                )
            position = aseatom.position
            symbol = aseatom.symbol
            kind = symbol + str(aseatom.tag).replace("0","")
            charge = aseatom.charge
            magnetization = aseatom.magmom
            mass = aseatom.mass
        else:
            position = kwargs.pop('position', None)
            if position is None:
                raise ValueError('You have to specify the position of the new atom')
            # all remaining parameters
            symbol = kwargs.pop('symbol', None)
            if symbol is None:
                raise ValueError('You have to specify the symbol of the new atom')
            kind = kwargs.pop('kind', symbol)
            charge = kwargs.pop('charge', 0)
            magnetization = kwargs.pop('magnetization', 0)
            mass = kwargs.pop('mass', _atomic_masses[symbol])
        
        # I look for identical species only if the name is not specified
        #_kinds = self.kinds

        site = Site(# The above code is not complete and seems to be missing the actual code or
        # context. It appears to have a variable `kind_name` declared but without any
        # assignment or usage. If you provide more information or context, I can help
        # explain what the code is intended to do.
        symbol=symbol,
        kind_name=kind, 
        position=position, 
        mass=mass, 
        charge=charge, 
        magnetization=magnetization
        )
        self.append_site(site)

    def clear_kinds(self):
        """Removes all kinds for the StructureData object.

        .. note:: Also clear all sites!
        """
        from aiida.common.exceptions import ModificationNotAllowed

        if self.is_stored:
            raise ModificationNotAllowed('The StructureData object cannot be modified, it has already been stored')

        self.base.attributes.set('kinds', [])
        self._internal_kind_tags = {}
        self.clear_sites()

    def clear_sites(self):
        """Removes all sites for the StructureData object."""
        from aiida.common.exceptions import ModificationNotAllowed

        if self.is_stored:
            raise ModificationNotAllowed('The StructureData object cannot be modified, it has already been stored')

        self.base.attributes.set('sites', [])

    @property
    def sites(self):
        """Returns a list of sites."""
        try:
            raw_sites = self.base.attributes.get('sites')
        except AttributeError:
            raw_sites = []
        return [Site(raw=i) for i in raw_sites]

    @property
    def kinds(self, kind_tags=[], exclude=[], custom_thr={}):
        """Returns a list of kinds."""
        #try:
        #    raw_kinds = self.base.attributes.get('kinds')
        #except AttributeError:
        #    raw_kinds = []
        #return [Kind(raw=i) for i in raw_kinds]
        return set(get_kinds(structure=self, kind_tags=kind_tags, exclude=exclude, custom_thr=custom_thr)["kinds"])
    
    def __getitem__(self, index):
        "ENABLE SLICING. Return a sliced StructureData"
        # Handle slicing
        sliced_structure = self.clone()
        raw_sites = sliced_structure.base.attributes.get('sites')
        if isinstance(index, slice):
            raw_sites = raw_sites[index]
            sliced_structure.base.attributes.set('sites',raw_sites)
            return sliced_structure
        elif isinstance(index, int):
            raw_sites = raw_sites[index]
            sliced_structure.base.attributes.set('sites',raw_sites)
            return sliced_structure
        else:
            raise TypeError(f"Invalid argument type: {type(index)}")
    
    # TOBE deleted  
    def get_kind(self, kind_name):
        """Return the kind object associated with the given kind name.

        :param kind_name: String, the name of the kind you want to get

        :return: The Kind object associated with the given kind_name, if
           a Kind with the given name is present in the structure.

        :raise: ValueError if the kind_name is not present.
        """
        # Cache the kinds, if stored, for efficiency
        if self.is_stored:
            try:
                kinds_dict = self._kinds_cache
            except AttributeError:
                self._kinds_cache = {_.name: _ for _ in self.kinds}
                kinds_dict = self._kinds_cache
        else:
            kinds_dict = {_.kind_name: _ for _ in self.sites}

        # Will raise ValueError if the kind is not present
        try:
            return kinds_dict[kind_name]
        except KeyError:
            raise ValueError(f"Kind name '{kind_name}' unknown")
    
    # TOBE deleted  
    def get_kind_names(self):
        """Return a list of kind names (in the same order of the ``self.kinds``
        property, but return the names rather than Kind objects)

        .. note:: This is NOT necessarily a list of chemical symbols! Use
            get_symbols_set for chemical symbols

        :return: a list of strings.
        """
        return [site.kind_name for site in self.sites]

    @property
    def cell(self) -> t.List[t.List[float]]:
        """Returns the cell shape.

        :return: a 3x3 list of lists.
        """
        return copy.deepcopy(self.base.attributes.get('cell'))

    @cell.setter
    def cell(self, value):
        """Set the cell."""
        self.set_cell(value)

    def set_cell(self, value):
        """Set the cell."""
        from aiida.common.exceptions import ModificationNotAllowed

        if self.is_stored:
            raise ModificationNotAllowed('The StructureData object cannot be modified, it has already been stored')

        the_cell = _get_valid_cell(value)
        self.base.attributes.set('cell', the_cell)

    def reset_cell(self, new_cell):
        """Reset the cell of a structure not yet stored to a new value.

        :param new_cell: list specifying the cell vectors

        :raises:
            ModificationNotAllowed: if object is already stored
        """
        from aiida.common.exceptions import ModificationNotAllowed

        if self.is_stored:
            raise ModificationNotAllowed()

        self.base.attributes.set('cell', new_cell)

    def reset_sites_positions(self, new_positions, conserve_particle=True):
        """Replace all the Site positions attached to the Structure

        :param new_positions: list of (3D) positions for every sites.

        :param conserve_particle: if True, allows the possibility of removing a site.
            currently not implemented.

        :raises aiida.common.ModificationNotAllowed: if object is stored already
        :raises ValueError: if positions are invalid

        .. note:: it is assumed that the order of the new_positions is
            given in the same order of the one it's substituting, i.e. the
            kind of the site will not be checked.
        """
        from aiida.common.exceptions import ModificationNotAllowed

        if self.is_stored:
            raise ModificationNotAllowed()

        if not conserve_particle:
            raise NotImplementedError
        else:
            # test consistency of th enew input
            n_sites = len(self.sites)
            if n_sites != len(new_positions) and conserve_particle:
                raise ValueError('the new positions should be as many as the previous structure.')

            new_sites = []
            for i in range(n_sites):
                try:
                    this_pos = [float(j) for j in new_positions[i]]
                except ValueError:
                    raise ValueError(f'Expecting a list of floats. Found instead {new_positions[i]}')

                if len(this_pos) != 3:
                    raise ValueError(f'Expecting a list of lists of length 3. found instead {len(this_pos)}')

                # now append this Site to the new_site list.
                new_site = Site(site=self.sites[i])  # So we make a copy
                new_site.position = copy.deepcopy(this_pos)
                new_sites.append(new_site)

            # now clear the old sites, and substitute with the new ones
            self.clear_sites()
            for this_new_site in new_sites:
                self.append_site(this_new_site)

    @property
    def pbc1(self):
        return self.base.attributes.get('pbc1')

    @property
    def pbc2(self):
        return self.base.attributes.get('pbc2')

    @property
    def pbc3(self):
        return self.base.attributes.get('pbc3')

    @property
    def pbc(self):
        """Get the periodic boundary conditions.

        :return: a tuple of three booleans, each one tells if there are periodic
            boundary conditions for the i-th real-space direction (i=1,2,3)
        """
        # return copy.deepcopy(self._pbc)
        return (self.base.attributes.get('pbc1'), self.base.attributes.get('pbc2'), self.base.attributes.get('pbc3'))

    @pbc.setter
    def pbc(self, value):
        """Set the periodic boundary conditions."""
        self.set_pbc(value)

    def set_pbc(self, value):
        """Set the periodic boundary conditions."""
        from aiida.common.exceptions import ModificationNotAllowed

        if self.is_stored:
            raise ModificationNotAllowed('The StructureData object cannot be modified, it has already been stored')
        the_pbc = get_valid_pbc(value)

        # self._pbc = the_pbc
        self.base.attributes.set('pbc1', the_pbc[0])
        self.base.attributes.set('pbc2', the_pbc[1])
        self.base.attributes.set('pbc3', the_pbc[2])

    @property
    def cell_lengths(self):
        """Get the lengths of cell lattice vectors in angstroms."""
        import numpy

        cell = self.cell
        return [
            numpy.linalg.norm(cell[0]),
            numpy.linalg.norm(cell[1]),
            numpy.linalg.norm(cell[2]),
        ]

    @cell_lengths.setter
    def cell_lengths(self, value):
        self.set_cell_lengths(value)

    def set_cell_lengths(self, value):
        raise NotImplementedError('Modification is not implemented yet')

    @property
    def cell_angles(self):
        """Get the angles between the cell lattice vectors in degrees."""
        import numpy

        cell = self.cell
        lengths = self.cell_lengths
        return [
            float(numpy.arccos(x) / numpy.pi * 180)
            for x in [
                numpy.vdot(cell[1], cell[2]) / lengths[1] / lengths[2],
                numpy.vdot(cell[0], cell[2]) / lengths[0] / lengths[2],
                numpy.vdot(cell[0], cell[1]) / lengths[0] / lengths[1],
            ]
        ]

    @cell_angles.setter
    def cell_angles(self, value):
        self.set_cell_angles(value)

    def set_cell_angles(self, value):
        raise NotImplementedError('Modification is not implemented yet')

    @property
    def is_alloy(self):
        """Return whether the structure contains any alloy kinds.

        :return: a boolean, True if at least one kind is an alloy
        """
        return any(kind.is_alloy for kind in self.kinds)

    @property
    def has_vacancies(self):
        """Return whether the structure has vacancies in the structure.

        :return: a boolean, True if at least one kind has a vacancy
        """
        return any(kind.has_vacancies for kind in self.kinds)

    def get_cell_volume(self):
        """Returns the three-dimensional cell volume in Angstrom^3.

        Use the `get_dimensionality` method in order to get the area/length of lower-dimensional cells.

        :return: a float.
        """
        return calc_cell_volume(self.cell)

    def get_cif(self, converter='ase', store=False, **kwargs):
        """Creates :py:class:`aiida.orm.nodes.data.cif.CifData`.

        :param converter: specify the converter. Default 'ase'.
        :param store: If True, intermediate calculation gets stored in the
            AiiDA database for record. Default False.
        :return: :py:class:`aiida.orm.nodes.data.cif.CifData` node.
        """
        from aiida.tools.data import structure as structure_tools

        param = Dict(kwargs)
        try:
            conv_f = getattr(structure_tools, f'_get_cif_{converter}_inline')
        except AttributeError:
            raise ValueError(f"No such converter '{converter}' available")
        ret_dict = conv_f(struct=self, parameters=param, metadata={'store_provenance': store})
        return ret_dict['cif']

    def _get_object_phonopyatoms(self):
        """Converts StructureData to PhonopyAtoms

        :return: a PhonopyAtoms object
        """
        from phonopy.structure.atoms import PhonopyAtoms

        atoms = PhonopyAtoms(symbols=[_.kind_name for _ in self.sites])
        # Phonopy internally uses scaled positions, so you must store cell first!
        atoms.set_cell(self.cell)
        atoms.set_positions([_.position for _ in self.sites])

        return atoms

    def _get_object_ase(self):
        """Converts
        :py:class:`StructureData <aiida.orm.nodes.data.structure.StructureData>`
        to ase.Atoms

        :return: an ase.Atoms object
        """
        import ase

        asecell = ase.Atoms(cell=self.cell, pbc=self.pbc)
        _kinds = self.get_kind_names()

        for site in self.sites:
            asecell.append(site.get_ase(kinds=_kinds))
            
        #asecell.set_initial_charges(self.get_site_property("charge"))
          
        return asecell

    def _get_object_pymatgen(self, **kwargs):
        """Converts
        :py:class:`StructureData <aiida.orm.nodes.data.structure.StructureData>`
        to pymatgen object

        :return: a pymatgen Structure for structures with periodic boundary
            conditions (in three dimensions) and Molecule otherwise

        .. note:: Requires the pymatgen module (version >= 3.0.13, usage
            of earlier versions may cause errors).
        """
        if any(self.pbc):
            return self._get_object_pymatgen_structure(**kwargs)

        return self._get_object_pymatgen_molecule(**kwargs)

    def _get_object_pymatgen_structure(self, **kwargs):
        """Converts
        :py:class:`StructureData <aiida.orm.nodes.data.structure.StructureData>`
        to pymatgen Structure object
        :param add_spin: True to add the spins to the pymatgen structure.
        Default is False (no spin added).

        .. note:: The spins are set according to the following rule:

            * if the kind name ends with 1 -> spin=+1

            * if the kind name ends with 2 -> spin=-1

        :return: a pymatgen Structure object corresponding to this
          :py:class:`StructureData <aiida.orm.nodes.data.structure.StructureData>`
          object
        :raise ValueError: if the cell is not set (i.e. is the default one);
          if there are partial occupancies together with spins
          (defined by kind names ending with '1' or '2').

        .. note:: Requires the pymatgen module (version >= 3.0.13, usage
            of earlier versions may cause errors)
        """
        from pymatgen.core.lattice import Lattice
        from pymatgen.core.structure import Structure

        species = []
        additional_kwargs = {}

        lattice = Lattice(matrix=self.cell, pbc=self.pbc)

        if kwargs.pop('add_spin', False) and any(n.endswith('1') or n.endswith('2') for n in self.get_kind_names()):
            # case when spins are defined -> no partial occupancy allowed
            from pymatgen.core.periodic_table import Specie

            oxidation_state = 0  # now I always set the oxidation_state to zero
            for site in self.sites:
                kind = site.kind_name
                if len(kind.symbols) != 1 or (len(kind.weights) != 1 or sum(kind.weights) < 1.0):
                    raise ValueError('Cannot set partial occupancies and spins at the same time')
                spin = -1 if site.kind_name.endswith('1') else 1 if site.kind_name.endswith('2') else 0
                try:
                    specie = Specie(kind.symbols[0], oxidation_state, properties={'spin': spin})
                except TypeError:
                    # As of v2023.9.2, the ``properties`` argument is removed and the ``spin`` argument should be used.
                    # See: https://github.com/materialsproject/pymatgen/commit/118c245d6082fe0b13e19d348fc1db9c0d512019
                    # The ``spin`` argument was introduced in v2023.6.28.
                    # See: https://github.com/materialsproject/pymatgen/commit/9f2b3939af45d5129e0778d371d814811924aeb6
                    specie = Specie(kind.symbols[0], oxidation_state, spin=spin)
                species.append(specie)
        else:
            # case when no spin are defined
            for site in self.sites:
                kind = self.get_kind(site.kind_name)
                species.append(dict(zip(kind.symbols, kind.weights)))
            if any(
                create_automatic_kind_name(self.get_kind(name).symbols, self.get_kind(name).weights) != name
                for name in self.get_site_property("kind_name")
            ):
                # add "kind_name" as a properties to each site, whenever
                # the kind_name cannot be automatically obtained from the symbols
                additional_kwargs['site_properties'] = {'kind_name': self.get_site_property("kind_name")}

        if kwargs:
            raise ValueError(f'Unrecognized parameters passed to pymatgen converter: {kwargs.keys()}')

        positions = [list(x.position) for x in self.sites]

        try:
            return Structure(lattice, species, positions, coords_are_cartesian=True, **additional_kwargs)
        except ValueError as err:
            raise ValueError('Singular cell detected. Probably the cell was not set?') from err

    def _get_object_pymatgen_molecule(self, **kwargs):
        """Converts
        :py:class:`StructureData <aiida.orm.nodes.data.structure.StructureData>`
        to pymatgen Molecule object

        :return: a pymatgen Molecule object corresponding to this
          :py:class:`StructureData <aiida.orm.nodes.data.structure.StructureData>`
          object.

        .. note:: Requires the pymatgen module (version >= 3.0.13, usage
            of earlier versions may cause errors)
        """
        from pymatgen.core.structure import Molecule

        if kwargs:
            raise ValueError(f'Unrecognized parameters passed to pymatgen converter: {kwargs.keys()}')

        species = []
        for site in self.sites:
            if hasattr(site,"weight"):
                weight = site.weight
            else:
                weight = 1
            species.append({site.symbol: weight})

        positions = [list(site.position) for site in self.sites]
        return Molecule(species, positions)
    
    def _get_dimensionality(self,):
        """Return the dimensionality of the structure and its length/surface/volume.

        Zero-dimensional structures are assigned "volume" 0.

        :return: returns a dictionary with keys "dim" (dimensionality integer), "label" (dimensionality label)
            and "value" (numerical length/surface/volume).
        """
        import numpy as np

        retdict = {}

        pbc = np.array(self.pbc)
        cell = np.array(self.cell)

        dim = len(pbc[pbc])

        retdict['dim'] = dim
        retdict['label'] = StructureData._dimensionality_label[dim]

        if dim not in (0, 1, 2, 3):
            raise ValueError(f'Dimensionality {dim} must be one of 0, 1, 2, 3')

        if dim == 0:
            # We have no concept of 0d volume. Let's return a value of 0 for a consistent output dictionary
            retdict['value'] = 0
        elif dim == 1:
            retdict['value'] = np.linalg.norm(cell[pbc])
        elif dim == 2:
            vectors = cell[pbc]
            retdict['value'] = np.linalg.norm(np.cross(vectors[0], vectors[1]))
        elif dim == 3:
            retdict['value'] = calc_cell_volume(cell)

        return retdict

    def _validate_dimensionality(self,):
        """Check whether the given pbc and cell vectors are consistent."""
        dim = self._get_dimensionality()

        # 0-d structures put no constraints on the cell
        if dim['dim'] == 0:
            return

        # finite-d structures should have a cell with finite volume
        if dim['value'] == 0:
            raise ValueError(f'Structure has periodicity {self.pbc} but {dim["dim"]}-d volume 0.')

        return