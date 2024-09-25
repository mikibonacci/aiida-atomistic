import copy
import json
import typing as t
import numpy as np

from aiida import orm
from aiida.common.constants import elements

from aiida_atomistic.data.structure.site import SiteImmutable as Site
from aiida_atomistic.data.structure.models import MutableStructureModel

try:
    import ase  # noqa: F401
    from ase import io as ase_io

    has_ase = True
    ASE_ATOMS_TYPE = ase.Atoms
except ImportError:
    has_ase = False

    ASE_ATOMS_TYPE = t.Any

try:
    import pymatgen.core as core  # noqa: F401

    has_pymatgen = True
    PYMATGEN_MOLECULE = core.structure.Molecule
    PYMATGEN_STRUCTURE = core.structure.Structure
except ImportError:
    has_pymatgen = False

    PYMATGEN_MOLECULE = t.Any
    PYMATGEN_STRUCTURE = t.Any

_MASS_THRESHOLD = 1.0e-3
# Threshold to check if the sum is one or not
_SUM_THRESHOLD = 1.0e-6
# Default cell
_DEFAULT_CELL = ((0, 0, 0),) * 3

_DEFAULT_PROPERTIES = {
            'pbc',
            'cell',
            'sites',
            'kinds',
            'masses',
            'symbols',
            'positions',
            'weights',
        }

_valid_symbols = tuple(i["symbol"] for i in elements.values())
_atomic_masses = {el["symbol"]: el["mass"] for el in elements.values()}
_atomic_numbers = {data["symbol"]: num for num, data in elements.items()}

_default_values = {
    "charges": 0,
    "magmoms": [0, 0, 0],
}

class GetterMixin:

    @property
    def is_alloy(self):
        return any(_.is_alloy for _ in self.properties.sites)

    @property
    def has_vacancies(self):
        return any(_.has_vacancies for _ in self.properties.sites)

    @property
    def is_collinear(self):
        if "magmoms" not in self.get_defined_properties()["computed"]:
            return False
        namelist = {"starting_magnetization":{},"angle1":{},"angle2":{}}
        for site in self.properties.sites:
            for variable in namelist.keys():
                namelist[variable][site.kinds] = site.get_magmom_coord(coord="spherical")[variable]
        return len(set([namelist["angle1"][site.kinds] for site in self.properties.sites])) == 1 and \
            len(set([namelist["angle2"][site.kinds] for site in self.properties.sites])) == 1

    @classmethod
    def from_ase(
        cls,
        aseatoms: ASE_ATOMS_TYPE,
        detect_kinds: bool = True):
        """Load the structure from a ASE object"""

        if not has_ase:
            raise ImportError("The ASE package cannot be imported.")

        # Read the ase structure
        data = {}
        data["cell"] = aseatoms.cell.array.tolist()
        data["pbc"] = aseatoms.pbc.tolist()

        data["sites"] = []
        # self.clear_kinds()  # This also calls clear_sites
        for atom in aseatoms:
            new_site = Site.atom_to_site(aseatom=atom)
            data["sites"].append(new_site.model_dump())

        structure = cls(**data)

        if detect_kinds:
            data["sites"] = structure.get_kinds(ready_to_use=True)
            data["kinds"] = [
                site["kinds"] for site in data["sites"]
            ]

        structure = cls(**data)

        return structure

    @classmethod
    def from_file(
        cls,
        filename,
        format="cif",
        detect_kinds: bool = True,
        **kwargs):
        """Load the structure from a file"""

        ase_read = ase_io.read(filename, format=format, **kwargs)

        return cls.from_ase(aseatoms=ase_read, detect_kinds=detect_kinds)

    @classmethod
    def from_pymatgen(
        cls,
        pymatgen_obj: t.Union[PYMATGEN_MOLECULE, PYMATGEN_STRUCTURE],
        detect_kinds: bool = True,
        **kwargs,
    ):
        """Load the structure from a pymatgen object.

        .. note:: Requires the pymatgen module (version >= 3.0.13, usage
            of earlier versions may cause errors).
        """
        if not has_pymatgen:
            raise ImportError("The pymatgen package cannot be imported.")

        if isinstance(pymatgen_obj, PYMATGEN_MOLECULE):
            structure = cls._from_pymatgen_molecule(pymatgen_obj, detect_kinds=detect_kinds)
        else:
            structure = cls._from_pymatgen_structure(pymatgen_obj, detect_kinds=detect_kinds)

        return structure

    @classmethod
    def _from_pymatgen_molecule(
        cls,
        mol: PYMATGEN_MOLECULE,
        margin=5,
        detect_kinds: bool = True,
        ):
        """Load the structure from a pymatgen Molecule object.

        :param margin: the margin to be added in all directions of the
            bounding box of the molecule.

        .. note:: Requires the pymatgen module (version >= 3.0.13, usage
            of earlier versions may cause errors).
        """
        box = [
            max(x.coords.tolist()[0] for x in mol.properties.sites)
            - min(x.coords.tolist()[0] for x in mol.properties.sites)
            + 2 * margin,
            max(x.coords.tolist()[1] for x in mol.properties.sites)
            - min(x.coords.tolist()[1] for x in mol.properties.sites)
            + 2 * margin,
            max(x.coords.tolist()[2] for x in mol.properties.sites)
            - min(x.coords.tolist()[2] for x in mol.properties.sites)
            + 2 * margin,
        ]
        structure = cls._from_pymatgen_structure(mol.get_boxed_structure(*box), detect_kinds=detect_kinds)
        structure.properties.pbc = [False, False, False]

        return structure

    @classmethod
    def _from_pymatgen_structure(
        cls,
        struct: PYMATGEN_STRUCTURE,
        detect_kinds: bool = True,
        ):
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
            has_spin_attribute = hasattr(species[0], "_spin")

            if has_spin_attribute:
                has_spin = any(specie.spin != 0 for specie in species)
            else:
                has_spin = any(
                    specie.as_dict().get("properties", {}).get("spin", 0) != 0
                    for specie in species
                )

            has_partial_occupancies = len(occupations) != 1 or occupations[0] != 1.0

            if has_partial_occupancies and has_spin:
                raise ValueError(
                    "Cannot set partial occupancies and spins at the same time"
                )

            if has_spin:
                from aiida_atomistic.data.structure.utils import create_automatic_kind_name
                symbols = [specie.symbol for specie in species]
                kind_name = create_automatic_kind_name(symbols, occupations)

                # If there is spin, we can only have a single specie, otherwise we would have raised above
                specie = species[0]
                if has_spin_attribute:
                    spin = specie.spin
                else:
                    spin = specie.as_dict().get("properties", {}).get("spin", 0)

                if spin < 0:
                    kind_name += "1"
                else:
                    kind_name += "2"

                return kind_name

            return None

        inputs = {}
        inputs["cell"] = struct.lattice.matrix.tolist()
        inputs["pbc"] = [True, True, True]
        # self.clear_kinds()

        inputs["sites"] = []
        sites_collection = struct.properties["sites"] if "sites" in struct.properties.keys() else struct.sites
        for site in sites_collection:

            if "kind_name" in site.properties:
                kind_name = site.properties["kinds"]
            else:
                kind_name = site.label

            site_info = {
                "symbol": site.specie.symbol,
                "mass": site.species.weight,
                "position": site.coords.tolist(),
                "charge": site.properties.get("charge", 0.0),
                'magmom': site.properties.get("magmom").moment if "magmom" in site.properties.keys() else [0,0,0]
            }

            if kind_name is not None:
                site_info["kinds"] = kind_name.replace("+", "").replace("-", "")

            inputs["sites"].append(site_info)

        structure = cls(**inputs)

        if detect_kinds:
            inputs["sites"] = structure.get_kinds(ready_to_use=True)
            inputs["kinds"] = [
                site["kinds"] for site in inputs["sites"]
            ]

        structure = cls(**inputs)

        return structure

    def to_dict(
            self,
            detect_kinds: bool = False
        ):
            """
            Convert the structure to a dictionary representation.

            :param detect_kinds: Whether to detect and include the kinds of the structure.
            :type detect_kinds: bool, optional
            :return: The structure as a dictionary.
            :rtype: dict
            """
            dict_repr = copy.deepcopy(self.properties.model_dump())

            if detect_kinds:
                dict_repr["sites"] = self.get_kinds(ready_to_use=True)
                dict_repr["kinds"] = [
                    site["kinds"] for site in dict_repr["sites"]
                ]

            # dict_repr = get_serialized_data(dict_repr)

            return dict_repr

    def get_site_property(self, property_name):
        """Return a list with length equal to the number of sites of this structure,
        where each element of the list is the property of the corresponding site.

        :return: a list of floats
        """
        return np.array([getattr(this_site, property_name) for this_site in self.properties.sites])

    @classmethod
    def get_supported_properties(cls, for_plugin_check=False):
        """Get a list of properties.

        Args:
            for_plugin_check (bool, optional): If True, only include properties that need to be checked for a plugin. Defaults to False.
        """

        properties = {'direct': set(MutableStructureModel.model_fields.keys()), 'computed': set(MutableStructureModel.model_computed_fields.keys()), 'sites': set(Site.model_fields.keys())}

        # in the following `default_properties` we don't put also the `computed`` ones. They depends on the other properties (`direct` and `site`).


        if for_plugin_check:
            for domain in ["direct","sites"]:
                properties[domain] = set(properties[domain]).difference(_DEFAULT_PROPERTIES)

        return properties

    def check_plugin_support(self, plugin_properties):
        """
        Check if the plugin supports the given properties.
        :param plugin_properties: The supported properties in the plugin.
        :return: The properties supported by the plugin but not excluded by the mixin.
        :rtype: set
        """

        excluded_properties = self.get_supported_properties(for_plugin_check=True)
        defined_properties = self.get_defined_properties()

        excluded_properties = set(excluded_properties["direct"]).union(excluded_properties["sites"])
        defined_properties = set(defined_properties["direct"]).union(defined_properties["sites"]).difference(_DEFAULT_PROPERTIES)

        return defined_properties.difference(plugin_properties)

    def get_charges(self,):
        return self.get_site_property("charge")

    def get_magmoms(self,):
        return self.get_site_property("magmom")

    def get_kind_names(self,):
        return self.get_site_property("kinds")

    def get_positions(self,):
        return self.get_site_property("position")

    def get_symbols(self,):
        return self.get_site_property("symbol")

    def get_cell_volume(self):
        """Returns the three-dimensional cell volume in Angstrom^3.

        Use the `get_dimensionality` method in order to get the area/length of lower-dimensional cells.

        :return: a float.
        """
        from aiida_atomistic.data.structure.utils import calc_cell_volume
        return calc_cell_volume(self.properties.cell)

    def get_cif(self, converter="ase", store=False, **kwargs):
        """Creates :py:class:`aiida.orm.nodes.data.cif.CifData`.

        :param converter: specify the converter. Default 'ase'.
        :param store: If True, intermediate calculation gets stored in the
            AiiDA database for record. Default False.
        :return: :py:class:`aiida.orm.nodes.data.cif.CifData` node.
        """
        from aiida.tools.data import structure as structure_tools

        param = orm.Dict(kwargs)
        try:
            conv_f = getattr(structure_tools, f"_get_cif_{converter}_inline")
        except AttributeError:
            raise ValueError(f"No such converter '{converter}' available")
        ret_dict = conv_f(
            struct=self, parameters=param, metadata={"store_provenance": store}
        )
        return ret_dict["cif"]

    def get_description(self):
        """Returns a string with infos retrieved from StructureData node's properties

        :param self: the StructureData node
        :return: retsrt: the description string
        """
        return self.get_formula(mode="hill_compact")

    def get_formula(self, mode="hill", separator=""):
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
        from aiida_atomistic.data.structure.utils import get_formula
        symbol_list = [s.symbol for s in self.properties.sites]

        return get_formula(symbol_list, mode=mode, separator=separator)

    def get_composition(self, mode="full"):
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

        symbols_list = self.properties.symbols

        symbols_set = set(symbols_list)

        if mode == "full":
            return {symbol: symbols_list.count(symbol) for symbol in symbols_set}

        if mode == "reduced":
            gcd = np.gcd.reduce([symbols_list.count(symbol) for symbol in symbols_set])
            return {
                symbol: (symbols_list.count(symbol) / gcd) for symbol in symbols_set
            }

        if mode == "fractional":
            sum_comp = sum(symbols_list.count(symbol) for symbol in symbols_set)
            return {
                symbol: symbols_list.count(symbol) / sum_comp for symbol in symbols_set
            }

        raise ValueError(
            f"mode `{mode}` is invalid, choose from `full`, `reduced` or `fractional`."
        )

    def get_kinds(self, kind_tags=[], exclude=["weights"], custom_thr={}, ready_to_use=False):
        """
        Get the list of kinds, taking into account all the properties.
        If the list of kinds is already provided--> len(kind_tags)>0, we check the consistency of it
        by computing the kinds with threshold=0 for each property.


        Algorithm:
        it generated the kinds_list for each property separately in Step 1, then
        it creates the matrix k = k.T where the rows are the sites, the columns are the properties and each element
        is the corresponding kind for the given property and the given site:

        ```bash
                p1 p2 p3
        site1 = | 1  1  2 | = kind1
        site2 = | 1  2  3 | = kind2
        site3 = | 2  2  3 | = kind3
        site4 = | 1  2  3 | = kind4
        ```

        In Step 2 it checks for the matrix which rows have the same numbers in the same order, i.e. recognize the different
        kinds considering all the properties. This is done by subtracting a row from the others and see if all the elements
        are zero, meaning that we have the same combination of kinds.

        In Step 3 we override the kinds with the kind_tags.

        Args:
            kind_tags (list, optional): list of kind names as user defined: in principle this input trigger a check in the kind
                                        determination -> the mapping should be the same as obtained with get_kinds(kind_tags=[],) and
                                        all thresholds = 0. And this is what is done: `if not None in kind_tags: thr = 0`.
                                        For now we support also for only some selected kinds defined: ["kind1",None, ...] but with the same length as the symbols (sites).
            exclude (list, optional): list of properties to be excluded in the kind determination
            custom_thr (dict, options): dictionary with the custom threshold for given properties (key: property, value: thr).
                                        if not provided, we fallback into the default threshold define in the property class.

        Returns:
            kinds_dictionary (dictionary): the associated per-site (and per-kind) value of the property. The structure of the dictionary is the one that you may
                                        have in the `properties` dictionary input of the StructureData constructor.
                                        We also provide the `kinds` property: list of kind-per-site to be used in a plugin which requires it. If kind tags are all decided, then we
                                        do not compute anything and we return kind_tags and None. In this way, we know that we basically already defined
                                        the kinds in our StructureData.

        Comments:

        - Implementation can and should be improved, but the functionalities are the desired ones.
        - Moreover, the method should be accessible to run on a given properties dictionary, so to predict the kinds before the StructureData instance generation.
        """

        # cannot do properties.symbols.value due to recursion problem if called in Kinds:
        # if I call properties, this will again reinitialize the properties attribute and so on.
        # should be this:
        # symbols = self.base.attributes.get("_property_attributes")['symbols']['value']
        # However, for now I do not let the kinds to be automatically generated when we initialise the structure:
        symbols = self.get_site_property("symbol")
        default_thresholds = {
            "charge": 0.1,
            "mass": 1e-4,
            "magmom": 1e-4, # _MAGMOM_THRESHOLD
        }

        list_tags = []
        if len(kind_tags) == 0:
            kind_tags = [None] * len(
                symbols
            )  # <== For now we support also for only ... see above doc string.
            check_kinds = False
            # kind=tags = self.properties.kinds.value
        else:
            list_tags = [kind_tags.index(n) for n in kind_tags]
            check_kinds = True

        array_tags = np.array(list_tags)

        # Step 1:
        kind_properties = []
        kinds_dictionary = {"kinds": {}}
        for single_property in self.properties.sites[0].model_dump().keys():
            if single_property not in ["symbol", "position", "kinds",] + exclude:
                #prop = self.get_site_property(single_property)
                thr = custom_thr.get(
                    single_property, default_thresholds.get(single_property)
                )
                kinds_dictionary[single_property] = {}

                kinds_per_property = self._to_kinds(
                    property_name=single_property, symbols=symbols, thr=thr
                )

                kind_properties.append(kinds_per_property[0])
                # I prefer to store again under the key 'value', may be useful in the future
                kinds_dictionary[single_property] = kinds_per_property[1]

        k = np.array(kind_properties)
        k = k.T

        # Step 2:
        # kinds = np.zeros(len(self.get_site_property("symbol")), dtype=int) - 1
        check_array = np.zeros(len(self.get_site_property("position")), dtype=int) -1
        kind_names = symbols.tolist()
        kind_numeration = np.zeros_like(check_array, dtype=int)
        for i in range(len(k)):
            # Goes from the first symbol... so the numbers will be from zero to N (Please note: the symbol does not matter: Li0, Cu1... not Li0, Cu0.)
            element = symbols[i]
            diff = k - k[i]
            diff_sum = np.sum(np.abs(diff), axis=1)

            # kinds[np.where(diff_sum == 0)[0]] = i
            for where in np.where(diff_sum == 0)[0]:
                if not check_array[where] == -1:
                    continue
                if (
                    f"{element}{i}" in kind_tags
                ):  # If I encounter the same tag as provided as input or generated here:
                    kind_numeration[where] = i + len(k)
                else:
                    kind_numeration[where] = i

                kind_names[where] = f"{element}{kind_numeration[where]}"

                check_array[where] = i
                #print(f"site {where} is {element}{kind_numeration[-1]}")

            if len(np.where(check_array == -1)[0]) == 0:
                #print(f"search ended at iteration {i}")
                break

        # Step 3:
        kinds_dictionary["kinds"] = [
            kind_names[i]  if not kind_tags[i] else kind_tags[i]
            for i in range(len(kind_tags))
        ]

        kinds_dictionary["index"] = kind_numeration
        kinds_dictionary["symbol"] = symbols.tolist()
        kinds_dictionary["position"] = self.get_site_property("position").tolist()

        # Step 4: check on the kind_tags consistency with the properties value.
        if check_kinds and not np.array_equal(check_array, array_tags):
            raise ValueError(
                "The kinds you provided in the `kind_tags` input are not correct, as properties values are not consistent with them. Please check that this is what you want."
            )

        '''if ready_to_use:
            new_sites = []
            for index_kind in kinds_dictionary["index"]:
                dict_site = {}
                for k,v in kinds_dictionary.items():
                    if k not in ["symbol","position","index"]:
                        dict_site[k] = v[index_kind].tolist() if isinstance(v[index_kind], np.ndarray) else v[index_kind]
                for value in ["symbol","position"]:
                    dict_site[value] = kinds_dictionary[value][index_kind]
                new_sites.append(dict_site)
            return new_sites
        '''
        if ready_to_use:
            new_sites = []
            for index_global, index_kind in enumerate(kinds_dictionary["index"]):
                dict_site = {}
                for k,v in kinds_dictionary.items():
                    if k not in ["symbol","position","index"]:
                        dict_site[k] = v[index_kind].tolist() if isinstance(v[index_kind], np.ndarray) else v[index_kind]
                for value in ["symbol","position"]:
                    # even for same kind, the position should be different
                    dict_site[value] = kinds_dictionary[value][index_global]
                new_sites.append(dict_site)
            return new_sites

        return kinds_dictionary

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

    '''def to_legacy(self) -> LegacyStructureData:

        """
        Returns: orm.StructureData object, used for backward compatibility.
        """
        if not has_ase:
            raise ImportError("The ASE package cannot be imported.")

        aseatoms = self.to_ase()

        return LegacyStructureData(ase=aseatoms)
    '''

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

    def _validate(self):
        """Performs some standard validation tests."""
        from aiida.common.exceptions import ValidationError
        from aiida_atomistic.data.structure.utils import _get_valid_cell, _get_valid_pbc

        super()._validate()

        try:
            _get_valid_cell(self.properties.cell)
        except ValueError as exc:
            raise ValidationError(f"Invalid cell: {exc}")

        try:
            _get_valid_pbc(self.properties.pbc)
        except ValueError as exc:
            raise ValidationError(f"Invalid periodic boundary conditions: {exc}")

        self._validate_dimensionality()

        try:
            # This will try to create the kinds objects
            kinds = set(self.get_site_property("kinds"))
        except ValueError as exc:
            raise ValidationError(f"Unable to validate the kinds: {exc}")

        from collections import Counter

        counts = Counter([k for k in kinds])
        for count in counts:
            if counts[count] != 1:
                raise ValidationError(
                    f"Kind with name '{count}' appears {counts[count]} times instead of only one"
                )

        try:
            # This will try to create the sites objects
            sites = self.properties.sites
        except ValueError as exc:
            raise ValidationError(f"Unable to validate the sites: {exc}")

        for site in sites:
            if site.kinds not in kinds:
                raise ValidationError(
                    f"A site has kind {site.kinds}, but no specie with that name exists"
                )

        kinds_without_sites = {k for k in kinds} - {s.kinds for s in sites}
        if kinds_without_sites:
            raise ValidationError(
                f"The following kinds are defined, but there are no sites with that kind: {list(kinds_without_sites)}"
            )

    def _prepare_xsf(self, main_file_name=""):
        """Write the given structure to a string of format XSF (for XCrySDen)."""
        if self.is_alloy or self.has_vacancies:
            raise NotImplementedError(
                "XSF for alloys or systems with vacancies not implemented."
            )

        sites = self.properties.sites

        return_string = "CRYSTAL\nPRIMVEC 1\n"
        for cell_vector in self.properties.cell:
            return_string += " ".join([f"{i:18.10f}" for i in cell_vector])
            return_string += "\n"
        return_string += "PRIMCOORD 1\n"
        return_string += f"{int(len(sites))} 1\n"
        for site in sites:
            # I checked above that it is not an alloy, therefore I take the
            # first symbol
            return_string += (
                f"{_atomic_numbers[self.get_kind(site.kinds).symbols[0]]} "
            )
            return_string += "%18.10f %18.10f %18.10f\n" % tuple(site.position)
        return return_string.encode("utf-8"), {}

    def _prepare_cif(self, main_file_name=""):
        """Write the given structure to a string of format CIF."""
        from aiida.orm import CifData

        cif = CifData(ase=self.to_ase())
        return cif._prepare_cif()

    def _prepare_chemdoodle(self, main_file_name=""):
        """Write the given structure to a string of format required by ChemDoodle."""
        from itertools import product
        from aiida_atomistic.data.structure.utils import atom_kinds_to_html

        import numpy as np

        supercell_factors = [1, 1, 1]

        # Get cell vectors and atomic position
        lattice_vectors = np.array(self.base.attributes.get("cell"))
        base_sites = self.base.attributes.get("sites")

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
                shift = (
                    ix * lattice_vectors[0]
                    + iy * lattice_vectors[1]
                    + iz * lattice_vectors[2]
                    - center
                ).tolist()

                kind_name = base_site["kinds"]
                kind_string = self.get_kind(kind_name).get_symbols_string()

                atoms_json.append(
                    {
                        "l": kind_string,
                        "x": base_site["position"][0] + shift[0],
                        "y": base_site["position"][1] + shift[1],
                        "z": base_site["position"][2] + shift[2],
                        "atomic_elements_html": atom_kinds_to_html(kind_string),
                    }
                )

        cell_json = {
            "t": "UnitCell",
            "i": "s0",
            "o": (-center).tolist(),
            "x": (lattice_vectors[0] - center).tolist(),
            "y": (lattice_vectors[1] - center).tolist(),
            "z": (lattice_vectors[2] - center).tolist(),
            "xy": (lattice_vectors[0] + lattice_vectors[1] - center).tolist(),
            "xz": (lattice_vectors[0] + lattice_vectors[2] - center).tolist(),
            "yz": (lattice_vectors[1] + lattice_vectors[2] - center).tolist(),
            "xyz": (
                lattice_vectors[0] + lattice_vectors[1] + lattice_vectors[2] - center
            ).tolist(),
        }

        return_dict = {"s": [cell_json], "m": [{"a": atoms_json}], "units": "&Aring;"}

        return json.dumps(return_dict).encode("utf-8"), {}

    def _prepare_xyz(self, main_file_name=""):
        """Write the given structure to a string of format XYZ."""
        if self.is_alloy or self.has_vacancies:
            raise NotImplementedError(
                "XYZ for alloys or systems with vacancies not implemented."
            )

        sites = self.properties.sites
        cell = self.properties.cell

        return_list = [f"{len(sites)}"]
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
                self.properties.pbc[0],
                self.properties.pbc[1],
                self.properties.pbc[2],
            )
        )
        for site in sites:
            # I checked above that it is not an alloy, therefore I take the
            # first symbol
            return_list.append(
                "{:6s} {:18.10f} {:18.10f} {:18.10f}".format(
                    self.get_kind(site.kinds).symbols[0],
                    site.position[0],
                    site.position[1],
                    site.position[2],
                )
            )

        return_string = "\n".join(return_list)
        return return_string.encode("utf-8"), {}

    def _parse_xyz(self, inputstring):
        """Read the structure from a string of format XYZ."""
        from aiida.tools.data.structure import xyz_parser_iterator

        # idiom to get to the last block
        atoms = None
        for _, _, atoms in xyz_parser_iterator(inputstring):
            pass

        if atoms is None:
            raise TypeError("The data does not contain any XYZ data")

        #self.clear_kinds()
        self.properties.pbc = (False, False, False)

        for sym, position in atoms:
            self.add_atom(atom_info={'symbol':sym, 'position':position})

    def _adjust_default_cell(
        self, vacuum_factor=1.0, vacuum_addition=10.0, pbc=(False, False, False)
    ):
        """If the structure was imported from an xyz file, it lacks a cell.
        This method will adjust the cell
        """
        import numpy as np

        def get_extremas_from_positions(positions):
            """Returns the minimum and maximum value for each dimension in the positions given"""
            return list(
                zip(*[(min(values), max(values)) for values in zip(*positions)])
            )

        # Calculating the minimal cell:
        positions = np.array([site.position for site in self.properties.sites])
        position_min, _ = get_extremas_from_positions(positions)

        # Translate the structure to the origin, such that the minimal values in each dimension
        # amount to (0,0,0)
        positions -= position_min
        for index, site in enumerate(self.base.attributes.get("sites")):
            site["position"] = list(positions[index])

        # The orthorhombic cell that (just) accomodates the whole structure is now given by the
        # extremas of position in each dimension:
        minimal_orthorhombic_cell_dimensions = np.array(
            get_extremas_from_positions(positions)[1]
        )
        minimal_orthorhombic_cell_dimensions = np.dot(
            vacuum_factor, minimal_orthorhombic_cell_dimensions
        )
        minimal_orthorhombic_cell_dimensions += vacuum_addition

        # Transform the vector (a, b, c ) to [[a,0,0], [0,b,0], [0,0,c]]
        newcell = np.diag(minimal_orthorhombic_cell_dimensions)
        self.set_cell(newcell.tolist())

        # Now set PBC (checks are done in set_pbc, no need to check anything here)
        self.set_pbc(pbc)

        return self

    def _get_object_phonopyatoms(self):
        """Converts StructureData to PhonopyAtoms

        :return: a PhonopyAtoms object
        """
        from phonopy.structure.atoms import PhonopyAtoms

        atoms = PhonopyAtoms(symbols=[_.kinds for _ in self.properties.sites])
        # Phonopy internally uses scaled positions, so you must store cell first!
        atoms.set_cell(self.properties.cell)
        atoms.set_positions([_.position for _ in self.properties.sites])

        return atoms

    def _get_object_ase(self):
        """Converts
        :py:class:`StructureData <aiida.orm.nodes.data.structure.StructureData>`
        to ase.Atoms

        :return: an ase.Atoms object
        """
        import ase

        asecell = ase.Atoms(cell=self.properties.cell, pbc=self.properties.pbc)

        for site in self.properties.sites:
            asecell.append(site.to_ase(kinds=site.kinds))

        # asecell.set_initial_charges(self.get_site_property("charge"))

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
        if any(self.properties.pbc):
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
        from pymatgen.core.periodic_table import Specie
        from pymatgen.core.structure import Structure

        species = []
        additional_kwargs = {}

        lattice = Lattice(matrix=self.properties.cell, pbc=self.properties.pbc)

        if kwargs.pop("add_spin", False) and any(
            n.endswith("1") or n.endswith("2") for n in self.get_kind_names()
        ):
            # case when spins are defined -> no partial occupancy allowed

            oxidation_state = 0  # now I always set the oxidation_state to zero
            for site in self.properties.sites:
                kind = site.kinds
                if len(kind.symbols) != 1 or (
                    len(kind.weights) != 1 or sum(kind.weights) < 1.0
                ):
                    raise ValueError(
                        "Cannot set partial occupancies and spins at the same time"
                    )
                spin = (
                    -1
                    if site.kinds.endswith("1")
                    else 1
                    if site.kinds.endswith("2")
                    else 0
                )
                try:
                    specie = Specie(
                        kind.symbols[0], oxidation_state, properties={"spin": spin}
                    )
                except TypeError:
                    # As of v2023.9.2, the ``properties`` argument is removed and the ``spin`` argument should be used.
                    # See: https://github.com/materialsproject/pymatgen/commit/118c245d6082fe0b13e19d348fc1db9c0d512019
                    # The ``spin`` argument was introduced in v2023.6.28.
                    # See: https://github.com/materialsproject/pymatgen/commit/9f2b3939af45d5129e0778d371d814811924aeb6
                    specie = Specie(kind.symbols[0], oxidation_state, spin=spin)
                species.append(specie)
        else:
            # case when no spin are defined
            for site in self.properties.sites:
                kind = site.kinds
                specie = Specie(
                    site.symbol,
                    site.charge,
                )  # spin)
                species.append(specie)
            # if any(
            #    create_automatic_kind_name(self.get_kind(name).symbols, self.get_kind(name).weights) != name
            #    for name in self.get_site_property("kinds")
            # ):
            # add "kinds" as a properties to each site, whenever
            # the kinds cannot be automatically obtained from the symbols
            additional_kwargs["site_properties"] = {
                "kinds": self.properties.kinds,
                "charge": self.properties.charges,
                "magmom": self.properties.magmoms
            }

        if kwargs:
            raise ValueError(
                f"Unrecognized parameters passed to pymatgen converter: {kwargs.keys()}"
            )

        positions = [list(x.position) for x in self.properties.sites]

        try:
            return Structure(
                lattice,
                species,
                positions,
                coords_are_cartesian=True,
                **additional_kwargs,
            )
        except ValueError as err:
            raise ValueError(
                "Singular cell detected. Probably the cell was not set?"
            ) from err

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
            raise ValueError(
                f"Unrecognized parameters passed to pymatgen converter: {kwargs.keys()}"
            )

        species = []
        additional_kwargs = {}

        for site in self.properties.sites:
            if hasattr(site, "weight"):
                weight = site.weight
            else:
                weight = 1
            species.append({site.symbol: weight})

        positions = [list(site.position) for site in self.properties.sites]
        mol =  Molecule(species, positions)

        additional_kwargs["site_properties"] = {
                "kinds": self.properties.kinds,
                "charge": self.properties.charges,
                "magmom": self.properties.magmoms
            }

        for prop,value in additional_kwargs.items():
            mol.add_site_property(prop, value)

    def _get_dimensionality(
        self,
    ):
        """Return the dimensionality of the structure and its length/surface/volume.

        Zero-dimensional structures are assigned "volume" 0.

        :return: returns a dictionary with keys "dim" (dimensionality integer), "label" (dimensionality label)
            and "value" (numerical length/surface/volume).
        """
        import numpy as np

        retdict = {}

        pbc = np.array(self.properties.pbc)
        cell = np.array(self.properties.cell)

        dim = len(pbc[pbc])

        retdict["dim"] = dim
        retdict["label"] = self._dimensionality_label[dim]

        if dim not in (0, 1, 2, 3):
            raise ValueError(f"Dimensionality {dim} must be one of 0, 1, 2, 3")

        if dim == 0:
            # We have no concept of 0d volume. Let's return a value of 0 for a consistent output dictionary
            retdict["value"] = 0
        elif dim == 1:
            retdict["value"] = np.linalg.norm(cell[pbc])
        elif dim == 2:
            vectors = cell[pbc]
            retdict["value"] = np.linalg.norm(np.cross(vectors[0], vectors[1]))
        elif dim == 3:
            from aiida_atomistic.data.structure.utils import calc_cell_volume
            retdict["value"] = calc_cell_volume(cell)

        return retdict

    def _validate_dimensionality(
        self,
    ):
        """Check whether the given pbc and cell vectors are consistent."""
        dim = self._get_dimensionality()

        # 0-d structures put no constraints on the cell
        if dim["dim"] == 0:
            return

        # finite-d structures should have a cell with finite volume
        if dim["value"] == 0:
            raise ValueError(
                f'Structure has periodicity {self.properties.pbc} but {dim["dim"]}-d volume 0.'
            )

        return

    def _to_kinds(self, property_name, symbols, thr: float = 0):
        """Called by the `get_kinds` function.
        Get the kinds for a generic site property. Can also be overridden in the specific property.

        ### Search algorithm:

        Basically we compute the indexes array which locates each point in regions centered on our values, considering
        min(values) as reference and each region being of width=thr:

            indexes = np.array((prop_array-np.min(prop_array))/thr,dtype=int)

        To understand this, try to draw the problem considering prop_array=[1,2,3,4] and thr=0.5.
        This methods allows to efficiently clusterize the point using the defined threshold.

        At the end, we reorder the kinds from zero (to have ordered list like Li0, Li1...).
        Basically we define the set of unordered kinds, and the range(len(set(kinds))) being the group of orderd kinds.
        Then we basically do a mapping with the np.where().

        Args:
            thr (float, optional): the threshold to consider two atoms of the same element to be the same kind.
                Defaults to structure.properties.<property>.default_kind_threshold.
                If thr==0, we just return different kind for each site with the original property value. This is
                needed when we have tags for each site, in the get_kind method of StructureData.

        Returns:
            kinds_labels: array of kinds (as integers) associated to the charge property. they are integers so that in the `get_kinds()` method
                                can be used in the matrix representation (the k.T).
            kinds_values: list of the associated property value to each kind detected.
        """
        symbols_array = np.array(symbols)

        if isinstance(self.get_site_property(property_name)[0], list) or isinstance(self.get_site_property(property_name)[0], np.ndarray):
            #reference_array = np.array(self.get_site_property(property_name)[0]) # I take the difference to detect also the case [1,0,0] != [-1,0,0]
            #prop_array = np.array([np.linalg.norm(row-reference_array) for row in self.get_site_property(property_name)])
            prop_array = np.array(self.get_site_property(property_name))
            shape_1 = len(prop_array[0])
            kinds_values = np.zeros((len(symbols_array),shape_1))
        else:
            prop_array = np.array(self.get_site_property(property_name))
            kinds_values = np.zeros(len(symbols_array))

        if thr == 0 or not thr:
            return np.array(range(len(prop_array))), prop_array

        # list for the value of the property for each generated kind.

        if isinstance(prop_array[0], np.ndarray):
            # here, to deal with set of 3D indexes and avoid to deal with directions of the vectors,
            # I transform the set of indexes into string, so I can compared them in the np.where
            indexes = np.array([np.array2string(np.array((row-prop_array[0])/ thr, dtype=int)) for row in prop_array])
        else:
            indexes = np.array((prop_array - np.min(prop_array)) / thr, dtype=int)

        # Here we select the closest value present in the property values
        set_indexes = set(indexes)

        for index in set_indexes:
            where_index_in_indexes = np.where(indexes == index)[0]
            kinds_values[where_index_in_indexes] = prop_array[where_index_in_indexes[0]]


        # here we reorder from zero the kinds.
        list_set_indexes = list(set_indexes)
        kinds_labels = np.zeros(len(symbols_array), dtype=int)
        for i in range(len(list_set_indexes)):
            kinds_labels[np.where(indexes == list_set_indexes[i])[0]] = i

        return kinds_labels, kinds_values

    def __getitem__(self, index):
        "ENABLE SLICING. Return a sliced StructureData."
        # Handle slicing
        sliced_structure_dict = self.to_dict()
        if isinstance(index, slice):
            sliced_structure_dict["sites"] = sliced_structure_dict["sites"][index]
            return self.__class__(**sliced_structure_dict)
        elif isinstance(index, int):
            sliced_structure_dict["sites"] = [sliced_structure_dict["sites"][index]]
            return self.__class__(**sliced_structure_dict)
        else:
            raise TypeError(f"Invalid argument type: {type(index)}")

    def __len__(
        self,
    ):
        return len(self.properties.sites)

    def get_defined_properties(self, exclude_defaults=True):
        """
            Retrieve the defined properties of the structure, categorized into direct, computed, and site-specific properties.

            Args:
                exclude_defaults (bool): If True, properties with default values will be excluded from the result.

            Returns:
                dict: A dictionary with keys "direct", "computed", and "sites". Each key maps to a set of property names:
                    - "direct": Properties directly defined in the structure.
                    - "computed": Properties that are computed based on the structure.
                    - "sites": Properties defined for individual sites within the structure.
        """
        defined_properties = {"direct":set(), "computed":set(),"sites":set()}
        supported_properties = self.get_supported_properties()

        for prop, value in self.properties.model_dump(exclude_defaults=exclude_defaults).items():
            if isinstance(value, list):
                if value.count(_default_values.get(prop, None)) == len(value):
                    # Skip charges, magmoms if not defined for any site.
                    continue
                elif prop == "sites":
                    site_properties = set() # I check on several sites
                    for site_prop in value:
                        site_properties = site_properties.union(site_prop.keys())
                    defined_properties["sites"] = site_properties
                else:
                    defined_properties["direct" if prop in supported_properties["direct"] else "computed"].add(prop)
            elif value is not _default_values.get(prop, None):
                defined_properties["direct" if prop in supported_properties["direct"] else "computed"].add(prop)

        return defined_properties

class SetterMixin:

    def set_pbc(self, value):
        """Set the periodic boundary conditions."""
        from aiida_atomistic.data.structure.utils import _get_valid_pbc
        the_pbc = _get_valid_pbc(value)
        self.properties.pbc = the_pbc

    def set_cell(self, value):
        """Set the cell."""
        from aiida_atomistic.data.structure.utils import _get_valid_cell
        the_cell = _get_valid_cell(value)
        self.properties.cell = the_cell

    def set_cell_lengths(self, value):
        raise NotImplementedError("This method is not implemented yet")

    def set_cell_angles(self, value):
        raise NotImplementedError("This method is not implemented yet")

    def update_site(self, site_index, **kwargs):
        """Update the site at the given index."""
        for key, value in kwargs.items():
            _value = getattr(self.properties, key, None)
            if not _value:
                raise ValueError(f"Invalid key '{key}' for site properties.")
            _value[site_index] = value
        return

    def set_charges(self, value):
        if not len(self.properties.sites) == len(value):
            raise ValueError(
                "The number of charges must be equal to the number of sites"
            )
        else:
            for site_index in range(len(value)):
                self.update_site(site_index, charge=value[site_index])

    def set_magmoms(self, value):
        if not len(self.properties.sites) == len(value):
            raise ValueError(
                "The number of magmom must be equal to the number of sites"
            )
        else:
            for site_index in range(len(value)):
                self.update_site(site_index, magmom=value[site_index])

    def set_kind_names(self, value):
        if not len(self.properties.sites) == len(value):
            raise ValueError(
                "The number of kind_names must be equal to the number of sites"
            )
        else:
            for site_index in range(len(value)):
                self.update_site(site_index, kinds=value[site_index])

    def set_site_property(self, name: str, values: t.List):
        """
        Set a property for each site in the structure.

        :param name (str): The name of the property.
        :param values (list): The values of the property for each site.
        """

        for index, value in enumerate(values):
            self.update_site(index, **{name: value})

    def set_kind_property(self, name: str, value, kind):
        """
        Set a property for all sites of a given kind.

        Parameters:
        :param name (str): The name of the property.
        :param value: The value to set for each site.
        :param kind (str): The name of the kind to filter sites.

        Returns:
        None
        """
        for index, site in enumerate(self.properties.sites):
            if  site.kinds == kind:
                self.update_site(index, **{name: value})

    def add_atom(self, index=-1, **atom_info):

        new_site = Site.atom_to_site(**atom_info)
        # I look for identical species only if the name is not specified
        # _kinds = self.kinds

        # check that the matrix is not singular. If it is, raise an error.
        # check to be done in the core.
        for site_position in self.properties.positions:
            if (
                np.linalg.norm(np.array(new_site.positions) - np.array(site_position))
                < 1e-3
            ):
                raise ValueError(
                    "You cannot define two different sites to be in the same position!"
                )

        if len(self.properties.sites) < index:
            raise IndexError("insert_atom index out of range")
        else:
            """Update the site at the given index."""
            for key, value in new_site.model_dump().items():
                _value = getattr(self.properties, key, None)
                if not _value:
                    raise ValueError(f"Invalid key '{key}' for site properties.")
                if index > -1:
                    _value.insert(index, value)
                else:
                    _value.append(value)
        return

    def pop_atom(self, index=-1):
        # If no index is provided, pop the last item
        site = self.properties.sites[index]
        # Check if the provided index is valid
        if index > len(self.properties.sites):
            raise IndexError("pop_atom index out of range")
        else:
            for key, value in site.model_dump().items():
                _value = getattr(self.properties, key, None)
                if not _value:
                    raise ValueError(f"Invalid key '{key}' for site properties.")
                else:
                    _value.pop(index)

    def clear_property(self, property_name):
        """Clear the given property."""
        mutable_dict = self.to_dict()
        mutable_dict.pop(property_name, None)

        return self.__class__(**mutable_dict)
