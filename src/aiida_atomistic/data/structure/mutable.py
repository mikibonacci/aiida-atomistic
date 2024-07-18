import typing as t 
import numpy as np

from aiida_atomistic.data.structure.site import Site
from aiida_atomistic.data.structure.utils import _get_valid_cell, _get_valid_pbc

from .core import StructureData, StructureDataCore


class StructureDataMutable(StructureDataCore):
    """
    The mutable StructureData object.
    contains mutation methods.
    It has the same data structure of the StructureData, so in
    principle we can also use automatic aiida data type serialization.

    :param pbc: A list of three boolean values indicating the periodic boundary conditions (PBC)
                for each spatial dimension. If not provided, defaults to (True, True, True).
    :param cell: A 3x3 matrix (list of lists) representing the lattice vectors of the cell.
                If not provided, a default cell matrix (_DEFAULT_CELL) will be used.
    :param sites: A list of Site objects representing the atomic positions and species within the structure.
                If not provided, an empty list will be used.
    """
    
    _mutable = True

    def __init__(self,
                pbc: t.Optional[list[bool]] = None,
                cell: t.Optional[list[list[float]]] = None,
                sites: t.Optional[list[Site]] = None):
        super().__init__(pbc, cell, sites)

        global_properties = self.get_global_properties()
        #for prop, value in global_properties.items():
        #    self._data[prop] = value
        
        
    def set_pbc(self, value):
        """Set the periodic boundary conditions."""
        the_pbc = _get_valid_pbc(value)
        self._data["pbc"] = the_pbc

    def set_cell(self, value):
        """Set the cell."""
        the_cell = _get_valid_cell(value)
        self._data["cell"] = the_cell

    def set_cell_lengths(self, value):
        raise NotImplementedError("Modification is not implemented yet")

    def set_cell_angles(self, value):
        raise NotImplementedError("Modification is not implemented yet")

    def update_site(self, site_index, **kwargs):
        """Update the site at the given index."""
        self._data["sites"][site_index].update(**kwargs)

    def set_charges(self, value):
        if not len(self._data["sites"]) == len(value):
            raise ValueError(
                "The number of charges must be equal to the number of sites"
            )
        else:
            for site_index in range(len(value)):
                self.update_site(site_index, charge=value[site_index])
                
    def set_magmoms(self, value):
        if not len(self._data["sites"]) == len(value):
            raise ValueError(
                "The number of magmom must be equal to the number of sites"
            )
        else:
            for site_index in range(len(value)):
                self.update_site(site_index, magmom=value[site_index])

    def set_kind_names(self, value):
        if not len(self._data["sites"]) == len(value):
            raise ValueError(
                "The number of kind_names must be equal to the number of sites"
            )
        else:
            for site_index in range(len(value)):
                self.update_site(site_index, kind_name=value[site_index])

    def add_atom(self, atom_info, index=-1):

        new_site = Site.atom_to_site(**atom_info)
        # I look for identical species only if the name is not specified
        # _kinds = self.kinds

        # check that the matrix is not singular. If it is, raise an error.
        # check to be done in the core.
        for site_position in self.get_site_property("position"):
            if (
                np.linalg.norm(np.array(new_site["position"]) - np.array(site_position))
                < 1e-3
            ):
                raise ValueError(
                    "You cannot define two different sites to be in the same position!"
                )

        if len(self._data["sites"]) < index:
            raise IndexError("insert_atom index out of range")
        else:
            self._data["sites"].append(new_site) if index == -1 else self._data[
                "sites"
            ].insert(index, new_site)

    def pop_atom(self, index=None):
        # If no index is provided, pop the last item
        if index is None:
            return self._data["sites"].pop()
        # Check if the provided index is valid
        elif 0 <= index < len(self._data["sites"]):
            return self._data["sites"].pop(index)
        else:
            raise IndexError("pop_atom index out of range")

    def to_structuredata(self):
        
        return StructureData(**self.to_dict())
