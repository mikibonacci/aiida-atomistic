import copy
import functools
import json
import typing as t
from pydantic import BaseModel, Field, field_validator, PrivateAttr
import numpy as np
import warnings

from aiida import orm
from aiida.common.constants import elements
from aiida.orm.nodes.data import Data

from aiida_atomistic.data.structure.models import MutableStructureModel, ImmutableStructureModel
from aiida_atomistic.data.structure.mixin import GetterMixin, SetterMixin

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


from aiida_atomistic.data.structure.utils import (
    _get_valid_cell,
    _get_valid_pbc,
    atom_kinds_to_html,
    calc_cell_volume,
    create_automatic_kind_name,
    get_formula,
    ObservedArray,
    FrozenList,
    freeze_nested,
)

_MASS_THRESHOLD = 1.0e-3
# Threshold to check if the sum is one or not
_SUM_THRESHOLD = 1.0e-6
# Default cell
_DEFAULT_CELL = ((0, 0, 0),) * 3

_valid_symbols = tuple(i["symbol"] for i in elements.values())
_atomic_masses = {el["symbol"]: el["mass"] for el in elements.values()}
_atomic_numbers = {data["symbol"]: num for num, data in elements.items()}


class StructureData(Data, GetterMixin):
        
    def __init__(self, **kwargs):
        
        self._properties = ImmutableStructureModel(**kwargs)
        super().__init__()
        
        for prop, value in self.properties.dict().items():
            self.base.attributes.set(prop, value)
    
    @property 
    def properties(self):
        if self.is_stored:
            return ImmutableStructureModel(**self.base.attributes.all)
        else:
            return self._properties
        
    def to_mutable(self):
        return StructureDataMutable(**self.properties.dict())
        
class StructureDataMutable(GetterMixin, SetterMixin):
        
    def __init__(self, **kwargs):
        
        self._properties = MutableStructureModel(**kwargs)
        
    @property 
    def properties(self):
        return self._properties
    
    def to_immutable(self):
        return StructureData(**self.properties.dict())
