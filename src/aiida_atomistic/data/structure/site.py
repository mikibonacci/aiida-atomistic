import numpy as np

import typing as t
from pydantic import BaseModel, Field, field_validator, computed_field,model_validator

try:
    import ase  # noqa: F401
except ImportError:
    pass

try:
    import pymatgen.core as core  # noqa: F401
except ImportError:
    pass

from aiida.common.constants import elements

from aiida_atomistic.data.structure.utils import (
    _create_weights_tuple,
    create_automatic_kind_name,
    validate_weights_tuple,
    ObservedArray,
    FrozenList,
    freeze_nested,
)


_MASS_THRESHOLD = 1.0e-3
# Threshold to check if the sum is one or not
_SUM_THRESHOLD = 1.0e-6
# Default cell
_DEFAULT_CELL = ((0, 0, 0), (0, 0, 0), (0, 0, 0))

_valid_symbols = tuple(i["symbol"] for i in elements.values())
_atomic_masses = {el["symbol"]: el["mass"] for el in elements.values()}
_atomic_numbers = {data["symbol"]: num for num, data in elements.items()}

_default_values = {
    "mass": _atomic_masses,
    "charge": 0,
    "magmom": [0, 0, 0],
    "hubbard": None,
    "weight": 1,
}

class SiteCore(BaseModel):
                
    symbol: t.Literal[_valid_symbols]
    kind_name: str | t.Any = None
    position: list | t.Any = Field(min_length=3, max_length=3)
    mass: float | t.Any 
    charge: float | t.Any = 0
    magmom: t.List[float] | t.Any = Field(min_length=3, max_length=3, default=[0.0, 0.0, 0.0])
    
    class Config:
        from_attributes = True
        frozen=False
        arbitrary_types_allowed=True

    """This class contains the information about a given site of the system.

    It can be a single atom, or an alloy, or even contain vacancies.
    """
    
    @field_validator('position','magmom')
    def validate_list(cls, v: t.List[float]) -> t.Any:
        
        if not cls._mutable.default:
            return freeze_nested(v)
        else:
            return v
        
    @model_validator(mode='before')
    def check_minimal_requirements(cls, data):
        if not data.get("mass", None):
            data["mass"] = _atomic_masses[data["symbol"]]
        return data


    @staticmethod
    def atom_to_site(
        aseatom: t.Optional[ase.Atom] = None,
        position: t.Optional[list] = None,
        symbol: t.Optional[t.Union[_valid_symbols]] = None,
        kind_name: t.Optional[str] = None,
        charge: t.Optional[float] = None,
        magmom: t.Optional[list] = None,
        mass: t.Optional[float] = None,
        ) -> dict: 
        """Convert an ASE atom or dictionary to a dictionary object which the correct format to describe a Site."""

        if aseatom is not None:
            if position:
                raise ValueError(
                    "If you pass 'aseatom' as a parameter to "
                    "append_atom, you cannot pass any further"
                    "parameter"
                )
            position = aseatom.position.tolist()
            symbol = aseatom.symbol
            kind_name = symbol + str(aseatom.tag)
            charge = aseatom.charge
            if aseatom.magmom is None:
                magmom = [0, 0, 0]
            elif isinstance(aseatom.magmom, (int, float)):
                magmom = [aseatom.magmom, 0, 0]
            else:
                magmom = aseatom.magmom
            mass = aseatom.mass
        else:
            if position is None:
                raise ValueError("You have to specify the position of the new atom")
            
            if symbol is None:  
                raise ValueError("You have to specify the symbol of the new atom")
            
            # all remaining parameters
            kind_name = symbol if kind_name is None else kind_name
            charge = 0 if charge is None else charge
            magmom = [0,0,0] if magmom is None else magmom
            mass = _atomic_masses[symbol] if mass is None else mass

        new_site = dict(
            symbol=symbol,
            kind_name=kind_name,
            position=position.tolist() if isinstance(position, np.ndarray) else position,
            mass=mass,
            charge=charge,
            magmom=magmom.tolist() if isinstance(magmom, np.ndarray) else magmom
        )

        return new_site
    
    def update(self, **new_data):
            for field, value in new_data.items():
                setattr(self, field, value)
    
    def set_automatic_kind_name(self, tag=None):
        """Set the type to a string obtained with the symbols appended one
        after the other, without spaces, in alphabetical order;
        if the site has a vacancy, a X is appended at the end too.
        """
        name_string = create_automatic_kind_name(self.symbol, self.weight)
        if tag is None:
            self.name = name_string
        else:
            self.name = f"{name_string}{tag}"

    def to_ase(self, kinds):
        """Return a ase.Atom object for this site.

        :param kinds: the list of kinds from the StructureData object.

        .. note:: If any site is an alloy or has vacancies, a ValueError
            is raised (from the site.get_ase() routine).
        """
        from collections import defaultdict

        import ase

        # I create the list of tags
        tag_list = []
        used_tags = defaultdict(list)

        # we should put a small routine to do tags. or instead of kinds, provide the tag (or tag mapping).
        tag = None
        aseatom = ase.Atom(
            **self.dict()
        )

        tag = self.kind_name.replace(self.symbol, "")
        if len(tag) > 0:
            tag = int(tag)
        else:
            tag = 0
        if tag is not None:
            aseatom.tag = tag
        return aseatom

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self!s}>"

    def __str__(self):
        return f"kind name '{self.kind_name}' @ {self.position[0]},{self.position[1]},{self.position[2]}"

# The Classes which are exposed to the user:
class SiteMutable(SiteCore):
    
    _mutable = True
    
    symbol: str
    kind_name: str | t.Any = None
    position: list | t.Any = None
    mass: float | t.Any = None
    charge: float | t.Any = 0
    magmom: t.List[float] | t.Any = Field(min_length=3, max_length=3, default=[0.0, 0.0, 0.0])
    
    class Config:
        from_attributes = True
        frozen= False
    
class SiteImmutable(SiteCore):
    
    _mutable = False
    
    symbol: str
    kind_name: str | t.Any = None
    position: list
    mass: float | t.Any = None
    charge: float | t.Any = 0
    magmom: t.List[float] | t.Any = Field(min_length=3, max_length=3, default=[0.0, 0.0, 0.0])
    
    class Config:
        from_attributes = True
        frozen= True