import numpy as np

import typing as t
import re
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

try:
    import ase  # noqa: F401
except ImportError:
    pass

try:
    import pymatgen.core as core  # noqa: F401
except ImportError:
    pass

from aiida.common.constants import elements
from plumpy.utils import AttributesFrozendict


_MASS_THRESHOLD = 1.0e-3
_MAGMOM_THRESHOLD = 1.0e-4
# Threshold to check if the sum is one or not
_SUM_THRESHOLD = 1.0e-6
# Default cell
_DEFAULT_CELL = ((0, 0, 0), (0, 0, 0), (0, 0, 0))

_valid_symbols = tuple(i["symbol"] for i in elements.values())
_atomic_masses = {el["symbol"]: el["mass"] for el in elements.values()}
_atomic_numbers = {atom["symbol"]: num for num, atom in elements.items()}

_default_values = {
    "mass": _atomic_masses,
    "charge": 0,
    "magmom": [0, 0, 0],
    "hubbard": None,
    "weights": (1,)
}

def freeze_nested(obj):
    """
    Recursively freezes a nested dictionary or list by converting it into an immutable object.

    Args:
        obj (dict or list): The nested dictionary or list to be frozen.

    Returns:
        AttributesFrozendict or FrozenList: The frozen version of the input object.

    """
    if isinstance(obj, dict):
        return AttributesFrozendict({k: freeze_nested(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return FrozenList(freeze_nested(v) for v in obj)
    else:
        return obj

class FrozenList(list):
    """
    A subclass of list that represents an immutable list.

    This class overrides the __setitem__ method to raise a ValueError
    when attempting to modify the list.

    Usage:
    >>> my_list = FrozenList([1, 2, 3])
    >>> my_list[0] = 4
    ValueError: This list is immutable
    """

    def __setitem__(self, index, value):
        raise ValueError("This list is immutable")

class SiteCore(BaseModel):
    """This class contains the core information about a given site of the system.

    It can be a single atom, or an alloy, or even contain vacancies.

    """
    model_config = ConfigDict(from_attributes = True,  frozen = False,  arbitrary_types_allowed = True)

    symbol: t.Optional[str] # validation is done in the check_is_alloy
    kind_name: t.Optional[str]
    position: t.List[float] = Field(min_length=3, max_length=3)
    mass: t.Optional[float] = Field(gt=0)
    charge: t.Optional[float] = Field(default=_default_values["charge"])
    magmom: t.Optional[t.List[float]] = Field(min_length=3, max_length=3, default=_default_values["magmom"])
    weights: t.Optional[t.Tuple[float, ...]] = Field(default=_default_values["weights"])

    @field_validator('position','magmom')
    def validate_list(cls, v: t.List[float]) -> t.Any:
        if not cls._mutable.default:
            return freeze_nested(v)
        else:
            return v

    @model_validator(mode='before')
    def check_minimal_requirements(cls, data):
        from aiida_atomistic.data.structure.utils import check_is_alloy
        if "symbol" not in data and cls._mutable.default:
            data["symbol"] = "H"

        # here below we proceed as in the old Kind, where we detect if
        # we have an alloy (i.e. more than one element for the given site)
        alloy_detector = check_is_alloy(data)
        if alloy_detector:
            data.update(alloy_detector)

        if "mass" not in data:
            data["mass"] = _atomic_masses[data["symbol"]]
        elif not data["mass"]:
            data["mass"] =  _atomic_masses[data["symbol"]]
        #elif data["mass"]<=0:
        #    raise ValueError("The mass of an atom must be positive")

        if "kind_name" not in data:
            data["kind_name"] = data["symbol"]

        return data

    @property
    def is_alloy(self):
        """Return whether the Site is an alloy, i.e. contains more than one element

        :return: boolean, True if the kind has more than one element, False otherwise.
        """
        return len(self.weights) != 1

    @property
    def alloy_list(self):
        """Return the list of elements in the given site which is defined as an alloy
        """
        return re.sub( r"([A-Z])", r" \1", self.symbol).split()

    @property
    def has_vacancies(self):
        """Return whether the Structure contains vacancies, i.e. when the sum of the weights is less than one.

        .. note:: the property uses the internal variable `_SUM_THRESHOLD` as a threshold.

        :return: boolean, True if the sum of the weights is less than one, False otherwise
        """
        return not 1.0 - sum(self.weights) < _SUM_THRESHOLD

    @classmethod
    def atom_to_site(
        cls,
        aseatom: t.Optional[ase.Atom] = None,
        position: t.Optional[list] = None,
        symbol: t.Optional[t.Literal[_valid_symbols]] = None,
        kind_name: t.Optional[str] = None,
        mass: t.Optional[float] = None,
        charge: t.Optional[float] = _default_values["charge"],
        magmom: t.Optional[t.List[float]] = _default_values["magmom"],
        weights: t.Optional[t.Tuple[float, ...]] = _default_values["weights"],
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
                magmom = _default_values["magmom"]
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
            mass = _atomic_masses[symbol] if mass is None else mass
            weights = _default_values["weights"] if weights is None else weights

        new_site = cls(
            symbol=symbol,
            kind_name=kind_name,
            position=position.tolist() if isinstance(position, np.ndarray) else position,
            mass=mass,
            charge=charge,
            magmom=magmom.tolist() if isinstance(magmom, np.ndarray) else magmom
        )

        return new_site

    def update(self, **new_data):
        """Update the attributes of the SiteCore instance with new values.

        :param new_data: keyword arguments representing the attributes to be updated
        """
        for field, value in new_data.items():
            setattr(self, field, value)

    def get_magmom_coord(self, coord="spherical"):
        """Get magnetic moment in given coordinate.

        :return: spherical theta and phi in unit rad
                cartesian x y and z in unit ang
        """
        if self.magmom == [0,0,0]:
            return {"starting_magnetization": 0, "angle1": 0, "angle2": 0} if coord == "spherical" else [0, 0, 0]

        magmom = self.magmom
        if coord not in ["spherical", "cartesian"]:
            raise ValueError("`coord` can only be `cartesian` or `spherical`")
        if coord == "cartesian":
            magmom_coord = magmom
        else:
            r = np.linalg.norm(magmom)
            if r < _MAGMOM_THRESHOLD:
                magmom_coord = [0.0, 0.0, 0.0]
            else:
                theta = np.arccos(magmom[2]/r) # arccos(z/r)
                theta = theta / np.pi * 180
                phi = np.arctan2(magmom[1], magmom[0]) # atan2(y, x)
                phi = phi / np.pi * 180
                magmom_coord = (r, theta, phi)
                # unit always in degree to fit qe inputs.
        return {"starting_magnetization": magmom_coord[0], "angle1": magmom_coord[1], "angle2": magmom_coord[2]}

    def set_automatic_kind_name(self, tag=None):
        """Set the type to a string obtained with the symbols appended one
        after the other, without spaces, in alphabetical order;
        if the site has a vacancy, a X is appended at the end too.

        :param tag: optional tag to be appended to the kind name
        """
        from aiida_atomistic.data.structure.utils import create_automatic_kind_name
        name_string = create_automatic_kind_name(self.symbol, self.weight)
        if tag is None:
            self.name = name_string
        else:
            self.name = f"{name_string}{tag}"

    def to_ase(self, kinds):
        """Return a ase.Atom object for this site.

        :param kinds: the list of kinds from the StructureData object.
        :return: ase.Atom object representing this site
        :raises ValueError: if any site is an alloy or has vacancies
        """
        from collections import defaultdict
        import ase

        # I create the list of tags
        tag_list = []
        used_tags = defaultdict(list)

        # we should put a small routine to do tags. or instead of kinds, provide the tag (or tag mapping).
        tag = None
        atom_dict = self.model_dump()
        atom_dict.pop("kind_name",None)
        atom_dict.pop("weights",None)
        aseatom = ase.Atom(
            **atom_dict
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
    """
    A mutable version of the `SiteCore` class.

    This class represents a site in a crystal structure that can be modified.

    Attributes:
        _mutable (bool): Flag indicating if the site is mutable.
    """

    _mutable = True


class SiteImmutable(SiteCore):
    """
    A class representing an immutable site in a crystal structure.

    This class inherits from the `SiteCore` class and adds the functionality to create an immutable site.
    An immutable site cannot be modified once it is created.

    Attributes:
        _mutable (bool): A flag indicating whether the site is mutable or immutable.
    """
    model_config = ConfigDict(from_attributes = True,  frozen = True,  arbitrary_types_allowed = True)

    _mutable = False
