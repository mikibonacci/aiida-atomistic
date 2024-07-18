import numpy as np
import typing as t 

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

class Site:

    _site_properties = [
        "symbol",
        "position",
        "mass",
        "kind_name",
        "weight",
        "charge",
        "magmom",
    ]

    """This class contains the information about a given site of the system.

    It can be a single atom, or an alloy, or even contain vacancies.
    """

    def __init__(self, parent=None, index=None, **kwargs):
                
        """Create a site.

        :param kind_name: a string that identifies the kind (species) of this site.
                This has to be found in the list of kinds of the StructureData
                object.
                Validation will be done at the StructureData level.
        :param position: the absolute position (three floats) in angstrom

        TBD: sites should be always immutable? so we just can use set_* in StructureDataMutable.
        """
        
        self._parent = parent
        self._index = index
        
        for site_property in self._site_properties:
            setattr(self, "_" + site_property, None)

        if "site" in kwargs:
            site = kwargs.pop("site")
            if kwargs:
                raise ValueError(
                    "If you pass 'site', you cannot pass any further parameter to the Site constructor"
                )
            if not isinstance(site, Site):
                raise ValueError("'site' must be of type Site")
            for site_property in self._site_properties:
                setattr(self, "_" + site_property, getattr(site, site_property))
        elif "raw" in kwargs:
            raw = kwargs.pop("raw")
            if kwargs:
                raise ValueError(
                    "If you pass 'raw', you cannot pass any further parameter to the Site constructor"
                )
            try:
                for site_property in self._site_properties:
                    if site_property in raw.keys():
                        setattr(self, "_" + site_property, raw[site_property])
                    else:
                        setattr(self, "_" + site_property, self._get_default(site_property))
            except KeyError as exc:
                raise ValueError(
                    f"Invalid raw object, it does not contain any key {exc.args[0]}"
                )
            except TypeError:
                raise ValueError("Invalid raw object, it is not a dictionary")

        else:
            try:
                for site_property in self._site_properties:
                    if site_property in kwargs.keys():
                        setattr(self, "_" + site_property, kwargs.pop(site_property))
            except KeyError as exc:
                raise ValueError(f"You need to specify {exc.args[0]}")
            if kwargs:
                raise ValueError(f"Unrecognized parameters: {kwargs.keys}")

    @property
    def symbol(self):
        """Return the symbol of this site (a string).

        The type of a site is used to decide whether two sites are identical
        (same mass, symbol, weight, ...) or not.
        """
        return self._symbol

    @property
    def mass(self):
        """Return the mass of this site (a float).

        The type of a site is used to decide whether two sites are identical
        (same mass, symbol, weight, ...) or not.
        """
        return self._mass

    @property
    def kind_name(self):
        """Return the kind name of this site (a string).

        The type of a site is used to decide whether two sites are identical
        (same mass, symbol, weight, ...) or not.
        """
        return self._kind_name

    @property
    def position(self):
        """Return the position of this site in absolute coordinates,
        in angstrom.
        """
        position = ObservedArray(self._position)
        position.flags.writeable = self._parent._mutable
        return position

    @position.setter
    def position(self, value):
        """Setter for the position of the site."""
        if not self._parent._mutable:
            raise ValueError("The site is not mutable")
        
        value = value.tolist() if isinstance(value, np.ndarray) else value
        self._position = value # we need to update also the parent
        self._parent.update_site(self._index, position=value)
    
    @property
    def charge(self):
        """Return the charge of this site in units of elementary charge."""
        return self._charge

    @property
    def magmom(self):
        """Return the magmom of this site in units of Bohr magneton."""
        magmom = ObservedArray(self._magmom)
        magmom.flags.writeable = False
        return magmom
    
    @magmom.setter
    def magmom(self, value):
        """Setter for the magmom of the site."""
        if not self._parent._mutable:
            raise ValueError("The magmom is not mutable")
        
        value = value.tolist() if isinstance(value, np.ndarray) else value
        self._magmom = value # we need to update also the parent
        self._parent.update_site(self._index, magmom=value)
    
    @property
    def weight(self):
        """weight for this species kind. Refer also to
        :func:validate_symbols_tuple for the validation rules on the weight.
        """
        return self._weight


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

    def get_raw(self):
        """Return the raw version of the site, mapped to a suitable dictionary.
        This is the format that is actually used to store each site of the
        structure in the DB.

        :return: a python dictionary with the site.
        """
        return {
            site_property: getattr(self, site_property)
            for site_property in self._site_properties
        }

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
            position=self.position,
            symbol=self.symbol,
            mass=self.mass,
            charge=self.charge,
            magmom=self.magmom,
        )

        tag = self.kind_name.replace(self.symbol, "")
        if len(tag) > 0:
            tag = int(tag)
        else:
            tag = 0
        if tag is not None:
            aseatom.tag = tag
        return aseatom

    def _get_default(self, site_property):
        if site_property == "mass":
            default_value = _atomic_masses[self.symbol]
        elif site_property == "kind_name":
            default_value = self.symbol
        else:
            default_value = _default_values[site_property]
        return default_value

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self!s}>"

    def __str__(self):
        return f"kind name '{self.kind_name}' @ {self.position[0]},{self.position[1]},{self.position[2]}"
