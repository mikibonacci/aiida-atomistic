import copy

try:
    import ase  # noqa: F401
except ImportError:
    pass

try:
    import pymatgen.core  as core # noqa: F401
except ImportError:
    pass

from aiida.common.constants import elements

_MASS_THRESHOLD = 1.0e-3
# Threshold to check if the sum is one or not
_SUM_THRESHOLD = 1.0e-6
# Default cell
_DEFAULT_CELL = ((0, 0, 0), (0, 0, 0), (0, 0, 0))

_valid_symbols = tuple(i['symbol'] for i in elements.values())
_atomic_masses = {el['symbol']: el['mass'] for el in elements.values()}
_atomic_numbers = {data['symbol']: num for num, data in elements.items()}

_default_values = {
    "mass": _atomic_masses,
}

class Site:
    
    _site_properties = [
        "symbol",
        "mass",
        "kind_name",
        "position",
        "charge",
        "magnetization"
    ]
    
    
    """This class contains the information about a given site of the system.

    It can be a single atom, or an alloy, or even contain vacancies.
    """

    def __init__(self, **kwargs):
        """Create a site.

        :param kind_name: a string that identifies the kind (species) of this site.
                This has to be found in the list of kinds of the StructureData
                object.
                Validation will be done at the StructureData level.
        :param position: the absolute position (three floats) in angstrom
        """
        for site_property in self._site_properties:
            setattr(self, "_"+site_property, None)

        if 'site' in kwargs:
            site = kwargs.pop('site')
            if kwargs:
                raise ValueError("If you pass 'site', you cannot pass any further parameter to the Site constructor")
            if not isinstance(site, Site):
                raise ValueError("'site' must be of type Site")
            for site_property in self._site_properties:
                setattr(self, site_property, getattr(site,site_property))
        elif 'raw' in kwargs:
            raw = kwargs.pop('raw')
            if kwargs:
                raise ValueError("If you pass 'raw', you cannot pass any further parameter to the Site constructor")
            try:
                for site_property in self._site_properties:
                    setattr(self, site_property, raw[site_property])
            except KeyError as exc:
                raise ValueError(f'Invalid raw object, it does not contain any key {exc.args[0]}')
            except TypeError:
                raise ValueError('Invalid raw object, it is not a dictionary')

        else:
            try:
                for site_property in self._site_properties:
                    setattr(self, site_property, kwargs.pop(site_property))
            except KeyError as exc:
                raise ValueError(f'You need to specify {exc.args[0]}')
            if kwargs:
                raise ValueError(f'Unrecognized parameters: {kwargs.keys}')

    def get_raw(self):
        """Return the raw version of the site, mapped to a suitable dictionary.
        This is the format that is actually used to store each site of the
        structure in the DB.

        :return: a python dictionary with the site.
        """
        return {site_property: getattr(self,site_property) for site_property in self._site_properties}

    def get_ase(self, kinds):
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
        aseatom = ase.Atom(position=self.position, symbol=self.symbol, mass=self.mass, charge=self.charge, magmom=self.magnetization)
        
        tag = self.kind_name.replace(self.symbol,"")
        if len(tag)>0:
            tag = int(tag)
        else:
            tag=0
        if tag is not None:
            aseatom.tag = tag
        return aseatom

    @property
    def symbol(self):
        """Return the symbol of this site (a string).

        The type of a site is used to decide whether two sites are identical
        (same mass, symbol, weights, ...) or not.
        """
        return self._symbol

    @symbol.setter
    def symbol(self, value:str):
        """Set the type of this site (a string)."""
        if value not in _valid_symbols:
            raise ValueError(f'Wrong symbol, must be a valid one, not {value}.')
        self._symbol = str(value)
        
    @property
    def mass(self):
        """Return the mass of this site (a float).

        The type of a site is used to decide whether two sites are identical
        (same mass, symbol, weights, ...) or not.
        """
        return self._mass

    @mass.setter
    def mass(self, value: float | int):
        """Set the mass of this site (a float)."""
        if not isinstance(value, float) and not isinstance(value,int):
            if value is None:
                # we fix to the default value.
                self._mass = _atomic_masses[self.symbol]
            else:
                raise ValueError(f'Wrong format for mass, must be a float or an int, not {type(value)}.')
        else:
            self._mass = value
        
    @property
    def kind_name(self):
        """Return the kind name of this site (a string).

        The type of a site is used to decide whether two sites are identical
        (same mass, symbol, weights, ...) or not.
        """
        return self._kind_name

    @kind_name.setter
    def kind_name(self, value: str):
        """Set the type of this site (a string)."""
        self._kind_name = str(value)

    @property
    def position(self):
        """Return the position of this site in absolute coordinates,
        in angstrom.
        """
        return copy.deepcopy(self._position)

    @position.setter
    def position(self, value):
        """Set the position of this site in absolute coordinates,
        in angstrom.
        """
        try:
            internal_pos = tuple(float(i) for i in value)
            if len(internal_pos) != 3:
                raise ValueError
        # value is not iterable or elements are not floats or len != 3
        except (ValueError, TypeError):
            raise ValueError('Wrong format for position, must be a list of three float numbers.')
        self._position = internal_pos
        
    @property
    def charge(self):
        """Return the charge of this site in units of elementary charge.
        """
        return copy.deepcopy(self._charge)

    @charge.setter
    def charge(self, value: float | int):
        """Set the charge of this site in units of elementary charge.
        """
        if not isinstance(value, float) and not isinstance(value,int):
            raise ValueError(f'Wrong format for charge, must be a float or an int, not {type(value)}.')
        self._charge = value
        
    @property
    def magnetization(self):
        """Return the magnetization of this site in units of Bohr magneton.
        """
        return copy.deepcopy(self._magnetization)

    @magnetization.setter
    def magnetization(self, value: float | int):
        """Set the magnetization of this site in units of Bohr magneton.
        """
        if not isinstance(value, float) and not isinstance(value,int):
            raise ValueError(f'Wrong format for magnetization, must be a float or an int, not {type(value)}.')
        self._magnetization = value

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self!s}>'

    def __str__(self):
        return f"kind name '{self.kind_name}' @ {self.position[0]},{self.position[1]},{self.position[2]}"