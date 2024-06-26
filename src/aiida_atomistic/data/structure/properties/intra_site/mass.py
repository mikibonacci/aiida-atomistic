from typing import List, Literal
from pydantic import Field, validator, root_validator

from aiida.common.constants import elements

from aiida_atomistic.data.structure.properties.intra_site import IntraSiteProperty

_atomic_masses = {el['symbol']: el['mass'] for el in elements.values()}

################################################## Start: Mass property:

class Mass(IntraSiteProperty):
    """
    The mass property. 
    """
    default_kind_threshold: float = 1e-3
    # units... maybe specify in the docs.
    value: List[float] = Field(default=None)

    @validator("value", pre=True, always=True)
    def validate_masses(cls,value,values):
        # I have to use the _property_attributes, as accessing directly parent.properties gives recursion error.
        # Maybe it is possible to change how we get the properties? 
        properties = values["parent"].base.attributes.get("_property_attributes")
        
        if not "positions" in properties:
            # this also validated that we have symbols.
            raise ValueError("If you define masses, you should define also the corresponding positions.")
        
        if not value:
            # This is done if we do not define the default?
            symbols = properties["symbols"]["value"]
            
            # Here I play on the fact that then the dictionary is updated, so I will have the new masses also 
            # in the node.
            properties["mass"]["value"] = [_atomic_masses[symbol] for symbol in symbols]
            return properties["mass"]["value"]
        
        if not len(properties["mass"]["value"]) == len(properties["positions"]["value"]):
            raise ValueError("The number of provided masses should either be zero or match the number of positions.")
        
        
        return value