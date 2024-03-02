from typing import List
from pydantic import Field, validator

from aiida.common.constants import elements

from aiida_atomistic.data.structure.properties.intra_site import IntraSiteProperty

################################################## Start: Symbols property:

_valid_symbols = tuple(i['symbol'] for i in elements.values())

class Kinds(IntraSiteProperty):
    """
    The kinds property, for each atom(site). 
    Can also be not defined. In that case, the plugin will use the get_kinds() method to have the list of kinds to use. 
    """
    domain = "intra-site"
    value: List[str]
    
    @validator("value", always=True)
    def validate_kinds(cls,value,values):
 
        properties = values["parent"].base.attributes.get("_property_attributes")
              
        if not "symbols" in properties.keys():
            raise ValueError("If you define kinds, you should define also the corresponding symbols.")
        #elif not value:
            # Here I play on the fact that then the dictionary is updated, so I will have the new masses also 
            # in the node.
        #    properties["kinds"]["value"] = properties["symbols"]["value"]
        #    return properties["kinds"]["value"]
        elif not len(value) == len(properties["symbols"]["value"]):
            raise ValueError("The number of provided kinds should match the number of symbols.")
        
        # Check that the properties are not inconsistent with respect to the defined kinds: i.e. 
        # same kinds should have same properties.
        # Please note: the check is done via the get_kinds() method called in the StructureData.__init__
        
        return value
################################################## End: PBC property.