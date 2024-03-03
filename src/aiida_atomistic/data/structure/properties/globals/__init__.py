from aiida_atomistic.data.structure.properties.property_utils import BaseProperty

################################################## Start: IntraSiteProperty class:
class GlobalProperty(BaseProperty):
    
    """Generic class for global properties. 
    Extends the BaseProperty class with specific methods for the properties which relatesto the whole system.
    """
    domain: str = "global"
