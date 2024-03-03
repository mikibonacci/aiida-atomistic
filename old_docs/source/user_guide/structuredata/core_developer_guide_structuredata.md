# AiiDA-core developer guide for the StructureData class and properties

This explains in detail the implementation of the classes and methods used to build the StructureData class. 

## The new StructureData object

All the properties are under the `properties` attribute. We do not have the kinds, pbc, cell, symbols and so on 
as direct attribute of the StructureData instance.

In the `__init__` method, for now we still have there the ASE and Pymatgen stuff, because we want to have the possibility to start from Atoms or Pymatgen structure objects, but is to be implemented in the future. What instead we do for now is to provide a `properties` dictionary input  