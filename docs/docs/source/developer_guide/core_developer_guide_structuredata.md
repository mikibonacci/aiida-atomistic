# AiiDA-core developer guide for the StructureData, PropertyCollector classes and properties

This explains in detail the implementation of the classes and methods used in the StructureData class. 

## The StructureData class

It is implemented in the `src/aiida_atomistic/data/structure/__init__.py` file. 
This is basically the old orm.StructureData python file with some major modifications. 
It is still under development, so you will still find the `Site` and `Kind` classes which, in the final version of the file, are meant to be dropped.

*All the properties (pbc, cell, symbols, charge, magnetization) are under the `properties` attribute*. We do no more have the kinds, pbc, cell, symbols and so on as direct attributes of the StructureData instance.

In the `__init__` method, for now we still have there the ASE and Pymatgen stuff, because we want to have the possibility to start from Atoms or Pymatgen structure objects, but is to be implemented in the future. The same should be to dump out ase and pymatgen objects. 
What instead we do for now is to provide a `properties` dictionary input, which is used to initialise the `_properties` attribute of the StructureData. This is a `PropertyCollector` instance, and is a *property*:
```python
@property
def properties(self):
    """ 
    Load the `_property_attribute` stored in the aiida db.
    """
    return PropertyCollector(parent=self, properties=self.base.attributes.get('_property_attributes'))
```
with the setter attribute which is basically disabled, by putting it as 
```python
@properties.setter
def properties(self,value):
    raise AttributeError("After the initialization, `properties` is a read-only attribute")
```
this is done because we want the StructureData to be only a container of data, and then immutable after the initialization. This means that the `append_atom` method is not more supported (or at least should not be in the final version). 

To resume, we have the following logic for the initialisation for a StructureData instance:
1) if no argument is passed to the constructor, we initialise a StructureData with one Hydrogen atom  at the origin (0,0,0), and a cell with zero volume. Just a placeholder to be able to have the `get_supported_properties` method of the collector;
2) if one among ase, pymatgen, pymatgen_structure, pymatgen_molecule are passed, we initialise using a corresponding class method (**to be implemented**);
3) if the `properties` dictionary is provided as input, we proceed with the standard initialisation of the `PropertyCollector` instance.

In particular, the option (3) has the following crucial block of code (that can be optimised):
```python
# Store the properties in the StructureData node.
if not self.is_stored: 
    self.base.attributes.set('_property_attributes',self._properties._property_attributes)  
    if not "kinds" in copied_properties.keys():
        # Generate kinds. Code should be improved.
        new_properties = self.get_kinds()
        copied_properties.update(new_properties)
        self._properties = PropertyCollector(parent=self, properties=copied_properties)
        self.base.attributes.set('_property_attributes',self._properties._property_attributes)
    else:
        # Validation, step 1 - Final get_kinds() check - this is a bad way to do it, but it works
        self.get_kinds(kind_tags=self.properties.kinds.value)
```
where we generate or check the `kinds` property: if kinds are not defined, we generate them via the `get_kinds` method, which return a dictionary of properties to update the input one (*copied_properties*) to be
then used to initialise the final PropertyCollector instance. Instead, if kinds are defined, we try to call the same `get_kinds` method, but providing as input the `self.properties.kinds.value` list, which is then check for consistency (i.e., atoms with the same kind should have also the same value of each `intra-site` related property).
**To test**: store and load the StructureData node.

### Methods provided in the class

#### `get_kinds`: generating a list of kinds and/or checking pre-existing ones

The method is meant to provide a list of kinds and related properties (the *intra-site* ones), starting from what you have defined under the `properties` attribute of the StructureData node. For details on the algorithm and inputs and outputs, see the corresponding method docstring. 
In principle, if in a plugin we need to provide the properties in a kind-wise format, this is the function to be used. *For now, the list in output is still site-defined, so its length is the same as, for examples, the symbols property*
Here we want to underline that this method works also as a check for user-defined kinds: if they are already present in the properties, the method finds its kinds imposing a threshold for the properties of zero and checks if the mapping is the same in the automatic and user-defined cases, as it should be if the user defined consistently the kinds. Indeed, the zero threshold will always find same kinds for sites in which the properties are all the same. This should happen also in the user-defined set of kinds, otherwise there will be an ambiguity on the value of the related properties. To be more clear, the user cannot define, in a two atoms system, the same kind for the atoms if they have, for example, different magnetization. 

#### `to_dict`: a way to initialise a new modified StructureData instance

This methods return a dictionary which can be directly used to feed the StructureData constructor. It is fundamental in case we want to start from a given structure and modify something. Moreover, can be useful in cases where we have as output of a CalcJob a new StructureData. For example when we obtain a new output magnetization, relaxed positions and so on. 

#### Planned methods

from/to_ase, from/to_pymatgen...

## The PropertyCollector class

