# AiiDA-core developer guide for the StructureData, PropertyCollector classes and properties

The following document explains in detail the implementation of the classes and methods used in the StructureData class. 

## The StructureData class

It is implemented in the `src/aiida_atomistic/data/structure/__init__.py` file. 
This is basically the old orm.StructureData python file with some major modifications. 
It is still under development, so you will still find the `Site` and `Kind` classes which, in the final version of the file, are meant to be dropped. 

*All the properties (pbc, cell, symbols, charge, magnetization) are under the `properties` attribute*. Maybe we can still allow to access these properties
directly, without passing from the `properties` attribute. 
*The properties are stored in the database via the line*

```python
self.base.attributes.set('_property_attributes',self._properties._property_attributes)
```

And are then never modified, except in the case the kinds are generated automatically. See below.

In the `__init__` method, for now we still have the ASE and Pymatgen stuff, because we want to have the possibility to start from Atoms or Pymatgen structure objects, but is to be implemented in the future. The same should be to dump ASE and Pymatgen objects. 
What instead we do for now is to provide a `properties` dictionary input, which is used to initialise the `_properties` private attribute of the StructureData. This is a `PropertyCollector` instance, and is a *property*:
```python
@property
def properties(self):
    """ 
    Load the `_property_attribute` stored in the aiida db.
    """
    return PropertyCollector(parent=self, properties=self.base.attributes.get('_property_attributes'))
```
with the setter attribute which is disabled with a raise error:
```python
@properties.setter
def properties(self,value):
    raise AttributeError("After the initialization, `properties` is a read-only attribute")
```
this is done because we want the StructureData to be only a container of data, and then immutable after the initialization. This means that the `append_atom` method is not more supported (or at least should not be present in the final version). 

We have the following logic for the initialisation for a StructureData instance:

- if no argument is passed to the constructor, we initialise a StructureData with one Hydrogen atom at the origin (0,0,0), and a cell with zero volume. Just a placeholder to be able to call the `get_supported_properties` method of the collector, which print a list of properties that can be set;
- if one among ase, pymatgen, pymatgen_structure, pymatgen_molecule are passed, we initialise using a corresponding class method (**to be implemented**);
- if the `properties` dictionary is provided as input, we proceed with the standard initialisation of the `PropertyCollector` instance.

In particular, the option (3) has the following crucial block of code (can be optimised):
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
where we generate and/or check the `kinds` property: if kinds are not defined, we generate them via the `get_kinds` method, which return a dictionary of properties to update the input one (*copied_properties*) to be
then used to initialise the final PropertyCollector instance. We can disable the automatic kind creation by providing as False the boolean input `allow_kinds`. Instead, if kinds are defined, we try to call the same `get_kinds` method, but providing as input the `self.properties.kinds.value` list, which is then checked for consistency (i.e., atoms with the same kind should have also the same value of each `intra-site` related property). The method is described below in the next section.

### Methods provided in the class

#### Generating a list of kinds and/or checking pre-existing ones: `get_kinds()`

The method is meant to provide a list of kinds and related properties (the *intra-site* ones), starting from what the user defines under the `properties` attribute of the StructureData node. For details on the algorithm and inputs and outputs, see the corresponding docstring. 
In principle, if in a plugin we need to provide the properties in a kind-wise format, this is the function to be used. *For now, the list in output is still site-defined, so its length is the same as, for examples, the symbols property*.
Here we want to underline that this method works also as a check for user-defined kinds: if they are already present in the properties, the method finds its kinds imposing a threshold for the properties of zero and checks if the mapping is the same in the automatic and user-defined cases, as it should be if the user defined consistently the kinds. Indeed, the zero threshold will always find same kinds for sites in which the properties are all the same. This should happen also in the user-defined set of kinds, otherwise there will be an ambiguity on the value of the related properties. To be more clear, the user cannot define, in a two atoms structure, the same kind for the atoms if they have, for example, different magnetization. This will introduce inconsistencies for example in the quantumespresso input file generation.

*Using the automatic kind generation, the value of the properties stored in the node will be different with respect to the input ones, as we use the `property kind threshold` to discretize the property value space (in such a way to find the kinds).*

#### A way to initialise a new modified StructureData instance: `to_dict()`

This methods return a dictionary which can be directly used to feed the StructureData constructor. It is useful in case we want to start from a given structure and modify something. Moreover, it can serve in cases where we have as output of a CalcJob a new StructureData. For example when we obtain a new output magnetization, relaxed positions and so on. 

#### Planned methods

from/to_ase, from/to_pymatgen...

## The PropertyCollector class

This class contains all the properties of our StructureData. It is immutable after the creation, in such a way to avoid unwanted modifications and unexpected behaviour of the StructureData. It is a subclass of the `HasPropertyMixin` class, which has the `PropertyMixinMetaclass` as metaclass. This last class contains the logic to attach the setter and getter method to each property. Respectively, these are the `_set_property` and `_template_property` methods. However, only the getter method is really defined, as we do not want the properties to be mutable. If we decide instead that we want something which can be changed, we can always activate the `_set_property` and modify it to work with the current implementation. 

Among useful methods, we have:

-   `get_supported_properties`
-    `get_stored_properties`

The list of supported properties are  just before the __init__ method of the `PropertyCollector`.

## General discussion on the properties

Properties are divided in three domains/classes (let's call them `DomainPropertyClass`):

-   `GlobalProperty`: pbc, cell...
-   `IntraSiteProperty`: sites, charge, magnetization...
-   `InterSiteProperty`: still to be defined.

In principle, a property should be defined in this format:

```python
class GenericProperty(DomainPropertyClass):
    """
    The generic property. 
    """
    default_kind_threshold: float = 1e-3
    value: List[float] = Field(default=None)

    @validator("value", pre=True, always=True)
    def validate_property_value(cls,value,values):
        ... #validation procedure on the value
        return value
```
Required attributes to be defined are:

-   `default_kind_threshold`, needed for the automatic kinds generation;
-   `value`, which express the value of the property (site-wise if `DomainPropertyClass`=`IntraSiteProperty`).

It is possible to define the validator in such a way to initialize the property by default, even if it is not set in the property dictionary. For example, this can be useul for masses, which we do not define everytime. This are then taken from the tabulated values with respect to the `symbols` property. Similar thing happens for `pbc`, set [True,True,True]*3 by default. 
These are called `derived_properties`, and are set to *None* in the __init__ of `PropertyCollector`. Then this triggers the validation to set the default values. The logic can be improved for sure.