# User guide on the StructureData class

The atomistic `StructureData` class is basically an enhanced version of the `orm.StructureData`, which was implemented in `aiida-core`. 
Relevant changes are:

- introduction of the `properties` attribute, used to store all the properties associated to the crystal structure;
- dropped the kind-based definition of the atoms, *no more supported* in favour of a code-agnostic site-based definition of the properties;
- the StructureData node is now really a *data container*, meaning that we do not have methods to modify it after its creation, i.e. it is *immutable* even before we store the node in the AiiDA database; 
explanation on this decisions can be found in the following.


<div style="border:2px solid #f7d117; padding: 10px; margin: 10px 0;">
    <strong>Site-based definition of properties:</strong> this simplifies multiple properties defintion and respect the philosophy of a code-agnostic data for the structure. The kinds determination can be done using the built-in `get_kinds()` method of the StructureData. It is also possible to provide a user-defined set of kinds via *tags*.
</div>

### Properties
Properties are divided in three main domains:  *global*, *intra-site*, and *inter-site*, e.g.:

global:

  - cell
  - periodic boundary conditions (PBC)

intra-site:

  - positions
  - symbols 
  - masses
  - charge
  - magnetization - TOBE added
  - Hubbard U parameters - TOBE added

inter-site:

  - Hubbard V parameters - TOBE added 

Some of these properties are related to the sites/atoms (e.g. atomic positions, symbols, electronic charge) and some are related to the whole structure (e.g. PBC, cell). So, each property will have an attribute `domain`, which can be "intra-site", "inter-site", "global". 

### Custom properties
The possibility to have user defined custom properties is discussed in another section (TOBE added).

## The first StructureData instance
One of the principle of the new StructureData is the fact that it is "just" a container of the information about a given structure: this means that, after that instances of this class are immutable. After the initialization, it is not possible to change the stored properties.

Properties should be contained in a dictionary, where the value of each defined property is defined under the corresponding dictionary, under the key `value`:

```python
from aiida import orm, load_profile
load_profile()

from aiida_atomistic.data.structure import StructureData

properties_dict = {
    "cell":{"value":[[3.5, 0.0, 0.0], [0.0, 3.5, 0.0], [0.0, 0.0, 3.5]]},
    "pbc":{"value":[True,True,True]},
    "positions":{"value":[[0.0, 0.0, 0.0],[1.5, 1.5, 1.5]]},
    "symbols":{"value":["Li","Li"]},
    }
```

Then we can initialise the StructureData instance:

```python
structure = StructureData(properties = properties_dict)
print(f"The cell property class: \n{structure.properties.cell}\n")
print(f"The cell property value: \n{structure.properties.cell.value}\n")
print(f"The cell property domain: \n{structure.properties.cell.domain}\n")

print(f"The positions property class: \n{structure.properties.positions}\n")
print(f"The positions property value: \n{structure.properties.positions.value}\n")
print(f"The positions property domain: \n{structure.properties.positions.domain}\n")
```

A list of supported and stored properties can be printed:

```python
print(f"The whole list of currently supported properties is: \n{StructureData().properties.get_supported_properties()}")
print(f"Stored properties are: \n{structure.properties.get_stored_properties()}")
```

## StructureData as a data container - immutability

We already anticipated that the StructureData is just a data container, .i.e. is immutable. This is a safety measure needed to 
avoid unpredicted behavior of a step-by-step data manipulation, which moreover may introduce incosistencies among the various properties.
In this way, only an initial consistency check can be performed among the whole set of defined properties. 

The StructureData is a *read-only* type of Data. 
All of the following command will produce an exception:

```python
structure.properties.cell.value = [[1,2,3],[1,2,3],[1,2,3]]
structure.properties.cell = [[1,2,3],[1,2,3],[1,2,3]]
```

## The `to_dict()` method

A crucial aspect of the new `StructureData` is that it is immutable even if the node is not stored, i.e. the API does not support on-the-fly or interactive modifications (it will raise errors). This helps in avoiding unexpected 
behaviour coming from a step-by-step defintion of the structure, e.g. incosistencies between properties definitions, which are then not cross-checked again.

To modify the related properties, one has to define a new `StructureData` instance by scratch.
To make user life simpler, we provide a `to_dict` method, which can be used to generate the properties dictionary:

```python
structure.to_dict()
```

The dictionary can be changed and used directly to generate a new instance:

```python
new_properties_dict = structure.to_dict()
new_properties_dict["pbc"] = {"value":[True,True,False]}
new_properties_dict["cell"]["value"][2] = [0,0,15]

new_structure = StructureData(properties=new_properties_dict)

print(f"The cell property value: \n{new_structure.properties.cell.value}\n")
```

## Kinds

It is possible to define kinds as a property:

```python
properties = {'cell': {'value': [[3.5, 0.0, 0.0], [0.0, 3.5, 0.0], [0.0, 0.0, 3.5]]},
 'pbc': {'value': [True, True, True]},
 'positions': {'value': [[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]]},
 'symbols': {'value': ['Li', 'Li']},
 'mass': {'value': [6.941, 6.941]},
 'charge': {'value': [1.0, 0.0]},
 'kinds': {'value': ['Li0', 'Li1']}}
```

A consistency check is done to see if we have not defined sites with different value of the properties to have the same kind. This would create inconsistencies in plugins.

## Automatic kinds detection

By default, if no kinds are provided in the properties, these will be generated automatically and put in the StructureData properties. The only drawback is that then properties value may change with respect to the default associated threshold. 
It is possible to avoid this by setting `allow_kinds`=False when we generate the StructureData instance.

```python
structure_with_kinds = StructureData(properties=new_properties, allow_kinds=False)
```

### The `get_kinds()` method

It is possible to get a list of kinds using the `get_kinds` method. 
This will generate the corresponding predicted kinds for all the properties (the "intra-site" ones) 
and then generate the list of global different kinds.

This method should be used in the plugins which requires a kind-based definition of properties, e.g. the aiida-quantumespresso one.

```python
unit_cell = [[3.5, 0.0, 0.0], [0.0, 3.5, 0.0], [0.0, 0.0, 3.5]]
atomic_positions = [[0.0, 0.0, 0.0],[1.5, 1.5, 1.5]]
symbols = ["Li"]*2
mass = [6.941,6.941]
charge = [1,0]

properties = {
    "cell":{"value":unit_cell},
    "pbc":{"value":[True,True,True]},
    "positions":{"value":atomic_positions,},
    "symbols":{"value":symbols},
    "mass":{"value":mass,},
    "charge":{"value":charge}
    }

structure = StructureData(
        properties=properties
        )
kinds = structure.get_kinds()

print(kinds)
```

**Please note**: the kinds generation done in this way will change the value of the properties in such a way to match the tolerance threshold for each property. See below the instructions on how to change the threshold for a given property.

#### Specification of default threshold for the kinds
It is possible to specify a custom threshold for a given property, if needed.
See the following example:

```python
print(structure.properties.charge.default_kind_threshold)

kinds = structure.get_kinds(custom_thr={"charge":2})
print(kinds)
```

#### Specification of `kind_tags`

We can assign tags to each atom, in such a way to override results of the `get_kinds` method. If we define a tag for 
each atom of the structure, the method will return unchanged value of the properties
with the desired tags.

```python
kinds = structure.get_kinds(kind_tags=["Li1","Li2"])

print(kinds)
```

It is possible also to exclude one property, when we determine kinds (maybe we ignore it in the plugin):

```python
kinds = structure.get_kinds(exclude=["charge"])

print(kinds)
```

It is possible to combine the `to_dict` and the `get_kinds` methods, in such a way to have a ready-to-use dictionary with also the kinds, automatically generated:

```python
new_properties = structure.to_dict(generate_kinds= True, kinds_exclude=['mass'],kinds_thresholds={"charge":1.5})
print(new_properties)

structure_with_kinds = StructureData(properties=new_properties)
structure_with_kinds.properties.kinds
```

### The `to_legacy_structuredata` method

Used to obtain the orm.StructureData from the atomistic one. Temporary method used to be able to run easily aiida-quantumespresso, aiida-pseudo.