# How to migrate your plugin to use the `atomistic` package

In the following we present guidelines on how to migrate your existing plugin to use the new AiiDA data types delivered in the `aiida-atomistic` code.
This mainly has to do with the new `properties` attribute of new atomistic `StructureData` nodes.

## What changes from the `orm` to the `atomistic` StructureData

There are two main differences which you should take into account when migrating:

- `properties` attribute
- immutability of `StructureData`

The `properties` attribute in the new `atomistic` StructureData replaces the previous `orm` implementation. It allows you to store additional information associated with each site/atom in the structure. To migrate your plugin, you need to update your code to access and manipulate the `properties` attribute instead of the previous methods.
All the properties are now under the `properties` attribute, even the old ones.

- pbc
- cell
- sites

... and so on. A full list of properties can be accessed via the *classmethod* `get_property_names`.

Additionally, the `StructureData` nodes in the `atomistic` module are now immutable. This means that you cannot modify the structure once it is created. If you need to make changes, you should create a new `StructureDataMutable` which allows you to modify the properties. This can be done via the `to_mutable` method. Once finished, you can invoke the `to_immutable` method of the `StructureDataMutable` to have the update instance of the `StructureData` class. For further details, please have a look at the dedicated section (link).


### Backward compatiblity

For some backward compatibility support, refer to the documentation at [mikibonacci.github.io/aiida-atomistic/structuredata#backward-compatibility-support](mikibonacci.github.io/aiida-atomistic/structuredata#backward-compatibility-support).
The support for the old `orm.StructureData` is meant to be dropped soon.


## How a plugin should behave: parsing the properties for input file generation

When using the `StructureData` in your plugin, the idea is that each defined properties should be used in the calculation. The rationale is that in this way we have no ambiguity about if a property was used or not in the job.
For example, if we use a structure with defined `magmoms`, the related DFT simulation submitted should be a magnetic one. It would sound confusing if a non-magnetic calculation was done using a magnetic structure.

What if your plugin does not support magnetic calculations but the structure contains `magmoms` in its properties? There are two possibilities:

- exception
- warning plus calcfunction.

Let's describe more on the two approaches.

esempio di codice nei due casi + calcfunction per stripe.... ANZI METTI UN METODO PER FARE LO STRIPE? CI STA... CON ANCHE LA POSSIBILITA" DI RIFARE I KINDS....




==============================================================================================
- orm.StructureData ==> LegacyStructureData     to_atomistic!
- https://aiidateam.github.io/aiida-atomistic/structuredata#backward-compatibility-support
- everything is under properties
- how to add one property to all kinds? let's put a new method which iterates over the sites with the same kinds!!!!!
- kind ==> kind_name
