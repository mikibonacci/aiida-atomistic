# How to migrate your plugin to use the `atomistic` package

- orm.StructureData ==> LegacyStructureData     to_atomistic!
- https://aiidateam.github.io/aiida-atomistic/structuredata#backward-compatibility-support
- everything is under properties
- how to add one property to all kinds? let's put a new method which iterates over the sites with the same kinds!!!!!
- kind ==> kind_name
