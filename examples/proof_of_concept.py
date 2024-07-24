# # The new `StructureData` API

# # The `atomistic.StructureData` and `atomistic.StructureDataMutable` classes

# Two main rules: (i) immutability, and (ii) site-based. This means that our node will be just a container of the crystal structure + properties, and it cannot really be modified in any way.
# We will only provide `from_*`, `get_*` and `to_*` methods. Each site property will be defined as site-based and not kind-based, at variance with the old `orm.StructureData`. Kinds now can be defined as a property of each site (`kind_name`).
# The idea is to provide another python class which is just the mutable version of the `atomistic.StructureData` used to build, manipulate the crystal structure before the effective AiiDA node initialization. For now, let's call this class `StructureDataMutable`. This two classes have the same data structure, i.e. the same `properties` and the same `from_*`, `get_*` and `to_*` methods. The only difference is that the `atomistic.StructureDataMutable` has also `set_*` methods which can be used to mutate the properties. **Rule**: no property can be modified directly (i.e. accessing it); this is useful to avoid the introduction of inconsistencies in the structure instance.


# # How to initialize the `StructureData`(s)

# As both `StructureData` and `StructureDataMutable` share the same data structure, they also share the same constructor input parameter, which is just a python dictionary. The format of this dictionary exactly reflects how the data are store in the AiiDA database:

# +
from aiida_atomistic import StructureData
from aiida_atomistic import StructureDataMutable

from aiida import load_profile
load_profile()

structure_dict = {
    'cell':[[2.75,2.75,0],[0,2.75,2.75],[2.75,0,2.75]],
    'pbc': [True,True,True],
    'sites':[
        {
            'symbol':'Si',
            'position':[3/4, 3/4, 3/4],
        },
        {
            'symbol':'Si',
            'position':[1/2, 1/2, 1/2],
        },
    ],
}

mutable_structure = StructureDataMutable(**structure_dict)
structure = StructureData(**structure_dict)
# -

# When this dictionary is provided to the constructor, validation check for each of the provided property is done (**for now, only pbc and cell**).
# Then, you can access the properties directly:

# +
print("immutable pbc: ", structure.pbc)
print("mutable pbc: ", structure.pbc)

print("immutable cell: ", structure.cell)
print("mutable cell: ", structure.cell)

print("immutable sites: ", structure.sites)
print("mutable sites: ", structure.sites)
# -

# the expected output is:
#
# immutable pbc:  [ True  True  True]
#
# mutable pbc:  [ True  True  True]
#
# immutable cell:  [[2.75 2.75 0.  ]
#  [0.   2.75 2.75]
#  [2.75 0.   2.75]]
#
# mutable cell:  [[2.75 2.75 0.  ]
#  [0.   2.75 2.75]
#  [2.75 0.   2.75]]
#
# immutable sites:  [<Site: kind name 'Si' @ 0.75,0.75,0.75>, <Site: kind name 'Si' @ 0.5,0.5,0.5>]
#
# mutable sites:  [<Site: kind name 'Si' @ 0.75,0.75,0.75>, <Site: kind name 'Si' @ 0.5,0.5,0.5>]

# To inspect the properties of a single site, we can access it:

print(structure.sites[0].symbol,structure.sites[0].position) # output: Si [0.75 0.75 0.75]

# All the properties can be accessed via tab completion, and a list of the supported properties can be accessed via `structure.get_property_names()`.
# For now, other supported properties are `charge` (not yet `tot_charge`), `kind_name`, `mass`.
# For example, we can initialize a charged structure in this way:

# +
structure_dict = {
    'cell':[[2.75,2.75,0],[0,2.75,2.75],[2.75,0,2.75]],
    'pbc': [True,True,True],
    'sites':[
        {
            'symbol':'Si',
            'position':[3/4, 3/4, 3/4],
            'charge': +1,
            'kind_name': 'Si2',
        },
        {
            'symbol':'Si',
            'position':[1/2, 1/2, 1/2],
            'kind_name': 'Si1',
        },
    ],
}

mutable_structure = StructureDataMutable(**structure_dict)
structure = StructureData(**structure_dict)
# -

# then, `structure.sites[0].charge` will be equal to 1. When the plugins will be adapted, with this information we can build the correct input file for the corresponding quantum engine.

# ## Initialization from ASE or Pymatgen

# If we already have an ASE Atoms or a Pymatgen Structure object, we can use the `from_ase` and `from_pymatgen` methods:

# +
from ase.build import bulk
atoms = bulk('Cu', 'fcc', a=3.6)
atoms.set_initial_charges([1,])
atoms.set_tags(["2"])

mutable_structure = StructureDataMutable.from_ase(atoms)
structure = StructureData.from_ase(atoms)

structure.to_dict()
# -

# This should have as output:
#
# {'pbc': (True, True, True),
#  'cell': [[0.0, 1.8, 1.8], [1.8, 0.0, 1.8], [1.8, 1.8, 0.0]],
#  'sites': [{'symbol': 'Cu',
#    'kind_name': 'Cu2',
#    'position': [0.0, 0.0, 0.0],
#    'mass': 63.546,
#    'charge': 1.0,
#    'magmom': 0.0}]}
#
# This support also the properties like charges (coming soon: magmoms and so on). In the same way, for pymatgen we can proceed as follows:

# +
from pymatgen.core import Lattice, Structure, Molecule

coords = [[0, 0, 0], [0.75,0.5,0.75]]
lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120,
                                  beta=90, gamma=60)
struct = Structure(lattice, ["Si", "Si"], coords)

struct.add_oxidation_state_by_site([1,0])

mutable_structure = StructureDataMutable.from_pymatgen(struct)

mutable_structure.to_dict()
# -

# the output being:

{'pbc': (True, True, True),
 'cell': [[3.84, 0.0, 2.351321854362918e-16],
  [1.92, 2.7152900397563426, -1.919999999999999],
  [0.0, 0.0, 3.84]],
 'sites': [{'symbol': 'Si',
   'weights': 28.0855,
   'position': [0.0, 0.0, 0.0],
   'charge': 1,
   'kind_name': 'Si'},
  {'symbol': 'Si',
   'weights': 28.0855,
   'position': [3.84, 1.3576450198781713, 1.9200000000000006],
   'charge': 0,
   'kind_name': 'Si0'}]}

# Moreover, we also provide `to_ase` and `to_pymatgen` methods to obtain the corresponding instances. Also this methods for now only support charges, among the new properties.

# # Mutation of a structure

# Let's suppose you want to update some property in the `StructureData` before to use it in a calculation. You cannot. The way to go is either to use ASE or Pymatgen to modify you object and store it back into `StructureData`, or to use the `StructureDataMutable` and its mutation methods, and then convert it into `StructureData`.
# The latter method is the preferred one, as you then have support also for additional properties (to be implemented) like hubbard, which is not supported by the former.
# `StructureDataMutable` contains several `set_` methods and more, needed to update a structure:

old_structure = structure
old_structure.store()

# +
from aiida import orm
structure = orm.load_node(old_structure.pk)

mutable_structure = structure.to_mutable_structuredata()
mutable_structure.set_charges([1])
mutable_structure.set_kind_names(['Si2'])

new_structure = mutable_structure.to_structuredata()
# -

# Other available methods are `add_atom`, `pop_atom`, `update_site` and so on.
# Indeed, we can also start from scratch:

# +
mutable_structure = StructureDataMutable()
mutable_structure.set_cell([[0.0, 1.8, 1.8], [1.8, 0.0, 1.8], [1.8, 1.8, 0.0]])
mutable_structure.add_atom({
            'symbol':'Si',
            'position':[3/4, 3/4, 3/4],
            'charge': 1,
            'kind_name': 'Si2'
        })

mutable_structure.add_atom({
            'symbol':'Si',
            'position':[1/2, 1/2, 1/2],
            'charge': 0,
            'kind_name': 'Si1'
        })
# -


# # Slicing a structure

# It is possible to *slice* a structure, i.e. returning only a part of it (in terms of sites). Let's that you have an heterostructure and you want to obtain only the first layer, composed of the first 4 atoms over 10 total. This works for both `StructureDataMutable` and `StructureData` (we return a new `StructureData` instance).

sliced_structure = structure[:4]

# # Backward compatibility support

# We can use the `to_legacy` method to return the corresponding `orm.StructureData` instance, in case a given plugin does not yet support the new `StructureData`.
