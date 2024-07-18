from aiida import orm, load_profile
load_profile()

from aiida_atomistic.data.structure import StructureData

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

print("Kinds: ", kinds)

print("Charge default threshold: ",structure.properties.charge.default_kind_threshold)

kinds = structure.get_kinds(custom_thr={"charge":2})
print("New kinds with updated charge threshold (2): ", kinds)

kinds = structure.get_kinds(exclude=["charge"])
print("New kinds excluding the charge property: ", kinds)
