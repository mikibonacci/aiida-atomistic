from aiida import orm, load_profile
load_profile()

from aiida_atomistic.data.structure import StructureData

properties_dict = {
    "cell":{"value":[[3.5, 0.0, 0.0], [0.0, 3.5, 0.0], [0.0, 0.0, 3.5]]},
    "pbc":{"value":[True,True,True]},
    "positions":{"value":[[0.0, 0.0, 0.0],[1.5, 1.5, 1.5]]},
    "symbols":{"value":["Li","Li"]},
    }

structure = StructureData(properties = properties_dict)
print(f"The cell property class: \n{structure.properties.cell}")
print(f"The cell property value: \n{structure.properties.cell.value}")
print(f"The cell property domain: \n{structure.properties.cell.domain}\n\n")

print(f"The positions property class: \n{structure.properties.positions}")
print(f"The positions property value: \n{structure.properties.positions.value}\n")
print(f"The positions property domain: \n{structure.properties.positions.domain}\n\n")

print(f"The whole list of currently supported properties is: \n{StructureData().properties.get_supported_properties()}\n")
print(f"Stored properties are: \n{structure.properties.get_stored_properties()}")