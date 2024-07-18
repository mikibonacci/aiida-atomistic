"""
aiida_atomistic

AiiDA plugin which contains data and methods for atomistic simulations
"""
from aiida_atomistic.data.structure.core import StructureData
from aiida_atomistic.data.structure.mutable import StructureDataMutable

__version__ = "0.1.0a0"

__all__ = [
    StructureData,
    StructureDataMutable
]
