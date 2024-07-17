import functools

from aiida.orm.nodes.data import Data

from .core import StructureDataCore


class StructureData(Data, StructureDataCore):
    """
    The idea is that this class has the same as StructureDataMutable, but does not allow methods to change it.
    We should find a way to hide that methods. or better, we can provide a MixinClass to the helper.
    """

    def __init__(self, data: dict):
        StructureDataCore.__init__(self, data)
        Data.__init__(self)

        for prop, value in self.data.items():
            self.base.attributes.set(prop, value)


def immutability_cloak(attribute):
    @functools.wraps(attribute)
    def wrapper_immutability_cloak(self, value):
        if isinstance(self, StructureData):
            from aiida.common.exceptions import ModificationNotAllowed

            raise ModificationNotAllowed(
                "The StructureData object cannot be modified, it is immutable.\n \
                If you want to modify a structure object, use StructureDataMutable object and then\n \
                transform it into StructureData node via the `to_structuredata` method."
            )
        else:
            pass

    return wrapper_immutability_cloak
