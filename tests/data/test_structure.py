from ase.build import bulk
import numpy as np
import pytest

from aiida_atomistic.data.structure.structure import StructureData, StructureDataMutable
from aiida_atomistic.data.structure.site import SiteImmutable

from pydantic import ValidationError

"""
General tests for the atomistic StructureData.
The comments the test categories should be replaced by the pytest.mark in the future.
"""

# StructureData initialization:

def test_structure_initialization(example_structure_dict):
    """
    Testing that the StructureDataMutable is initialized correctly when:
    (1) nothing is provided;
    (2) properties are provided.
    """

    # (1.1) Empty StructureDataMutable
    structure = StructureDataMutable()

    assert isinstance(
        structure, StructureDataMutable
    ), f"Expected type for empty StructureDataMutable: {type(StructureDataMutable)}, \
                                            received: {type(structure)}"

    # (1.2) Empty StructureData: cannot be done
    with pytest.raises(ValidationError):
        structure = StructureData()

    # (2)
    for structure_type in [StructureDataMutable, StructureData]:
        structure = structure_type(**example_structure_dict)

        assert isinstance(
            structure, structure_type
        ), f"Expected type: {type(structure_type)}, \
                                            received: {type(structure)}"


# StructureData methods:

def test_dict(example_structure_dict):
    """
    Testing that the StructureData.to_dict() method works properly.

    NB: if pbc and cell are  not provided, this test will except, as it will then define the default pbc and cell.
    """
    for structure_type in [StructureDataMutable, StructureData]:
        structure = structure_type(**example_structure_dict)

        returned_dict = structure.to_dict()

        for derived_property in structure.properties.model_computed_fields.keys():
            returned_dict.pop(derived_property, None)
        for property_to_delete in ["custom", "tot_charge", "tot_magnetization"]:
            returned_dict.pop(property_to_delete, None)

        assert (
            returned_dict == example_structure_dict
        ), f"The dictionary returned by the method, {returned_dict}, \
                                                is different from the initial one: {example_structure_dict}"


def test_structure_ASE_initialization():
    """
    Testing that the StructureData/StructureDataMutable is initialized correctly when ASE Atoms object is provided.
    """

    atoms = bulk("Cu", "fcc", a=3.6)
    for structure_type in [StructureDataMutable, StructureData]:
        structure = structure_type.from_ase(atoms)

        assert isinstance(structure, structure_type)

    atoms = bulk('Cu', 'fcc', a=3.6)
    atoms.set_initial_charges([1,])
    atoms.set_initial_magnetic_moments([[0,0,1]])
    for structure_type in [StructureDataMutable, StructureData]:
        structure = structure_type.from_ase(atoms)

        assert structure.properties.charges == [1]
        assert structure.properties.magmoms == [[0,0,1]]

def test_structure_Pymatgen_initialization():
    """
    Testing that the StructureData/StructureDataMutable is initialized correctly when Pymatgen object is provided.
    """

    from pymatgen.core import Lattice, Structure, Molecule


    coords = [[0, 0, 0], [0.75,0.5,0.75]]
    lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120,
                                beta=90, gamma=60)

    struct = Structure(lattice, ["Si", "Si"], coords)
    struct.sites[0].properties["charge"]=1

    for structure_type in [StructureDataMutable, StructureData]:
        structure = structure_type.from_pymatgen(struct)

        assert structure.properties.charges == [1, 0]
        assert structure.properties.magmoms == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

def test_mutability():
    atoms = bulk("Cu", "fcc", a=3.6)
    # test StructureData
    s = StructureData.from_ase(atoms)

    assert isinstance(s.properties.pbc, list)
    assert any(s.properties.pbc)
    assert np.array_equal(
        s.properties.cell, np.array([[0.0, 1.8, 1.8], [1.8, 0.0, 1.8], [1.8, 1.8, 0.0]])
    )
    assert isinstance(s.properties.sites[0], SiteImmutable)

    with pytest.raises(ValueError):
        s.properties.pbc[0] = False

    with pytest.raises(ValidationError):
        s.properties.pbc = [True, False, True]

    # test StructureDataMutable
    m = StructureDataMutable.from_ase(atoms)

    assert isinstance(m.properties.pbc, list)
    assert any(m.properties.pbc)
    assert np.array_equal(
        m.properties.cell, [[0.0, 1.8, 1.8], [1.8, 0.0, 1.8], [1.8, 1.8, 0.0]])
    assert isinstance(m.properties.sites[0], SiteImmutable)

    # test StructureDataMutable mutability

    assert np.array_equal(m.properties.pbc,np.array([True, True, True]))

    m.set_pbc([False, False, False])
    assert not any(m.properties.pbc)

    # check StructureData and StructureDataMutable give the same properties.
    # in this way I check that it works well.
    m.set_pbc([True, True, True])

    returned_dict = s.to_dict()

    assert returned_dict == m.to_dict()

    # check append_atom works properly
    m.add_atom(
        {
            "symbols": "Cu",
            "mass": 63.546,
            "kind_name": "Cu",
            "positions": [1.0, 0.0, -1.0],
            "charge": 0.0,
            "magmom": [0,0,0],
        },
        index=0,
    )

    assert np.array_equal(m.get_charges(), np.array([0,0]))

def test_computed_fields(example_structure_dict):
    for structure_type in [StructureDataMutable, StructureData]:
        structure = structure_type(**example_structure_dict)

        assert structure.properties.magmoms == [[0,0,0]]
        assert structure.properties.charges == [1.0]
        assert structure.properties.cell_volume == 11.664000000000001
        assert structure.properties.dimensionality == {'dim': 3, 'label': 'volume', 'value': 11.664000000000001}

        if isinstance(structure, StructureDataMutable):
            structure.add_atom(
            {
                "symbols": "Cu",
                "mass": 63.546,
                "kind_name": "Cu",
                "positions": [1.0, 0.0, -1.0],
                "charge": 0.0,
                "magmom": [0,0,0],
            },
            index=0,
            )
            assert structure.properties.charges == [0.0, 1.0]


def test_model_validator(example_wrong_structure_dict,example_nomass_structure_dict):
    for structure_type in [StructureDataMutable, StructureData]:
        if isinstance(structure_type, StructureData):
            with pytest.raises(ValidationError):
                structure = structure_type(**example_wrong_structure_dict)
        elif isinstance(structure_type, StructureDataMutable):
            structure = structure_type(**example_wrong_structure_dict)

        structure = structure_type(**example_nomass_structure_dict)
        assert structure.properties.masses == [63.546]
        assert structure.properties.sites[0].mass == 63.546



## Test the get_kinds() method.

@pytest.mark.skip
@pytest.fixture
def kinds_properties():
    """
    Return the dictionary of properties as to be used in the tests about the get_kinds() method.
    """
    unit_cell = [[3.5, 0.0, 0.0], [0.0, 3.5, 0.0], [0.0, 0.0, 3.5]]
    atomic_positions = [
        [0.0, 0.0, 0.0],
        [1.5, 1.5, 1.5],
        [1.5, 2.5, 1.5],
        [1.5, 1.5, 2.5],
    ]
    symbols = ["Li"] * 2 + ["Cu"] * 2
    mass = [6.941] * 2 + [63.546] * 2
    charge = [1, 0.5, 0, 0]

    properties = {
        "cell": {"value": unit_cell},
        "pbc": {"value": [True, True, True]},
        "positions": {
            "value": atomic_positions,
        },
        "symbols": {"value": symbols},
        "mass": {
            "value": mass,
        },
        "charge": {"value": charge},
    }

    return properties

def test_get_kinds(example_structure_dict_for_kinds, complex_example_structure_dict_for_kinds):

    # (1) trivial system, defaults thr
    for structure_type in [StructureData, StructureDataMutable]:
        structure = structure_type(**example_structure_dict_for_kinds)

        new_structure = structure_type(**structure.to_dict(detect_kinds=True))

        assert new_structure.properties.kinds == ['Fe0', 'Fe1']
        assert new_structure.properties.magmoms == [[2.5, 0.1, 0.1], [2.4, 0.1, 0.1]]

    # (2) complex system, defaults thr
    for structure_type in [StructureData, StructureDataMutable]:
        structure = structure_type(**complex_example_structure_dict_for_kinds)

        new_structure = structure_type(**structure.to_dict(detect_kinds=True))

        assert new_structure.properties.kinds == ['Mn0', 'Mn1', 'Mn0', 'Mn1', 'Mn4', 'Mn4', 'Sn6', 'Sn6']
        assert new_structure.properties.magmoms == [[1.5000000000000002, 2.598076211353316, 0.0],
                                                    [-3.0, 0.0, 0.0],
                                                    [1.5000000000000002, 2.598076211353316, 0.0],
                                                    [-3.0, 0.0, 0.0],
                                                    [1.4999999999999998, -2.598076211353316, 0.0],
                                                    [1.4999999999999998, -2.598076211353316, 0.0],
                                                    [0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0]]

def test_alloy(example_structure_dict_alloy):

    for structure_type in [StructureData, StructureDataMutable]:
        structure = structure_type(**example_structure_dict_alloy)

        assert structure.properties.masses == [45.263768999999996]
        assert structure.properties.symbols == ["CuAl"]
