"""pytest fixtures for simplified testing."""
import pytest

pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]


@pytest.fixture(scope="function", autouse=True)
def clear_database_auto(clear_database):  # pylint: disable=unused-argument
    """Automatically clear database in between tests."""


@pytest.fixture(scope="function")
def atomistic_code(aiida_local_code_factory):
    """Get a atomistic code."""
    return aiida_local_code_factory(executable="diff", entry_point="atomistic")


@pytest.fixture
def supported_properties():
    """
    Should be updated every time I add properties.
    """
    return [
        "cell",
        "pbc",
        "position",
        "symbol",
        "mass",
        "charge",
        "magmom",
        "kind_name",
        "weights",
    ]


@pytest.fixture
def example_structure_dict():
    """
    Return the dictionary of properties as to be used in the standards tests.
    """
    structure_dict = {
        "pbc": [True, True, True],
        "cell": [[0.0, 1.8, 1.8], [1.8, 0.0, 1.8], [1.8, 1.8, 0.0]],
        "sites": [
            {
                "symbol": "Cu",
                "kind_name": "Cu2",
                "position": [0.0, 0.0, 0.0],
                "mass": 63.546,
                "charge": 1.0,
                "magmom": [0.0,0.0,0.0],
                "weights": (1,)
            }
        ],
    }

    return structure_dict

@pytest.fixture
def example_nomass_structure_dict():
    """
    Return the dictionary of properties as to be used in the standards tests.
    """
    structure_dict = {
        "pbc": [True, True, True],
        "cell": [[0.0, 1.8, 1.8], [1.8, 0.0, 1.8], [1.8, 1.8, 0.0]],
        "sites": [
            {
                "symbol": "Cu",
                "kind_name": "Cu2",
                "position": [0.0, 0.0, 0.0],
                #"mass": 63.546,
                "charge": 1.0,
                "magmom": [0,0,0],
            }
        ],
    }

    return structure_dict

@pytest.fixture
def example_wrong_structure_dict():
    """
    Return the dictionary of properties as to be used in the standards tests.
    """
    structure_dict = {
        "pbc": [True, True, True],
        "cell": [[0.0, 1.8, 1.8], [1.8, 0.0, 1.8], [1.8, 1.8, 0.0]],
        "sites": [
            {
                "symbol": "Cu",
                "kind_name": "Cu2",
                "position": [0.0, 0.0, 0.0],
                "mass": 63.546,
                "charge": 1.0,
                "magmom": [0,0,0],
            },
            {
                "symbol": "Cu",
                "kind_name": "Cu2",
                "position": [0.0, 0.0, 0.0],
                "mass": 63.546,
                "charge": 1.0,
                "magmom": [0,0,0],
            }
        ],
    }

    return structure_dict

@pytest.fixture
def example_structure_dict_for_kinds():
    """
    Return the dictionary of properties as to be used in the standards tests.
    """
    structure_dict = {'pbc': (True, True, True),
        'cell': [[2.8403, 0.0, 1.7391821518091137e-16],
        [-1.7391821518091137e-16, 2.8403, 1.7391821518091137e-16],
        [0.0, 0.0, 2.8403]],
        'sites': [{'symbol': 'Fe',
        'mass': 55.845,
        'position': [0.0, 0.0, 0.0],
        'charge': 0.0,
        'magmom': [2.5, 0.1, 0.1],
        'kind_name': 'Fe'},
        {'symbol': 'Fe',
        'mass': 55.845,
        'position': [1.42015, 1.42015, 1.4201500000000002],
        'charge': 0.0,
        'magmom': [2.4, 0.1, 0.1],
        'kind_name': 'Fe'}]}

    return structure_dict

@pytest.fixture
def complex_example_structure_dict_for_kinds():
    """
    Return the dictionary of properties as to be used in the standards tests.
    the structure is here: aiida-atomistic/examples/structure/data/0.199_Mn3Sn.mcif
    """
    from pymatgen.core import Structure
    from aiida_atomistic import StructureData
    smag1 = Structure.from_file('./examples/structure/data/'+"0.199_Mn3Sn.mcif", primitive=True)

    return StructureData.from_pymatgen(smag1).to_dict()

@pytest.fixture
def example_structure_dict_alloy():
    """
    Return the dictionary of properties as to be used in the standards tests.
    """
    structure_dict ={
        'pbc': [True, True, True],
        'cell': [[0.0, 1.8, 1.8], [1.8, 0.0, 1.8], [1.8, 1.8, 0.0]],
        'sites': [{'symbol': 'CuAl',
        'position': [0.0, 0.0, 0.0],
        'weights': (0.5,0.5)
        }],}

    return structure_dict
