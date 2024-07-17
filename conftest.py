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
        "weight",
    ]


@pytest.fixture
def example_structure_dict():
    """
    Return the dictionary of properties as to be used in the standards tests.
    """
    structure_dict = {
        "pbc": (True, True, True),
        "cell": [[0.0, 1.8, 1.8], [1.8, 0.0, 1.8], [1.8, 1.8, 0.0]],
        "sites": [
            {
                "symbol": "Cu",
                "kind_name": "Cu2",
                "position": [0.0, 0.0, 0.0],
                "mass": 63.546,
                "charge": 1.0,
                "magmom": 0.0,
            }
        ],
    }

    return structure_dict
