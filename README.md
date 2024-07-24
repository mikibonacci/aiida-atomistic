[![Build Status][ci-badge]][ci-link]
[![Coverage Status][cov-badge]][cov-link]
[![Docs status][docs-badge]][docs-link]
[![PyPI version][pypi-badge]][pypi-link]

# aiida-atomistic

AiiDA plugin which contains data and methods for atomistic simulations.

## Installation

```shell
git clone https://github.com/aiidateam/aiida-atomistic .
pip install ./aiida-atomistic
verdi quicksetup  # better to set up a new profile
verdi plugin list aiida.data  # should now show your data plugins
```

## Development

```shell
git clone https://github.com/aiidateam/aiida-atomistic .
cd aiida-atomistic
pip install --upgrade pip
pip install -e .[pre-commit,testing]  # install extra dependencies
pre-commit install  # install pre-commit hooks
pytest -v  # discover and run all tests
```
See the [developer guide](http://aiida-atomistic.readthedocs.io/en/latest/developer_guide/index.html) for more information.

## License

MIT

## Contact

mikibonacci@psi.ch


[ci-badge]: https://github.com/aiidateam/aiida-atomistic/workflows/ci/badge.svg?branch=master
[ci-link]: https://github.com/aiidateam/aiida-atomistic/actions
[cov-badge]: https://coveralls.io/repos/github/aiidateam/aiida-atomistic/badge.svg?branch=master
[cov-link]: https://coveralls.io/github/aiidateam/aiida-atomistic?branch=master
[docs-badge]: https://readthedocs.org/projects/aiida-atomistic/badge
[docs-link]: http://aiida-atomistic.readthedocs.io/
[pypi-badge]: https://badge.fury.io/py/aiida-atomistic.svg
[pypi-link]: https://badge.fury.io/py/aiida-atomistic
