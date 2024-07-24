# `aiida-atomistic` package


AiiDA plugin which contains data and methods for atomistic simulations within AiiDA.


## Installation

```shell
git clone https://github.com/aiidateam/aiida-atomistic .
pip install ./aiida-atomistic
verdi quicksetup  # better to set up a new profile
verdi plugin list aiida.calculations  # should now show your calclulation plugins
```


## Usage

Here goes a complete example of how to submit a test calculation using this plugin.

A quick demo of how to submit a calculation:
```shell
verdi daemon start     # make sure the daemon is running
cd examples
./example_01.py        # run test calculation
verdi process list -a  # check record of calculation
```

The plugin also includes verdi commands to inspect its data types:
```shell
verdi data atomistic list
verdi data atomistic export <PK>
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
