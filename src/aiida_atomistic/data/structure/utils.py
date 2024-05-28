import copy
import numpy as np
import functools

from aiida.common.constants import elements
from aiida.common.exceptions import UnsupportedSpeciesError

try:
    import ase  # noqa: F401
except ImportError:
    pass

try:
    import pymatgen.core  as core # noqa: F401
except ImportError:
    pass


# Threshold used to check if the mass of two different Site objects is the same.

_MASS_THRESHOLD = 1.0e-3
# Threshold to check if the sum is one or not
_SUM_THRESHOLD = 1.0e-6
# Default cell
_DEFAULT_CELL = ((0, 0, 0), (0, 0, 0), (0, 0, 0))

_valid_symbols = tuple(i['symbol'] for i in elements.values())
_atomic_masses = {el['symbol']: el['mass'] for el in elements.values()}
_atomic_numbers = {data['symbol']: num for num, data in elements.items()}


def _get_valid_cell(inputcell):
    """Return the cell in a valid format from a generic input.

    :raise ValueError: whenever the format is not valid.
    """
    try:
        the_cell = list(list(float(c) for c in i) for i in inputcell)
        if len(the_cell) != 3:
            raise ValueError
        if any(len(i) != 3 for i in the_cell):
            raise ValueError
    except (IndexError, ValueError, TypeError):
        raise ValueError('Cell must be a list of three vectors, each defined as a list of three coordinates.')

    return the_cell


def get_valid_pbc(inputpbc):
    """Return a list of three booleans for the periodic boundary conditions,
    in a valid format from a generic input.

    :raise ValueError: if the format is not valid.
    """
    if isinstance(inputpbc, bool):
        the_pbc = (inputpbc, inputpbc, inputpbc)
    elif hasattr(inputpbc, '__iter__'):
        # To manage numpy lists of bools, whose elements are of type numpy.bool_
        # and for which isinstance(i,bool) return False...
        if hasattr(inputpbc, 'tolist'):
            the_value = inputpbc.tolist()
        else:
            the_value = inputpbc
        if all(isinstance(i, bool) for i in the_value):
            if len(the_value) == 3:
                the_pbc = tuple(i for i in the_value)
            elif len(the_value) == 1:
                the_pbc = (the_value[0], the_value[0], the_value[0])
            else:
                raise ValueError('pbc length must be either one or three.')
        else:
            raise ValueError('pbc elements are not booleans.')
    else:
        raise ValueError('pbc must be a boolean or a list of three booleans.', inputpbc)

    return the_pbc


def has_ase():
    """:return: True if the ase module can be imported, False otherwise."""
    try:
        import ase  # noqa: F401
    except ImportError:
        return False
    return True


def has_pymatgen():
    """:return: True if the pymatgen module can be imported, False otherwise."""
    try:
        import pymatgen  # noqa: F401
    except ImportError:
        return False
    return True


def get_pymatgen_version():
    """:return: string with pymatgen version, None if can not import."""
    if not has_pymatgen():
        return None
    try:
        from pymatgen import __version__
    except ImportError:
        # this was changed in version 2022.0.3
        from pymatgen.core import __version__
    return __version__


def has_spglib():
    """:return: True if the spglib module can be imported, False otherwise."""
    try:
        import spglib  # noqa: F401
    except ImportError:
        return False
    return True


def calc_cell_volume(cell):
    """Compute the three-dimensional cell volume in Angstrom^3.

    :param cell: the cell vectors; the must be a 3x3 list of lists of floats
    :returns: the cell volume.
    """
    import numpy as np

    return np.abs(np.dot(cell[0], np.cross(cell[1], cell[2])))


def _create_symbols_tuple(symbols):
    """Returns a tuple with the symbols provided. If a string is provided,
    this is converted to a tuple with one single element.
    """
    if isinstance(symbols, str):
        symbols_list = (symbols,)
    else:
        symbols_list = tuple(symbols)
    return symbols_list


def _create_weights_tuple(weights):
    """Returns a tuple with the weights provided. If a number is provided,
    this is converted to a tuple with one single element.
    If None is provided, this is converted to the tuple (1.,)
    """
    import numbers

    if weights is None:
        weights_tuple = (1.0,)
    elif isinstance(weights, numbers.Number):
        weights_tuple = (weights,)
    else:
        weights_tuple = tuple(float(i) for i in weights)
    return weights_tuple


def create_automatic_kind_name(symbols, weights):
    """Create a string obtained with the symbols appended one
    after the other, without spaces, in alphabetical order;
    if the site has a vacancy, a X is appended at the end too.
    """
    sorted_symbol_list = list(set(symbols))
    sorted_symbol_list.sort()  # In-place sort
    name_string = ''.join(sorted_symbol_list)
    if has_vacancies(weights):
        name_string += 'X'
    return name_string


def validate_weights_tuple(weights_tuple, threshold):
    """Validates the weight of the atomic kinds.

    :raise: ValueError if the weights_tuple is not valid.

    :param weights_tuple: the tuple to validate. It must be a
            a tuple of floats (as created by :func:_create_weights_tuple).
    :param threshold: a float number used as a threshold to check that the sum
            of the weights is <= 1.

    If the sum is less than one, it means that there are vacancies.
    Each element of the list must be >= 0, and the sum must be <= 1.
    """
    w_sum = sum(weights_tuple)
    if any(i < 0.0 for i in weights_tuple) or (w_sum - 1.0 > threshold):
        raise ValueError('The weight list is not valid (each element must be positive, and the sum must be <= 1).')


def is_valid_symbol(symbol):
    """Validates the chemical symbol name.

    :return: True if the symbol is a valid chemical symbol (with correct
        capitalization), or the dummy X, False otherwise.

    Recognized symbols are for elements from hydrogen (Z=1) to lawrencium
    (Z=103). In addition, a dummy element unknown name (Z=0) is supported.
    """
    return symbol in _valid_symbols


def validate_symbols_tuple(symbols_tuple):
    """Used to validate whether the chemical species are valid.

    :param symbols_tuple: a tuple (or list) with the chemical symbols name.
    :raises: UnsupportedSpeciesError if any symbol in the tuple is not a valid chemical
        symbol (with correct capitalization).

    Refer also to the documentation of :func:is_valid_symbol
    """
    if len(symbols_tuple) == 0:
        valid = False
    else:
        valid = all(is_valid_symbol(sym) for sym in symbols_tuple)
    if not valid:
        raise UnsupportedSpeciesError(
            f'At least one element of the symbol list {symbols_tuple} has not been recognized.'
        )

def group_symbols(_list):
    """Group a list of symbols to a list containing the number of consecutive
    identical symbols, and the symbol itself.

    Examples
    --------
    * ``['Ba','Ti','O','O','O','Ba']`` will return
      ``[[1,'Ba'],[1,'Ti'],[3,'O'],[1,'Ba']]``

    * ``[ [ [1,'Ba'],[1,'Ti'] ],[ [1,'Ba'],[1,'Ti'] ] ]`` will return
      ``[[2, [ [1, 'Ba'], [1, 'Ti'] ] ]]``

    :param _list: a list of elements representing a chemical formula
    :return: a list of length-2 lists of the form [ multiplicity , element ]
    """
    the_list = copy.deepcopy(_list)
    the_list.reverse()
    grouped_list = [[1, the_list.pop()]]
    while the_list:
        elem = the_list.pop()
        if elem == grouped_list[-1][1]:
            # same symbol is repeated
            grouped_list[-1][0] += 1
        else:
            grouped_list.append([1, elem])

    return grouped_list


def get_formula_from_symbol_list(_list, separator=''):
    """Return a string with the formula obtained from the list of symbols.

    Examples
    --------
    * ``[[1,'Ba'],[1,'Ti'],[3,'O']]`` will return ``'BaTiO3'``
    * ``[[2, [ [1, 'Ba'], [1, 'Ti'] ] ]]`` will return ``'(BaTi)2'``

    :param _list: a list of symbols and multiplicities as obtained from
        the function group_symbols
    :param separator: a string used to concatenate symbols. Default empty.

    :return: a string
    """
    list_str = []
    for elem in _list:
        if elem[0] == 1:
            multiplicity_str = ''
        else:
            multiplicity_str = str(elem[0])

        if isinstance(elem[1], str):
            list_str.append(f'{elem[1]}{multiplicity_str}')
        elif elem[0] > 1:
            list_str.append(f'({get_formula_from_symbol_list(elem[1], separator=separator)}){multiplicity_str}')
        else:
            list_str.append(f'{get_formula_from_symbol_list(elem[1], separator=separator)}{multiplicity_str}')

    return separator.join(list_str)


def get_formula_group(symbol_list, separator=''):
    """Return a string with the chemical formula from a list of chemical symbols.
    The formula is written in a compact" way, i.e. trying to group as much as
    possible parts of the formula.

    .. note:: it works for instance very well if structure was obtained
        from an ASE supercell.

    Example of result:
    ``['Ba', 'Ti', 'O', 'O', 'O', 'Ba', 'Ti', 'O', 'O', 'O',
    'Ba', 'Ti', 'Ti', 'O', 'O', 'O']`` will return ``'(BaTiO3)2BaTi2O3'``.

    :param symbol_list: list of symbols
        (e.g. ['Ba','Ti','O','O','O'])
    :param separator: a string used to concatenate symbols. Default empty.
    :returns: a string with the chemical formula for the given structure.
    """

    def group_together(_list, group_size, offset):
        """:param _list: a list
        :param group_size: size of the groups
        :param offset: beginning grouping after offset elements
        :return : a list of lists made of groups of size group_size
            obtained by grouping list elements together
            The first elements (up to _list[offset-1]) are not grouped
        example:
            ``group_together(['O','Ba','Ti','Ba','Ti'],2,1) =
                ['O',['Ba','Ti'],['Ba','Ti']]``
        """
        the_list = copy.deepcopy(_list)
        the_list.reverse()
        grouped_list = []
        for _ in range(offset):
            grouped_list.append([the_list.pop()])

        while the_list:
            sub_list = []
            for _ in range(group_size):
                if the_list:
                    sub_list.append(the_list.pop())
            grouped_list.append(sub_list)

        return grouped_list

    def cleanout_symbol_list(_list):
        """:param _list: a list of groups of symbols and multiplicities
        :return : a list where all groups with multiplicity 1 have
            been reduced to minimum
        example: ``[[1,[[1,'Ba']]]]`` will return ``[[1,'Ba']]``
        """
        the_list = []
        for elem in _list:
            if elem[0] == 1 and isinstance(elem[1], list):
                the_list.extend(elem[1])
            else:
                the_list.append(elem)

        return the_list

    def group_together_symbols(_list, group_size):
        """Successive application of group_together, group_symbols and
        cleanout_symbol_list, in order to group a symbol list, scanning all
        possible offsets, for a given group size
        :param _list: the symbol list (see function group_symbols)
        :param group_size: the size of the groups
        :return the_symbol_list: the new grouped symbol list
        :return has_grouped: True if we grouped something
        """
        the_symbol_list = copy.deepcopy(_list)
        has_grouped = False
        offset = 0
        while not has_grouped and offset < group_size:
            grouped_list = group_together(the_symbol_list, group_size, offset)
            new_symbol_list = group_symbols(grouped_list)
            if len(new_symbol_list) < len(grouped_list):
                the_symbol_list = copy.deepcopy(new_symbol_list)
                the_symbol_list = cleanout_symbol_list(the_symbol_list)
                has_grouped = True
                # print get_formula_from_symbol_list(the_symbol_list)
            offset += 1

        return the_symbol_list, has_grouped

    def group_all_together_symbols(_list):
        """Successive application of the function group_together_symbols, to group
        a symbol list, scanning all possible offsets and group sizes
        :param _list: the symbol list (see function group_symbols)
        :return: the new grouped symbol list
        """
        has_finished = False
        group_size = 2
        the_symbol_list = copy.deepcopy(_list)

        while not has_finished and group_size <= len(_list) // 2:
            # try to group as much as possible by groups of size group_size
            the_symbol_list, has_grouped = group_together_symbols(the_symbol_list, group_size)
            has_finished = has_grouped
            group_size += 1
            # stop as soon as we managed to group something
            # or when the group_size is too big to get anything

        return the_symbol_list

    # initial grouping of the chemical symbols
    old_symbol_list = [-1]
    new_symbol_list = group_symbols(symbol_list)

    # successively apply the grouping procedure until the symbol list does not
    # change anymore
    while new_symbol_list != old_symbol_list:
        old_symbol_list = copy.deepcopy(new_symbol_list)
        new_symbol_list = group_all_together_symbols(old_symbol_list)

    return get_formula_from_symbol_list(new_symbol_list, separator=separator)


def get_formula(symbol_list, mode='hill', separator=''):
    """Return a string with the chemical formula.

    :param symbol_list: a list of symbols, e.g. ``['H','H','O']``
    :param mode: a string to specify how to generate the formula, can
        assume one of the following values:

        * 'hill' (default): count the number of atoms of each species,
          then use Hill notation, i.e. alphabetical order with C and H
          first if one or several C atom(s) is (are) present, e.g.
          ``['C','H','H','H','O','C','H','H','H']`` will return ``'C2H6O'``
          ``['S','O','O','H','O','H','O']``  will return ``'H2O4S'``
          From E. A. Hill, J. Am. Chem. Soc., 22 (8), pp 478-494 (1900)

        * 'hill_compact': same as hill but the number of atoms for each
          species is divided by the greatest common divisor of all of them, e.g.
          ``['C','H','H','H','O','C','H','H','H','O','O','O']``
          will return ``'CH3O2'``

        * 'reduce': group repeated symbols e.g.
          ``['Ba', 'Ti', 'O', 'O', 'O', 'Ba', 'Ti', 'O', 'O', 'O',
          'Ba', 'Ti', 'Ti', 'O', 'O', 'O']`` will return ``'BaTiO3BaTiO3BaTi2O3'``

        * 'group': will try to group as much as possible parts of the formula
          e.g.
          ``['Ba', 'Ti', 'O', 'O', 'O', 'Ba', 'Ti', 'O', 'O', 'O',
          'Ba', 'Ti', 'Ti', 'O', 'O', 'O']`` will return ``'(BaTiO3)2BaTi2O3'``

        * 'count': same as hill (i.e. one just counts the number
          of atoms of each species) without the re-ordering (take the
          order of the atomic sites), e.g.
          ``['Ba', 'Ti', 'O', 'O', 'O','Ba', 'Ti', 'O', 'O', 'O']``
          will return ``'Ba2Ti2O6'``

        * 'count_compact': same as count but the number of atoms
          for each species is divided by the greatest common divisor of
          all of them, e.g.
          ``['Ba', 'Ti', 'O', 'O', 'O','Ba', 'Ti', 'O', 'O', 'O']``
          will return ``'BaTiO3'``

    :param separator: a string used to concatenate symbols. Default empty.

    :return: a string with the formula

    .. note:: in modes reduce, group, count and count_compact, the
        initial order in which the atoms were appended by the user is
        used to group and/or order the symbols in the formula
    """
    if mode == 'group':
        return get_formula_group(symbol_list, separator=separator)

    # for hill and count cases, simply count the occurences of each
    # chemical symbol (with some re-ordering in hill)
    if mode in ['hill', 'hill_compact']:
        if 'C' in symbol_list:
            ordered_symbol_set = sorted(set(symbol_list), key=lambda elem: {'C': '0', 'H': '1'}.get(elem, elem))
        else:
            ordered_symbol_set = sorted(set(symbol_list))
        the_symbol_list = [[symbol_list.count(elem), elem] for elem in ordered_symbol_set]

    elif mode in ['count', 'count_compact']:
        ordered_symbol_indexes = sorted([symbol_list.index(elem) for elem in set(symbol_list)])
        ordered_symbol_set = [symbol_list[i] for i in ordered_symbol_indexes]
        the_symbol_list = [[symbol_list.count(elem), elem] for elem in ordered_symbol_set]

    elif mode == 'reduce':
        the_symbol_list = group_symbols(symbol_list)

    else:
        raise ValueError('Mode should be hill, hill_compact, group, reduce, count or count_compact')

    if mode in ['hill_compact', 'count_compact']:
        from math import gcd

        the_gcd = functools.reduce(gcd, [e[0] for e in the_symbol_list])
        the_symbol_list = [[e[0] // the_gcd, e[1]] for e in the_symbol_list]

    return get_formula_from_symbol_list(the_symbol_list, separator=separator)


def get_symbols_string(symbols, weights):
    """Return a string that tries to match as good as possible the symbols
    and weights. If there is only one symbol (no alloy) with 100%
    occupancy, just returns the symbol name. Otherwise, groups the full
    string in curly brackets, and try to write also the composition
    (with 2 precision only).
    If (sum of weights<1), we indicate it with the X symbol followed
    by 1-sum(weights) (still with 2 digits precision, so it can be 0.00)

    :param symbols: the symbols as obtained from <kind>._symbols
    :param weights: the weights as obtained from <kind>._weights

    .. note:: Note the difference with respect to the symbols and the
        symbol properties!
    """
    if len(symbols) == 1 and weights[0] == 1.0:
        return symbols[0]

    pieces = []
    for symbol, weight in zip(symbols, weights):
        pieces.append(f'{symbol}{weight:4.2f}')
    if has_vacancies(weights):
        pieces.append(f'X{1.0 - sum(weights):4.2f}')
    return f"{{{''.join(sorted(pieces))}}}"


def has_vacancies(weights):
    """Returns True if the sum of the weights is less than one.
    It uses the internal variable _SUM_THRESHOLD as a threshold.
    :param weights: the weights
    :return: a boolean
    """
    w_sum = sum(weights)
    return not 1.0 - w_sum < _SUM_THRESHOLD


def symop_ortho_from_fract(cell):
    """Creates a matrix for conversion from orthogonal to fractional
    coordinates.

    Taken from
    svn://www.crystallography.net/cod-tools/trunk/lib/perl5/Fractional.pm,
    revision 850.

    :param cell: array of cell parameters (three lengths and three angles)
    """
    import math

    import numpy

    a, b, c, alpha, beta, gamma = cell
    alpha, beta, gamma = [math.pi * x / 180 for x in [alpha, beta, gamma]]
    ca, cb, cg = [math.cos(x) for x in [alpha, beta, gamma]]
    sg = math.sin(gamma)

    return numpy.array(
        [
            [a, b * cg, c * cb],
            [0, b * sg, c * (ca - cb * cg) / sg],
            [0, 0, c * math.sqrt(sg * sg - ca * ca - cb * cb + 2 * ca * cb * cg) / sg],
        ]
    )


def symop_fract_from_ortho(cell):
    """Creates a matrix for conversion from fractional to orthogonal
    coordinates.

    Taken from
    svn://www.crystallography.net/cod-tools/trunk/lib/perl5/Fractional.pm,
    revision 850.

    :param cell: array of cell parameters (three lengths and three angles)
    """
    import math

    import numpy

    a, b, c, alpha, beta, gamma = cell
    alpha, beta, gamma = [math.pi * x / 180 for x in [alpha, beta, gamma]]
    ca, cb, cg = [math.cos(x) for x in [alpha, beta, gamma]]
    sg = math.sin(gamma)
    ctg = cg / sg
    D = math.sqrt(sg * sg - cb * cb - ca * ca + 2 * ca * cb * cg)  # noqa: N806

    return numpy.array(
        [
            [1.0 / a, -(1.0 / a) * ctg, (ca * cg - cb) / (a * D)],
            [0, 1.0 / (b * sg), -(ca - cb * cg) / (b * D * sg)],
            [0, 0, sg / (c * D)],
        ]
    )


def ase_refine_cell(aseatoms, **kwargs):
    """Detect the symmetry of the structure, remove symmetric atoms and
    refine unit cell.

    :param aseatoms: an ase.atoms.Atoms instance
    :param symprec: symmetry precision, used by spglib
    :return newase: refined cell with reduced set of atoms
    :return symmetry: a dictionary describing the symmetry space group
    """
    from ase.atoms import Atoms
    from spglib import get_symmetry_dataset, refine_cell

    spglib_tuple = (
        aseatoms.get_cell(),
        aseatoms.get_scaled_positions(),
        aseatoms.get_atomic_numbers(),
    )
    cell, positions, numbers = refine_cell(spglib_tuple, **kwargs)

    refined_atoms = (
        cell,
        positions,
        numbers,
    )
    sym_dataset = get_symmetry_dataset(refined_atoms, **kwargs)

    unique_numbers = []
    unique_positions = []

    for i in set(sym_dataset['equivalent_atoms']):
        unique_numbers.append(numbers[i])
        unique_positions.append(positions[i])

    unique_atoms = Atoms(unique_numbers, scaled_positions=unique_positions, cell=cell, pbc=True)

    return unique_atoms, {
        'hm': sym_dataset['international'],
        'hall': sym_dataset['hall'],
        'tables': sym_dataset['number'],
        'rotations': sym_dataset['rotations'],
        'translations': sym_dataset['translations'],
    }


def atom_kinds_to_html(atom_kind):
    """Construct in html format

    an alloy with 0.5 Ge, 0.4 Si and 0.1 vacancy is represented as
    Ge<sub>0.5</sub> + Si<sub>0.4</sub> + vacancy<sub>0.1</sub>

    Args:
    -----
        atom_kind: a string with the name of the atomic kind, as printed by
        kind.get_symbols_string(), e.g. Ba0.80Ca0.10X0.10

    Returns:
    --------
        html code for rendered formula
    """
    # Parse the formula (TODO can be made more robust though never fails if
    # it takes strings generated with kind.get_symbols_string())
    import re

    matched_elements = re.findall(r'([A-Z][a-z]*)([0-1][.[0-9]*]?)?', atom_kind)

    # Compose the html string
    html_formula_pieces = []

    for element in matched_elements:
        # replace element X by 'vacancy'
        species = element[0] if element[0] != 'X' else 'vacancy'
        weight = element[1] if element[1] != '' else None

        if weight is not None:
            html_formula_pieces.append(f'{species}<sub>{weight}</sub>')
        else:
            html_formula_pieces.append(species)

    html_formula = ' + '.join(html_formula_pieces)

    return html_formula


def to_kinds(property, symbols, thr: float = 0):
    """Called by the `get_kinds` function. 
    Get the kinds for a generic site property. Can also be overridden in the specific property.

    ### Search algorithm:

    Basically we compute the indexes array which locates each point in regions centered on our values, considering
    min(values) as reference and each region being of width=thr:
    
        indexes = np.array((prop_array-np.min(prop_array))/thr,dtype=int)
    
    To understand this, try to draw the problem considering prop_array=[1,2,3,4] and thr=0.5.
    This methods allows to efficiently clusterize the point using the defined threshold.
    
    At the end, we reorder the kinds from zero (to have ordered list like Li0, Li1...).
    Basically we define the set of unordered kinds, and the range(len(set(kinds))) being the group of orderd kinds.
    Then we basically do a mapping with the np.where().
    
    Args:
        thr (float, optional): the threshold to consider two atoms of the same element to be the same kind. 
            Defaults to structure.properties.<property>.default_kind_threshold.
            If thr==0, we just return different kind for each site with the original property value. This is 
            needed when we have tags for each site, in the get_kind method of StructureData.
        
    Returns:
        kinds_labels: array of kinds (as integers) associated to the charge property. they are integers so that in the `get_kinds()` method
                            can be used in the matrix representation (the k.T).
        kinds_values: list of the associated property value to each kind detected.
    """ 
    symbols_array = np.array(symbols)
    prop_array = np.array(property)
    
    
    if thr == 0:
        return np.array(range(len(prop_array))), prop_array
    
    # list for the value of the property for each generated kind.
    kinds_values = np.zeros(len(symbols_array))

    indexes = np.array((prop_array-np.min(prop_array))/thr,dtype=int)
    
    # Here we select the closest value present in the property values
    set_indexes = set(indexes)
    for index in set_indexes:
        where_index_in_indexes = np.where(indexes == index)[0]
        kinds_values[where_index_in_indexes] = np.min(prop_array[where_index_in_indexes])
    
    # here we reorder from zero the kinds.
    list_set_indexes = list(set_indexes)
    kinds_labels = np.zeros(len(symbols_array),dtype=int)
    for i in range(len(list_set_indexes)):
        kinds_labels[np.where(indexes==list_set_indexes[i])[0]] = i
    
    return kinds_labels, kinds_values

def get_kinds(structure, kind_tags=[], exclude=[], custom_thr={}):
    """
    Get the list of kinds, taking into account all the properties.
    If the list of kinds is already provided--> len(kind_tags)>0, we check the consistency of it 
    by computing the kinds with threshold=0 for each property. 
    
    
    Algorithm:
    it generated the kinds_list for each property separately in Step 1, then
    it creates the matrix k = k.T where the rows are the sites, the columns are the properties and each element
    is the corresponding kind for the given property and the given site:
    
    ```bash
              p1 p2 p3 
    site1 = | 1  1  2 | = kind1
    site2 = | 1  2  3 | = kind2
    site3 = | 2  2  3 | = kind3
    site4 = | 1  2  3 | = kind4
    ```

    In Step 2 it checks for the matrix which rows have the same numbers in the same order, i.e. recognize the different
    kinds considering all the properties. This is done by subtracting a row from the others and see if all the elements
    are zero, meaning that we have the same combination of kinds.
    
    In Step 3 we override the kinds with the kind_tags.

    Args:
        kind_tags (list, optional): list of kind names as user defined: in principle this input trigger a check in the kind 
                                    determination -> the mapping should be the same as obtained with get_kinds(kind_tags=[],) and
                                    all thresholds = 0. And this is what is done: `if not None in kind_tags: thr = 0`.
                                    For now we support also for only some selected kinds defined: ["kind1",None, ...] but with the same length as the symbols (sites).
        exclude (list, optional): list of properties to be excluded in the kind determination 
        custom_thr (dict, options): dictionary with the custom threshold for given properties (key: property, value: thr).
                                    if not provided, we fallback into the default threshold define in the property class.
        
    Returns:
        kinds_dictionary (dictionary): the associated per-site (and per-kind) value of the property. The structure of the dictionary is the one that you may 
                                    have in the `properties` dictionary input of the StructureData constructor. 
                                    We also provide the `kinds` property: list of kind-per-site to be used in a plugin which requires it. If kind tags are all decided, then we 
                                    do not compute anything and we return kind_tags and None. In this way, we know that we basically already defined 
                                    the kinds in our StructureData.
    
    Comments:
    
    - Implementation can and should be improved, but the functionalities are the desired ones.
    - Moreover, the method should be accessible to run on a given properties dictionary, so to predict the kinds before the StructureData instance generation.
    """
    
    # cannot do properties.symbols.value due to recursion problem if called in Kinds:
    # if I call properties, this will again reinitialize the properties attribute and so on.
    # should be this:
    # symbols = self.base.attributes.get("_property_attributes")['symbols']['value'] 
    # However, for now I do not let the kinds to be automatically generated when we initialise the structure:
    symbols = structure.get_site_property("symbol")
    default_thresholds = {
        "charge":0.1,
        "mass": 1e-4,
        "magnetization":1e-2,
    }
    
    
    list_tags = []
    if len(kind_tags) == 0: 
        kind_tags = [None]*len(symbols) # <== For now we support also for only ... see above doc string.
        check_kinds = False
        #kind=tags = self.properties.kinds.value
    else:
        list_tags = [kind_tags.index(n) for n in kind_tags]
        check_kinds = True
                        
    array_tags = np.array(list_tags)
    
    # Step 1:
    kind_properties = []
    kinds_dictionary = {'kinds':{}}
    for single_property in structure.get_property_names(domain="site"):
        if single_property not in ["symbol","position","kind_name"]+exclude:
            prop = structure.get_site_property(single_property)
            thr = custom_thr.get(single_property, default_thresholds.get(single_property))
            kinds_dictionary[single_property] = {}
            # for this if, refer to the description of the `to_kinds` method of the IntraSiteProperty class.
            
            kinds_per_property = to_kinds(prop, symbols, thr=thr)
            kind_properties.append(kinds_per_property[0])
            # I prefer to store again under the key 'value', may be useful in the future
            kinds_dictionary[single_property] = kinds_per_property[1].tolist()
            
    k = np.array(kind_properties)
    k = k.T
    
    # Step 2:
    kinds = np.zeros(len(structure.get_site_property("symbol")),dtype=int) -1
    check_array = np.zeros(len(structure.get_site_property("position")),dtype=int)
    kind_names = copy.deepcopy(symbols)
    kind_numeration = []
    for i in range(len(k)):
        # Goes from the first symbol... so the numbers will be from zero to N (Please note: the symbol does not matter: Li0, Cu1... not Li0, Cu0.)
        element = symbols[i]
        diff = k-k[i]
        diff_sum = np.sum(np.abs(diff),axis=1)

        kinds[np.where(diff_sum == 0)[0]] = i
        for where in np.where(diff_sum == 0)[0]:
            if f"{element}{i}" in kind_tags: # If I encounter the same tag as provided as input or generated here:
                kind_numeration.append(i+len(k))
            else:
                kind_numeration.append(i)
                
            kind_names[where] = f"{element}{kind_numeration[-1]}"
                
            check_array[where] = i
            
        if len(np.where(kinds == -1)[0]) == 0 : 
            #print(f"search ended at iteration {i}")
            break
    
    # Step 3:
    kinds_dictionary['kinds'] = [kind_names[i] if not kind_tags[i] else kind_tags[i] for i in range(len(kind_tags))]
    kinds_dictionary['index'] = kind_numeration
    kinds_dictionary['symbols'] = symbols
    
    # Step 4: check on the kind_tags consistency with the properties value.
    if check_kinds and not np.array_equal(check_array, array_tags):
        raise ValueError (f"The kinds you provided in the `kind_tags` input are not correct, as properties values are not consistent with them. Please check that this is what you want.")
    
    return kinds_dictionary