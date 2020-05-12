from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import List, Tuple, Any, Optional
from pathlib import Path
import shutil

import numpy as np
import scipy.sparse as sparse
import h5py
import pytest

from filebacked import *


class BasicAttribs(FileBacked):
    _int: int
    _str: str
    _float: float
    _opt1: Optional[str]
    _opt2: Optional[str]
    inttuple: Tuple[int, ...]
    strtuple: Tuple[str, ...]
    array: np.ndarray
    floatlist: List[float]
    strlist: List[str]
    strdict: Dict[str, int]
    intdict: Dict[int, int]
    tupledict: Dict[Tuple[str, ...], int]
    union1: Union[int, str, float]
    union2: Union[int, str, float]
    union3: Union[int, str, float]
    csc: sparse.spmatrix
    csr: sparse.spmatrix
    coo: sparse.spmatrix

class StrDict(FileBackedDict[str, str]):
    pass

class IntDict(FileBackedDict[int, str]):
    pass

class TupleDict(FileBackedDict[Tuple[int, ...], str]):
    pass


@pytest.fixture(params=[True, False])
def lazy(request):
    return request.param


@contextmanager
def roundtrip(obj, wkwargs, rkwargs):
    with TemporaryDirectory() as path:
        path = Path(path)
        with h5py.File(path / 'out.hdf5', 'w') as f:
            obj.write(f, **wkwargs)
            ret = obj.__class__.read(f, **rkwargs)
            yield (obj, ret, f)


def test_basic_attribs(lazy):
    test = BasicAttribs()
    test._int = 1
    test._str = 'FileBacked'
    test._float = 3.14
    test._opt1 = 'has'
    test._opt2 = None
    test.inttuple = (1, 2, 3)
    test.strtuple = ('alpha', 'bravo', 'charlie')
    test.array = np.eye(3, 3, dtype=float)
    test.floatlist = list(test.array.flat)
    test.strlist = ['alpha', 'beta', 'gamma']
    test.strdict = {'alpha': 1, 'beta': 2, 'gamma': 3}
    test.intdict = {1: 2, 2: 3, 3: 1}
    test.tupledict = {('one','two'): 3, ('four', 'five'): 6}
    test.union1 = 1
    test.union2 = 'alpha'
    test.union3 = 4.5
    test.csc = sparse.eye(3, format='csc')
    test.csr = sparse.eye(3, format='csr')
    test.coo = sparse.eye(3, format='coo')

    with roundtrip(test, {}, {'lazy': lazy}) as (a, b, f):
        assert a._int == b._int == f['_int'][()] == 1
        assert a._str == b._str == f['_str'][()].decode('utf-8') == 'FileBacked'
        assert a._float == b._float == f['_float'][()] == 3.14
        assert a._opt1 == 'has'
        assert a._opt2 is None
        assert a.inttuple == b.inttuple == tuple(f['inttuple'][:]) == (1, 2, 3)
        assert a.strtuple == b.strtuple == ('alpha', 'bravo', 'charlie')
        assert f['strtuple']['0'][()].decode('utf-8') == 'alpha'
        assert f['strtuple']['1'][()].decode('utf-8') == 'bravo'
        assert f['strtuple']['2'][()].decode('utf-8') == 'charlie'
        assert (a.array == b.array).all()
        assert (b.array == np.eye(3, 3, dtype=float)).all()
        assert (a.array == f['array'][:]).all()
        assert a.floatlist == b.floatlist == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        assert (a.floatlist == f['floatlist'][:]).all()
        assert a.strlist == b.strlist == ['alpha', 'beta', 'gamma']
        assert f['strlist']['0'][()].decode('utf-8') == 'alpha'
        assert f['strlist']['1'][()].decode('utf-8') == 'beta'
        assert f['strlist']['2'][()].decode('utf-8') == 'gamma'
        assert a.strdict == b.strdict == {'alpha': 1, 'beta': 2, 'gamma': 3}
        assert f['strdict']['alpha'][()] == 1
        assert f['strdict']['beta'][()] == 2
        assert f['strdict']['gamma'][()] == 3
        assert list(a.strdict) == list(b.strdict) == ['alpha', 'beta', 'gamma']
        assert a.intdict == b.intdict == {1: 2, 2: 3, 3: 1}
        assert f['intdict']['1'][()] == 2
        assert f['intdict']['2'][()] == 3
        assert f['intdict']['3'][()] == 1
        assert list(a.intdict) == list(b.intdict) == [1, 2, 3]
        assert a.tupledict == b.tupledict == {('one', 'two'): 3, ('four', 'five'): 6}
        assert f['tupledict']['0']['key']['0'][()].decode('utf-8') == 'one'
        assert f['tupledict']['0']['key']['1'][()].decode('utf-8') == 'two'
        assert f['tupledict']['0']['value'][()] == 3
        assert f['tupledict']['1']['key']['0'][()].decode('utf-8') == 'four'
        assert f['tupledict']['1']['key']['1'][()].decode('utf-8') == 'five'
        assert f['tupledict']['1']['value'][()] == 6
        assert a.union1 == 1
        assert a.union2 == 'alpha'
        assert a.union3 == 4.5
        assert a.csc.getformat() == 'csc'
        assert (a.csc != sparse.eye(3)).nnz == 0
        assert a.csr.getformat() == 'csr'
        assert (a.csr != sparse.eye(3)).nnz == 0
        assert a.coo.getformat() == 'coo'
        assert (a.coo != sparse.eye(3)).nnz == 0


def test_strdict(lazy):
    test = StrDict()
    test['a'] = 'alpha'
    test['b'] = 'bravo'
    test['c'] = 'charlie'

    with roundtrip(test, {}, {'lazy': lazy}) as (a, b, f):
        assert a['a'] == b['a'] == 'alpha'
        assert a['b'] == b['b'] == 'bravo'
        assert a['c'] == b['c'] == 'charlie'
        assert f['a'][()].decode('utf-8') == 'alpha'
        assert f['b'][()].decode('utf-8') == 'bravo'
        assert f['c'][()].decode('utf-8') == 'charlie'
        assert len(a) == len(b) == 3
        assert list(a) == ['a', 'b', 'c']
        assert list(b) == ['a', 'b', 'c']


def test_intdict(lazy):
    test = IntDict()
    test[10] = 'alpha'
    test[100] = 'bravo'
    test[512] = 'charlie'

    with roundtrip(test, {}, {'lazy': lazy}) as (a, b, f):
        assert a[10] == b[10] == 'alpha'
        assert a[100] == b[100] == 'bravo'
        assert a[512] == b[512] == 'charlie'
        assert f['10'][()].decode('utf-8') == 'alpha'
        assert f['100'][()].decode('utf-8') == 'bravo'
        assert f['512'][()].decode('utf-8') == 'charlie'
        assert len(a) == len(b) == 3
        assert list(a) == [10, 100, 512]
        assert list(b) == [10, 100, 512]


def test_tupledict(lazy):
    test = TupleDict()
    test[(1,2)] = 'alpha'
    test[(3,4,5)] = 'bravo'
    test[(6,)] = 'charlie'
    test[()] = 'delta'
