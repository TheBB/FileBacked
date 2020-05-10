from collections.abc import MutableMapping
from inspect import isclass
from itertools import chain
from typing import Any, Dict, Generic, TypeVar, Union, Tuple, List, Optional
from typing import get_type_hints

import dill
import numpy as np
from typing_inspect import get_generic_bases, get_origin, get_args


class FileBackedDescriptor:
    """Descriptor that allows attribute-like access to an object backed by
    file storage.  These objects should not be created manually,
    instead they are created the FileBacked metaclass when required.
    """

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        # We set lazy=True, because the only way the object is not
        # cached at this point is if the file was originally opened
        # lazily.
        return self.read(obj, True)

    def __set__(self, obj, value):
        _setkey(obj, self.name, value)

    def __delete__(self, obj):
        _delkey(obj, self.name, AttributeError)

    def write(self, group, obj, **kwargs):
        """Issue a call to write this object to a HDF5 group."""
        value = self.__get__(obj)
        tp = obj.__filebacked_annotations__[self.name]
        write(group, self.name, value, tp, **kwargs)

    def read(self, obj, lazy=False):
        """Issue a call to read this object from a HDF5 group."""
        tp = obj.__filebacked_annotations__[self.name]
        return _getkey(obj, self.name, tp, AttributeError, lazy)


def _getkey(obj, key, tp, exc, lazy, filekey=None):
    """Helper function to get the value associated with a key.

    :param obj: A FileBacked object to which the key belongs
    :param key: The key to access
    :param tp: Type of the object to read
    :param exc: Type of exception to raise if key is not found
    :param lazy: If true, dependent objects will not be read until
        accessed
    :param filekey: If given, a key to be used when accessing from the
        file. This is useful if `key` is not a string.
    """

    if filekey is None:
        filekey = key

    # Deleting a key does not remove it from the file, so we must
    # explicitly check if it has been deleted.
    if key in obj.__filebacked_deleted__:
        raise exc(key)

    # If it is cached, return the cached object.
    if key in obj.__filebacked_data__:
        return obj.__filebacked_data__[key]

    # Otherwise, get it from the file
    if obj.__filebacked_group__ and filekey in obj.__filebacked_group__:
        value = read(obj.__filebacked_group__[filekey], tp, lazy=lazy)
        obj.__filebacked_data__[key] = value
        return value

    # If the type is of option type, it's missing
    elif _is_option(tp):
        return None

    raise exc(key)


def _setkey(obj, key, value):
    """Helper function to set the value associated with a key.

    :param obj: A FileBacked object to which the key belongs
    :param key: The key to set
    :param value: The value associated with the key
    """

    # If a key has been previously deleted, we must clear that flag
    if key in obj.__filebacked_deleted__:
        obj.__filebacked_deleted__.remove(key)

    obj.__filebacked_data__[key] = value


def _delkey(obj, key, exc, filekey=None):
    """Helper function to delete the value associated with a key.

    :param obj: A FileBacked object to which the key belongs
    :param key: The key to set
    :param exc: Type of exception to raise if the key does not exist
    :param filekey: If given, a key to be used when accessing from the
        file. This is useful if `key` is not a string.
    """
    if filekey is None:
        filekey = key

    # If the key has previously been flagged as deleted, raise exception
    if key in obj.__filebacked_deleted__:
        raise exc(key)

    # Delete cached value
    if key in obj.__filebacked_data__:
        del obj.__filebacked_data__[key]

    # If there's no cached value, and no file-backed value, raise
    elif not obj.__filebacked_group__ or filekey not in obj.__filebacked_group__:
        raise exc(key)

    # Flag as deleted
    obj.__filebacked_deleted__.add(key)


class FileBackedBase:
    """Base class for file-backed objects.

    Objects of type FileBackedBase have three special attributes:

    - `__filebacked_group__`: HDF5 group for reading cached data
    - `__filebacked_data__`: Dictionary mapping keys to cached objects
    - `__filebacked_deleted__`: Set that tracks which keys have been
      deleted

    When constructed normally, the FileBackedBase constructor MUST be
    called before the file-backed keys are written to or read from.
    After `__init__`, the special `__pyinit__` method is called with
    no arguments.

    When constructed by reading from a file, the standard constructor
    is not executed.  Instead, the `__pyinit__` method is called with
    no arguments.  Subclasses can use this method to complete
    initialization.

    Subclasses should implement the _read(**kwargs) method for eagerly
    reading data from the file.
    """

    def __init__(self, group=None, call_pyinit=True):
        """Initialize the file-backed data structures.  Subclasses should call
        this method with no arguments.
        """
        self.__filebacked_group__ = group
        self.__filebacked_data__ = {}
        self.__filebacked_deleted__ = set()
        self.__filebacked_annotations__ = get_type_hints(self.__class__)

        if call_pyinit:
            self.__pyinit__()

    def __pyinit__(self):
        """Special initialization method that will run both after regularly
        constructing an object, and after reading from a file.  Called
        with no arguments.
        """
        pass

    def allow_lazy(self):
        """Return True if this object supports lazy reading."""
        return True

    def write(self, group):
        """Write the contents of this object to an HDF5 group.  Writing of keys
        and values must be implemented in a subclass.
        """
        group.attrs['module'] = self.__class__.__module__
        group.attrs['class'] = self.__class__.__name__

    @classmethod
    def read(cls, group, lazy=False):
        """Construct a new object by reading from an HDF5 group.

        :param cls: The type of the new object, or any superclass. The
            correct class of the object will be found in the
            inheritance graph of `cls`.  The module where the correct
            class is defined must be imported for this to work.
        :param group: The HDF5 group to read from.
        :param lazy: If true, objects will be read from the file when
            accessed, rather than eagerly.  Some objects do not support
            this form of access.  If `lazy` is set, the file must
            remain open for the entire lifetime of the object.  This
            is the responsibility of the caller.
        """

        # Search for the appropriate subclass in the inheritance graph
        modulename = group.attrs['module']
        classname = group.attrs['class']
        for subcls in _subclasses(cls, root=True):
            if subcls.__module__ == modulename and subcls.__name__ == classname:
                break
        else:
            raise TypeError(f"Unable to find appropriate class: {modulename}.{classname}")

        # Instantiate a new object of the given type and initialize
        # the file-backed data structures.  Don't call __pyinit__ yet.
        obj = subcls.__new__(subcls)
        FileBackedBase.__init__(obj, group, call_pyinit=False)

        # Read eagerly if requested or required.  Note that lazy
        # reading may propagate to child objects even if the object
        # itself does not support lazy reading.
        if not lazy or not obj.allow_lazy():
            obj._read(lazy=lazy)
            obj.__filebacked_group__ = None

        obj.__pyinit__()
        return obj

    def _read(self, **kwargs):
        raise NotImplementedError


class FileBackedMeta(type):
    """Metaclass for file-backed objects."""

    def __new__(cls, name, bases, attrs):
        # Collect all file-backed attributes from this class and convert them to descriptors
        ignore = attrs.get('__filebacked_ignore__', set())
        annotations = attrs.get('__annotations__', dict())

        file_attribs = set()
        for name in annotations:
            if name in ignore:
                continue
            file_attribs.add(name)
            attrs[name] = FileBackedDescriptor(name)

        # Track all file-backed attributes from the entire inheritance diagram
        for base in bases:
            file_attribs |= set(getattr(base, '__filebacked_attribs__', []))
        attrs['__filebacked_attribs__'] = sorted(file_attribs)

        return super().__new__(cls, name, bases, attrs)


class FileBacked(FileBackedBase, metaclass=FileBackedMeta):
    """Superclass for objects with file-backed attributes."""

    def write(self, group, only=None, skip=None, **kwargs):
        """Write the contents of this object to an HDF5 group.

        :param group: The HDF5 group to write to.
        :param only: If given, only write these attributes.
        :param skip: If given, skip writing these attributes.
        :param kwargs: Arguments to pass through to child objects.
        """

        super().write(group)
        attribs = _distill_attribs(self.__filebacked_attribs__, only, skip)
        for attr in attribs:
            getattr(self.__class__, attr).write(group, self, **kwargs)

    def _read(self, **kwargs):
        """Eagerly read attributes from the file."""
        for attr in self.__filebacked_attribs__:
            getattr(self.__class__, attr).read(self, **kwargs)


K = TypeVar('K')
V = TypeVar('V')
class FileBackedDict(FileBackedBase, MutableMapping, Generic[K,V]):
    """Superclass for dictionaries with file-backed storage."""

    def __init__(self, *args, **kwargs):
        super().__init__()

        # The superclass creates this, but can't accept the full
        # dictionary constructor interface, so we recreate it.
        self.__filebacked_data__ = dict(*args, **kwargs)

    def __resolve_typevars(self):
        """Compute the key and value types."""
        self.__K, self.__V = _my_typevars(self, FileBackedDict)

    def allow_lazy(self):
        # For non-string or integer keys, the dictionary is stored as
        # essentially a list of (key, value)-pairs in the file.  For
        # such storage, lazy reading would be inefficient.
        self.__resolve_typevars()
        return self.__K in (str, int)

    def __pyinit__(self):
        super().__pyinit__()
        self.__resolve_typevars()

    def __getitem__(self, k):
        if self.__K == str:
            return _getkey(self, k, self.__V, KeyError, True)

        # For integer keys, the keys in the file are stringified
        elif self.__K == int:
            return _getkey(self, k, self.__V, KeyError, True, filekey=str(k))

        # Otherwise, we disallow lazy reading, so just read the cache
        # directly.
        else:
            return self.__filebacked_data__[k]

    def __setitem__(self, k, v):
        _setkey(self, k, v)

    def __delitem__(self, k):
        if self.__K == int:
            _delkey(self, k, KeyError, filekey=str(k))
        else:
            _delkey(self, k, KeyError)

    def __iter__(self):
        # To maintain order between stored and newly added keys, we
        # iterate over file-backed storage, followed by cached
        # storage, keeping track of keys we have already seen.
        found = set()

        # Integer keys must be converted from strings in the file.
        if self.__K == int and self.__filebacked_group__:
            group_keys = map(int, self.__filebacked_group__)

        # Otherwise, handle the file keys directly.  If they are not
        # strings, the group should be set to None (eager reading).
        else:
            group_keys = self.__filebacked_group__ or []

        for key in chain(group_keys, self.__filebacked_data__):
            if key in found or key in self.__filebacked_deleted__:
                continue
            found.add(key)
            yield key

    def __len__(self):
        return len(self.__filebacked_data__)

    def write(self, group, **kwargs):
        """Write the contents of this object to an HDF5 group.

        :param group: The HDF5 group to write to.
        :param only: If given, only write these attributes.
        :param skip: If given, skip writing these attributes.
        :param kwargs: Arguments to pass through to child objects.
        """

        super().write(group)
        _write_dict(group, self, self.__K, self.__V, **kwargs)

    def _read(self, **kwargs):
        """Eagerly read elements from the file."""

        # This method may be called before __pyinit__
        self.__resolve_typevars()
        _read_dict(self.__filebacked_group__, self.__filebacked_data__, self.__K, self.__V, **kwargs)


class Filter:
    """A filter handles reading and writing of various supported types.
    For each object to read or write, filters are tried in order until
    one is found that supports the data type.

    To register new filters, use `register_filter`.
    """

    def applicable(self, tp):
        """Return true if this filter can handle objects of type `tp`."""
        raise NotImplementedError

    def write(self, group, name, obj, tp, **kwargs):
        """Write an object to an HDF5 group.

        :param group: The HDF5 group to write to
        :param name: The name of the subgroup or dataset to be created
        :param obj: The object to write
        :param tp: The type of the object (same argument as passed to
            `applicable`)
        :param kwargs: Additional parameters that should be passed on
        """
        raise NotImplementedError

    def read(self, group, tp, **kwargs):
        """Read an object from an HDF5 group or dataset.

        :param group: The HDF5 group or dataset to read from.
        :param tp: The type of the object (same argument as passed to
            `applicable`)
        :param kwargs: Additional parameters that should be passed on
        """
        raise NotImplementedError


class FileBackedFilter:
    """Filter for I/O of FileBackedBase subclasses."""

    def applicable(self, tp):
        return isclass(tp) and issubclass(tp, FileBackedBase)

    def write(self, group, name, obj, tp, **kwargs):
        obj.write(group.require_group(name), **kwargs)

    def read(self, group, tp, **kwargs):
        return tp.read(group, **kwargs)


class StringFilter:
    """Filter for I/O of strings."""

    def applicable(self, tp):
        return tp == str

    def write(self, group, name, obj, tp, **kwargs):
        group[name] = np.string_(obj.encode('utf-8'))

    def read(self, group, tp, **kwargs):
        return group[()].decode('utf-8')


class ScalarFilter:
    """Filter for I/O of scalar-like objects."""

    def applicable(self, tp):
        return _is_scalar(tp)

    def write(self, group, name, obj, tp, **kwargs):
        group[name] = obj

    def read(self, group, tp, **kwargs):
        return group[()]


class NumpyFilter:
    """Filter for I/O of numpy n-dimensional arrays."""

    def applicable(self, tp):
        return isclass(tp) and tp == np.ndarray

    def write(self, group, name, obj, tp, **kwargs):
        group[name] = obj

    def read(self, group, tp, **kwargs):
        return group[:]


class BuiltinSequenceFilter:
    """Filter for I/O of lists and tuples."""

    def applicable(self, tp):
        orig = get_origin(tp)
        if orig == list or orig == List:
            return True
        if orig != tuple and orig != Tuple:
            return False
        args = get_args(tp)
        return len(args) == 2 and args[1] == Ellipsis

    def write(self, group, name, obj, tp, **kwargs):
        eltype, *_ = get_args(tp)
        if _is_scalar(eltype):
            group[name] = obj
        else:
            subgrp = group.require_group(name)
            for i, element in enumerate(obj):
                write(subgrp, str(i), element, eltype, **kwargs)

    def read(self, group, tp, **kwargs):
        eltype, *_ = get_args(tp)
        constructor = get_origin(tp)
        if constructor == Tuple:
            constructor = tuple
        elif constructor == List:
            constructor = list
        if _is_scalar(eltype):
            return constructor(group[:])
        return constructor(read(group[str(i)], eltype, **kwargs) for i in range(len(group)))


class DictFilter:
    """Filter for dictionaries."""

    def applicable(self, tp):
        return get_origin(tp) == dict or get_origin(tp) == Dict

    def write(self, group, name, obj, tp, **kwargs):
        K, V = get_args(tp, evaluate=True)
        _write_dict(group.require_group(name), obj, K, V, **kwargs)

    def read(self, group, tp, **kwargs):
        K, V = get_args(tp, evaluate=True)
        retval = {}
        _read_dict(group, retval, K, V, **kwargs)
        return retval


class OptionFilter:
    """Filter for Option[T] types."""

    def applicable(self, tp):
        return _is_option(tp)

    def write(self, group, name, obj, tp, **kwargs):
        if obj is not None:
            write(group, name, obj, **kwargs)

    def read(self, group, tp, **kwargs):
        T, _ = get_args(tp)
        return read(group, T, **kwargs)


class PickleFilter:
    """Filter for I/O of arbitrary Python objects, stored as pickled
    strings.
    """

    def applicable(self, tp):
        return True

    def write(self, group, name, obj, tp, **kwargs):
        group[name] = np.string_(dill.dumps(obj))

    def read(self, group, tp, **kwargs):
        return dill.loads(group[()])


_FILTERS = [
    FileBackedFilter(),
    StringFilter(),
    ScalarFilter(),
    NumpyFilter(),
    BuiltinSequenceFilter(),
    OptionFilter(),
    DictFilter(),
]

pickle_filter = PickleFilter()

def register_filter(flt):
    """Add `flt` to the registered filter list, prioritized before all
    existing filters.
    """
    _FILTERS.insert(0, flt)


def write(group, name, obj, tp=None, **kwargs):
    """Helper function for writing an object to an HDF5 group.  Tries all
    registered filters in order and delegates to the first applicable one.
    Same arguments as `Filter.write`.

    The optional keyword argument `allow_picke` (default false)
    determines whether the special pickling filter is used as a
    fallback for objects which cannot otherwise be stored.
    """

    if tp is None:
        tp = type(obj)

    allow_pickle = kwargs.get('allow_pickle', False)
    for flt in _FILTERS:
        if flt.applicable(tp):
            flt.write(group, name, obj, tp, **kwargs)
            return
    if allow_pickle or tp == object:
        pickle_filter.write(group, name, obj, tp, **kwargs)
        return
    raise TypeError(f"Unable to write type '{tp}'")


def read(group, tp, **kwargs):
    """Helper function for writing an object to an HDF5 group.  Tries all
    registered filters in order and delegates to the first applicable one.
    Same arguments as `Filter.read`.

    The optional keyword argument `allow_picke` (default false)
    determines whether the special pickling filter is used as a
    fallback for objects which cannot otherwise be read.
    """

    allow_pickle = kwargs.get('allow_pickle', False)
    for flt in _FILTERS:
        if flt.applicable(tp):
            return flt.read(group, tp, **kwargs)
    if allow_pickle or tp == object:
        return pickle_filter.read(group, tp, **kwargs)
    raise TypeError(f"Unable to read type '{tp}'")


def _write_dict(group, obj, K, V, only=None, skip=None, **kwargs):
    """Helper function for writing a dictionary to an HDF5 group.

    :param group: The group to write to
    :param obj: The dictionary to write
    :param K: Type of keys
    :param V: Type of values
    :param kwargs: Additional parameters that should be passed on
    """

    attribs = _distill_attribs(list(obj), only, skip)

    # String keys are stored as-is
    if K == str:
        for attr in attribs:
            write(group, attr, obj[attr], V)

    # Integer keys are stored stringified
    elif K == int:
        for attr in attribs:
            write(group, str(attr), obj[attr], V)

    # Otherwise, we store as a list of (key, value) pairs
    else:
        for i, attr in enumerate(attribs):
            subgrp = group.require_group(str(i))
            write(subgrp, 'key', attr, K)
            write(subgrp, 'value', obj[attr], V)


def _read_dict(group, obj, K, V, **kwargs):
    """Helper function for reading a dictionary from an HDF5 group.

    :param group: The group to read from
    :param obj: Dictionary to mutate
    :param K: Type of keys
    :param V: Type of values
    :param kwargs: Additional parameters that should be passed on'
    """

    # String keys are stored as-is
    if K == str:
        for attr, subgrp in group.items():
            obj[attr] = read(subgrp, V, **kwargs)

    # Integer keys are stored stringified
    elif K == int:
        for attr, subgrp in group.items():
            obj[int(attr)] = read(subgrp, V, **kwargs)

    # Otherwise, we store as a list of (key, value) pairs
    else:
        for attr, subgrp in group.items():
            k = read(subgrp['key'], K, **kwargs)
            v = read(subgrp['value'], V, **kwargs)
            obj[k] = v


def _my_typevars(obj, cls):
    """Compute the type arguments of `obj` (a class) relative to `cls` (a
    superclass).

    Example:

        from typing import Generic, TypeVar, List
        A, B, C = TypeVar('A'), TypeVar('B'), TypeVar('C')
        class Super(Generic[A, B, C]):
            pass
        class Sub(Super[int, str, float], List[int]):
            pass

        _my_typevars(Sub, Super) => (int, str, float)
        _my_typevars(Sub, list) => (int,)
    """

    while not isclass(obj):
        obj = type(obj)
    for base in get_generic_bases(obj):
        if get_origin(base) == cls:
            return get_args(base, evaluate=True)
    raise TypeError(f"Couldn't find type variables for {obj} relative to {cls}")


def _subclasses(cls, root=False):
    """Iterate over all subclasses of `cls`.  If `root` is true, `cls`
    itself is included.
    """
    if root:
        yield cls
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub, root=False)


def _distill_attribs(attribs, only, skip):
    """Return a list of all elements of `attribs` keeping either only the
    ones in `only` or skipping the ones in `skip`.
    """
    assert only is None or skip is None
    if only is not None:
        only = set(only)
        return [a for a in attribs if a in only]
    if skip is not None:
        skip = set(skip)
        return [a for a in attribs if a not in skip]
    return attribs


_SCALARS = {
    float, np.float, np.float64, np.float32, np.float16,
    int, np.int, np.int64, np.int32, np.int16, np.int8, np.int0,
}

if hasattr(np, 'float128'):
    _SCALARS.add(np.float128)

def _is_scalar(tp):
    """Check if a type is considered a scalar."""
    return tp in _SCALARS

def _is_option(tp):
    """Return true if tp is an option type."""
    orig = get_origin(tp)
    if orig != Union:
        return False
    args = get_args(tp)
    return args[1] == type(None)
