==========
FileBacked
==========

.. image:: https://badge.fury.io/py/FileBacked.svg
   :target: https://badge.fury.io/py/FileBacked

.. image:: https://travis-ci.org/TheBB/FileBacked.svg?branch=master
   :target: https://travis-ci.org/TheBB/FileBacked


The FileBacked library allows you to easily define complex Python
types which can be saved to disk in a format that is efficient,
inspectable and interfaceable outside of Python.

While pickling is generally quite reliable for storing Python objects
on disk, it cannot truly function as an interface format for other
languages, and it is also not secure and stable enough to be used for
anything other than storing and reading your own files.

FileBacked works by storing objects in HDF5 format. This is ideal for
numpy arrays, but also works well for most of the other standard
Python types.


How it works
------------

Define a class with attributes that are backed by disk storage.

.. code-block:: python

   from filebacked import FileBacked

   class MyClass(FileBacked):
       myint: int

   myobj = MyClass()
   myobj.myint = 1


The type and name of the attribute will influence the format of the
resulting HDF5 file.  Let us save the object.

.. code-block:: python

   import h5py

   with h5py.File('myfile.hdf5', 'w') as f:
       myobj.write(f)


The resulting file should have a root dataset named 'myint', a scalar
with value 1. And now let us read it again.

.. code-block:: python

   with h5py.File('myfile.hdf5', 'r') as f:
       newobj = MyClass.read(f)
   assert newobj.myint == 1


Supported types
---------------

The following types are supported:

- Scalar numbers (*int*, *float* and numpy scalar types)
- Strings (*str*)
- Numpy arrays (``numpy.ndarray``)
- Scipy sparse matrices (``scipy.sparse.spmatrix``) of CSR, CSC and
  COO type (although you are free to give the type hint as the general
  *spmatrix* superclass)
- Homogeneous tuples (``Tuple[eltype, ...]``) and lists (``List[eltype]``)
  where the element type is supported
- Dictionaries (``Dict[keytype, valuetype]``) where the key and value
  types are supported
- Option types (``Option[valuetype]``) where the value type is supported
- Union types with arguments that support type checking with
  ``isinstance(..., arg)``
- Subclasses of ``FileBacked`` and ``FileBackedDict[keytype, valuetype]``

Arbitrary Python objects are stored as pickled strings if the
*allow_pickle* keyword argument is passed to the *write* and *read*
methods, respectively, or if the type is ``object``.

Types can be specified using standard builtins or type hint objects
from the *typing* module, as above.

To add support for a custom type, create a new *Filter* subclass:

.. code-block:: python

   from filebacked import Filter, register_filter

   class MyFilter(Filter):

       def applicable(self, tp):
           # Return true if the filter can be used for objects of the
           # given type.

       def write(self, group, name, obj, tp, **kwargs):
           # Write the object to the given group as a subgroup or
           # dataset with the given name.

       def read(self, group, tp, **kwargs):
           # Read the object from the given group or dataset and
           # return it.

   register_filter(MyFilter())


Newly registered filters will take priority over existing filters.


Interface
---------

For writing subclasses of *FileBacked* or *FileBackedDict*, it is most
useful to use the following pattern.  In this case, you cannot write
more than one object to a file, or you risk overlapping attributes.

.. code-block:: python

   with h5py.File('myfile.hdf5', 'w') as f:
       myobj.write(f)


Alternatively, use the *write* function for arbitrary objects of
supported type.  In this case you must specify a name and optionally
a type for the object.  It is recommended to always specify the type,
because element types of generic objects cannot be deduced from the
object alone.

.. code-block:: python

   with h5py.File('myfile.hdf5', 'w') as f:
       filebacked.write(f, 'somename', 3, int)


The *write* function will detect subclasses of *FileBacked* or
*FileBackedDict* and delegate writing accordingly, and the *write*
method of those two classes will delegate writing of attributes to the
*write* function.

All the write functions take an arbitrary amount of keyword arguments
that are passed throughout the object reference tree.  You can use
this to customize writing behaviour.  For example, the
``FileBacked.write`` and ``FileBackedDict.write`` methods accept the
keyword arguments *only* and *skip*, to avoid writing some attributes
if necessary:

.. code-block:: python

   class MyClass(FileBacked):
       small: int
       large: np.ndarray

       def write(self, group, sparse=False, **kwargs):
           if sparse:
               super().write(group, skip=('small',), **kwargs)
           else:
               super().write(group, **kwargs)


Ignoring attributes
^^^^^^^^^^^^^^^^^^^

By default, subclasses of ``FileBacked`` will handle any attributes
with type annotations.  If you want some to be ignored, list them in
the special ``__filebacked_ignore__`` attribute:

.. code-block:: python

   class MyClass(FileBacked):

       __filebacked_ignore__ = ('will_not_be_saved',)

       will_be_saved: int
       will_not_be_saved: str


Lazy reading
^^^^^^^^^^^^

Read functions accept an optional *lazy* parameter that can activate
lazy reading.  In this case, when possible, objects will only be read
from disk when accessed.  This is possible for attributes of
*FileBacked* objects, and for *FileBackedDict* objects whose keys are
integers or strings.  All builtin Python types are read eagerly.  Note
that when using lazy reading, it is imperative that the file object is
kept open for as long necessary to allow objects to be read on
demand.  When using eager reading, the file object may be closed
immediately after the *read* call.


File objects
^^^^^^^^^^^^

The standard Python package for HDF5 is h5py.  However, FileBacked
does not itself require h5py or depend on it.  Any HDF5 package with a
compatible interface, such as pyfive, should work.


Initialization
^^^^^^^^^^^^^^

When subclassing *FileBacked* and *FileBackedDict*, it is necessary to
call the superclass constructor before accessing any of the attributes
or keys that are managed by files (in the case of *FileBackedDict*,
that means any keys at all).

Upon reading an object from a file, the constructor will not be
called as it otherwise would.  Instead, the ``__pyinit__`` method will
be called, with no arguments, both when constructing an object
normally *and* when reading it from the file.  You can use this method
to perform extra object initialization if required, such as assigning
attributes which are not file-backed.


Caution
-------

Unlike pickle, FileBacked will not maintain reference equality between
objects.  If the same (mutable) object is referenced more than once in
the reference graph, it will instantiate as two different mutable
objects upon reading.  For the same reason, circular references will
cause problems.

FileBacked uses type hints to determine the structure of the resulting
HDF5 file.  It does not prevent you from assigning objects with
incorrect types.
