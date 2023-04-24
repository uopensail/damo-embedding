# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _damo
else:
    import _damo

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class Parameters(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _damo.Parameters_swiginit(self, _damo.new_Parameters(*args))
    __swig_destroy__ = _damo.delete_Parameters

    def insert(self, *args):
        return _damo.Parameters_insert(self, *args)

    def to_string(self):
        return _damo.Parameters_to_string(self)
    params_ = property(_damo.Parameters_params__get, _damo.Parameters_params__set)

# Register Parameters in _damo:
_damo.Parameters_swigregister(Parameters)

class PyInitializer(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _damo.PyInitializer_swiginit(self, _damo.new_PyInitializer(*args))

    def call(self, w):
        return _damo.PyInitializer_call(self, w)
    __swig_destroy__ = _damo.delete_PyInitializer

# Register PyInitializer in _damo:
_damo.PyInitializer_swigregister(PyInitializer)

class PyOptimizer(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _damo.PyOptimizer_swiginit(self, _damo.new_PyOptimizer(*args))

    def call(self, w, gds, global_step):
        return _damo.PyOptimizer_call(self, w, gds, global_step)
    __swig_destroy__ = _damo.delete_PyOptimizer

# Register PyOptimizer in _damo:
_damo.PyOptimizer_swigregister(PyOptimizer)

class PyFilter(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _damo.PyFilter_swiginit(self, _damo.new_PyFilter(*args))

    def check(self, key):
        return _damo.PyFilter_check(self, key)

    def add(self, key, num):
        return _damo.PyFilter_add(self, key, num)
    __swig_destroy__ = _damo.delete_PyFilter

# Register PyFilter in _damo:
_damo.PyFilter_swigregister(PyFilter)

class PyStorage(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _damo.PyStorage_swiginit(self, _damo.new_PyStorage(*args))
    __swig_destroy__ = _damo.delete_PyStorage

    def dump(self, path, expires):
        return _damo.PyStorage_dump(self, path, expires)

# Register PyStorage in _damo:
_damo.PyStorage_swigregister(PyStorage)

class PyEmbedding(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _damo.PyEmbedding_swiginit(self, _damo.new_PyEmbedding(*args))
    __swig_destroy__ = _damo.delete_PyEmbedding

    def lookup(self, keys, w):
        return _damo.PyEmbedding_lookup(self, keys, w)

    def apply_gradients(self, keys, gds):
        return _damo.PyEmbedding_apply_gradients(self, keys, gds)

# Register PyEmbedding in _damo:
_damo.PyEmbedding_swigregister(PyEmbedding)



