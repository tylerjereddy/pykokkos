import math

import pykokkos as pk
from pykokkos.lib.ufuncs import _ufunc_kernel_dispatcher


def arange(start,
           /,
           stop=None,
           step=1,
           dtype=None,
           device=None):
    """
    Returns evenly spaced values within the half-open interval ``[start, stop)`` as a one-dimensional view.

    Parameters
    ----------
    start (Union[int, float]) – if stop is specified, the start of interval (inclusive);
    otherwise, the end of the interval (exclusive). If stop is not specified, the default
    starting value is 0.

    stop (Optional[Union[int, float]]) – the end of the interval. Default: None.

    step (Union[int, float]) – the distance between two adjacent elements (out[i+1] - out[i]).
    Must not be 0; may be negative, this results in an empty view if stop >= start. Default: 1.

    dtype (Optional[dtype]) – output view data type. If dtype is None, the output view data
    type must be inferred from start, stop and step. If those are all integers, the output
    view dtype must be the default integer dtype; if one or more have type float, then the
    output view dtype must be the default real-valued floating-point data type. Default: None.

    device (Optional[device]) – device on which to place the created view. Default: None.

    Returns
    -------
    out - a one-dimensional view containing evenly spaced values. The length of the output view
    must be ``ceil((stop-start)/step)`` if ``stop - start`` and ``step`` have the same sign, and
    length 0 otherwise.
    """
    print("arange received dtype:", dtype)
    print("arange received start, stop, step:", start, stop, step)
    if dtype is None:
        if isinstance(start, int) and isinstance(stop, int) and isinstance(step, int):
            dtype = pk.int64
        else:
            dtype = pk.float64
    if stop is not None:
        val_1 = stop - start
    else:
        val_1 = start
        stop = start
        start = 0
    if (val_1 > 0 and step > 0) or (val_1 < 0 and step < 0):
        size = math.ceil((val_1) / step)
        print("estimated size:", size)
        out = pk.View([size], dtype=dtype)
        if size == 1:
            out[:] = start
            return out
        print("before _ufunc_kernel_dispatcher with dtype", dtype)
        print("type(start), type(stop), type(step):", type(start), type(stop), type(step))
        _ufunc_kernel_dispatcher(tid=size,
                                 dtype=dtype,
                                 ndims=1,
                                 op="arange",
                                 sub_dispatcher=pk.parallel_for,
                                 out=out,
                                 start=start,
                                 stop=stop,
                                 step=step)
        return out
    else:
        out = pk.View([0], dtype=dtype)
        return out





def zeros(shape, *, dtype=None, device=None):
    if dtype is None:
        dtype = pk.double

    if isinstance(shape, int):
        return pk.View([shape], dtype=dtype)
    else:
        return pk.View([*shape], dtype=dtype)


def ones(shape, *, dtype=None, device=None):
    if dtype is None:
        # NumPy also defaults to a double for ones()
        dtype = pk.float64
    view: pk.View = pk.View([*shape], dtype=dtype)
    view[:] = 1
    return view


def ones_like(x, /, *, dtype=None, device=None):
    if dtype is None:
        dtype = x.dtype
    view: pk.View = pk.View([*x.shape], dtype=dtype)
    view[:] = 1
    return view


def full(shape, fill_value, *, dtype=None, device=None):
    if dtype is None:
        dtype = fill_value.dtype
    try:
        view: pk.View = pk.View([*shape], dtype=dtype)
    except TypeError:
        view: pk.View = pk.View([shape], dtype=dtype)
    view[:] = fill_value
    return view
