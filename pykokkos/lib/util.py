import pykokkos as pk
from pykokkos.lib.ufuncs import _ufunc_kernel_dispatcher

import numpy as np

# TODO: add proper implementations rather
# than wrapping NumPy
# These are required for the array API:
# https://data-apis.org/array-api/2021.12/API_specification/utility_functions.html

def all(x, /, *, axis=None, keepdims=False):
    if x == True:
        return True
    elif x == False:
        return False
    np_result = np.all(x)
    ret_val = pk.from_numpy(np_result)
    return ret_val


def any(x, /, *, axis=None, keepdims=False):
    return pk.View(pk.from_numpy(np.any(x)))


@pk.workunit
def sum_impl_1d_double(tid: int, acc: pk.Acc[pk.double], viewA: pk.View1D[pk.double]):
    acc += viewA[tid]


@pk.workunit
def sum_axis0_impl_1d_double(tid: int, viewA: pk.View2D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = 0
    for i in range(viewA.extent(0)):
        out[tid] += viewA[i][tid]


@pk.workunit
def sum_axis1_impl_1d_double(tid: int, viewA: pk.View2D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = 0
    for i in range(viewA.extent(1)):
        out[tid] += viewA[tid][i]


def sum(viewA, axis=None):
    if axis is not None:
        if axis == 0:
            out = pk.View([viewA.shape[1]], pk.double)
            pk.parallel_for(viewA.shape[1], sum_axis0_impl_1d_double, viewA=viewA, out=out)
            return out
        else:
            out = pk.View([viewA.shape[0]], pk.double)
            pk.parallel_for(viewA.shape[0], sum_axis1_impl_1d_double, viewA=viewA, out=out)

            return out


    if str(viewA.dtype) == "DataType.double":
        return pk.parallel_reduce(
            viewA.shape[0],
            sum_impl_1d_double,
            viewA=viewA)


def find_max(viewA):
    return max(viewA)


def searchsorted(view, ele):
    return np.searchsorted(view, ele)


@pk.workunit
def col_impl_2d_double(tid: int, view: pk.View2D[pk.double], col: pk.View1D[pk.int32], out: pk.View1D[pk.double]):
    out[tid] = view[tid][col[0]]


def col(view, col):
    if view.rank() != 2:
        raise RuntimeError("Only 2d views are supported for col")
    
    view_temp = pk.View([1], pk.int32)
    view_temp[0] = col
    col = view_temp
    
    if str(view.dtype) == "DataType.double":
        out = pk.View([view.shape[0]], pk.double)
        pk.parallel_for(
            view.shape[0],
            col_impl_2d_double,
            view=view,
            col=col,
            out=out)
    else:
        raise RuntimeError("col support views with type double only")

    return out

@pk.workunit
def linspace_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = ((view[1] - view[0])/(view[2] - 1))*tid + view[0]


def linspace(start, stop, num=50):
    inp = pk.View([3], pk.double)
    inp[:] = [start, stop, num]

    out = pk.View([num], pk.double)
    pk.parallel_for(num, linspace_impl_1d_double, view=inp, out=out)
    return out


def logspace(start, stop, num=50, base=10):
    y = linspace(start, stop, num)

    return power(base, y)


def flip(x, /, *, axis=None):
    """
    Reverses the order of elements in an array along the given axis. The shape of the array must be preserved.

    Parameters
    ----------
    x (View) – input view.

    axis (Optional[Union[int, Tuple[int, ...]]]) – axis (or axes) along which to flip. If axis is None,
    the function must flip all input view axes. If axis is negative, the function must count from the
    last dimension. If provided more than one axis, the function must flip only the specified axes. Default: None.


    Returns
    -------
    out (View) – an output view having the same data type and shape as ``x`` and whose elements, relative to ``x``, are reordered.
    """
    dtype = x.dtype
    ndims = len(x.shape)
    out = pk.View([*x.shape], dtype=dtype)
    if x.shape == ():
        tid = 1
    else:
        tid = x.shape[0]
    # TODO: flip() isn't really a ufunc, so
    # not sure we should be reusing this machinery;
    # perhaps refactor to use a separte dispatch mechanism
    # later?
    _ufunc_kernel_dispatcher(tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="flip",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=x)
    return out
