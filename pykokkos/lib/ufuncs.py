import pykokkos as pk

import numpy as np


@pk.workunit
def reciprocal_impl_1d_double(tid: int, view: pk.View1D[pk.double]):
    view[tid] = 1 / view[tid] # type: ignore


@pk.workunit
def reciprocal_impl_1d_float(tid: int, view: pk.View1D[pk.float]):
    view[tid] = 1 / view[tid] # type: ignore


@pk.workunit
def reciprocal_impl_2d_double(tid: int, view: pk.View2D[pk.double]):
    for i in range(view.extent(1)): # type: ignore
        view[tid][i] = 1 / view[tid][i] # type: ignore


@pk.workunit
def reciprocal_impl_2d_float(tid: int, view: pk.View2D[pk.float]):
    for i in range(view.extent(1)): # type: ignore
        view[tid][i] = 1 / view[tid][i] # type: ignore


def reciprocal(view):
    """
    Return the reciprocal of the argument, element-wise.

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    y : pykokkos view
        Output view.

    Notes
    -----
    .. note::
        This function is not designed to work with integers.

    """
    # see gh-29 for some discussion of the dispatching
    # awkwardness used here
    if str(view.dtype) == "DataType.double" and len(view.shape) == 1:
        pk.parallel_for(view.shape[0], reciprocal_impl_1d_double, view=view)
    elif str(view.dtype) == "DataType.float" and len(view.shape) == 1:
        pk.parallel_for(view.shape[0], reciprocal_impl_1d_float, view=view)
    elif str(view.dtype) == "DataType.float" and len(view.shape) == 2:
        pk.parallel_for(view.shape[0], reciprocal_impl_2d_float, view=view)
    elif str(view.dtype) == "DataType.double" and len(view.shape) == 2:
        pk.parallel_for(view.shape[0], reciprocal_impl_2d_double, view=view)
    # NOTE: pretty awkward to both return the view
    # and operate on it in place; the former is closer
    # to NumPy semantics
    return view


@pk.workunit
def log_impl_1d_double(tid: int, view: pk.View1D[pk.double]):
    view[tid] = log(view[tid]) # type: ignore


@pk.workunit
def log_impl_1d_float(tid: int, view: pk.View1D[pk.float]):
    view[tid] = log(view[tid]) # type: ignore


def log(view):
    """
    Natural logarithm, element-wise.

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    y : pykokkos view
        Output view.

    """
    if str(view.dtype) == "DataType.double":
        pk.parallel_for(view.shape[0], log_impl_1d_double, view=view)
    elif str(view.dtype) == "DataType.float":
        pk.parallel_for(view.shape[0], log_impl_1d_float, view=view)
    return view


@pk.workunit
def sqrt_impl_1d_double(tid: int, view: pk.View1D[pk.double]):
    view[tid] = sqrt(view[tid]) # type: ignore


@pk.workunit
def sqrt_impl_1d_float(tid: int, view: pk.View1D[pk.float]):
    view[tid] = sqrt(view[tid]) # type: ignore


def sqrt(view):
    """
    Return the non-negative square root of the argument, element-wise.

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    y : pykokkos view
        Output view.

    Notes
    -----
    .. note::
        This function should exhibit the same branch cut behavior
        as the equivalent NumPy ufunc.
    """
    # TODO: support complex types when they
    # are available in pykokkos?
    if str(view.dtype) == "DataType.double":
        pk.parallel_for(view.shape[0], sqrt_impl_1d_double, view=view)
    elif str(view.dtype) == "DataType.float":
        pk.parallel_for(view.shape[0], sqrt_impl_1d_float, view=view)
    return view


@pk.workunit
def log2_impl_1d_double(tid: int, view: pk.View1D[pk.double]):
    view[tid] = log2(view[tid]) # type: ignore


@pk.workunit
def log2_impl_1d_float(tid: int, view: pk.View1D[pk.float]):
    view[tid] = log2(view[tid]) # type: ignore


def log2(view):
    """
    Base-2 logarithm, element-wise.

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    y : pykokkos view
        Output view.

    """
    if str(view.dtype) == "DataType.double":
        pk.parallel_for(view.shape[0], log2_impl_1d_double, view=view)
    elif str(view.dtype) == "DataType.float":
        pk.parallel_for(view.shape[0], log2_impl_1d_float, view=view)
    return view


@pk.workunit
def log10_impl_1d_double(tid: int, view: pk.View1D[pk.double]):
    view[tid] = log10(view[tid]) # type: ignore


@pk.workunit
def log10_impl_1d_float(tid: int, view: pk.View1D[pk.float]):
    view[tid] = log10(view[tid]) # type: ignore


def log10(view):
    """
    Base-10 logarithm, element-wise.

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    y : pykokkos view
        Output view.

    """
    if str(view.dtype) == "DataType.double":
        pk.parallel_for(view.shape[0], log10_impl_1d_double, view=view)
    elif str(view.dtype) == "DataType.float":
        pk.parallel_for(view.shape[0], log10_impl_1d_float, view=view)
    return view


@pk.workunit
def log1p_impl_1d_double(tid: int, view: pk.View1D[pk.double]):
    view[tid] = log1p(view[tid]) # type: ignore


@pk.workunit
def log1p_impl_1d_float(tid: int, view: pk.View1D[pk.float]):
    view[tid] = log1p(view[tid]) # type: ignore


def log1p(view):
    """
    Return the natural logarithm of one plus the input array, element-wise.

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    y : pykokkos view
        Output view.

    """
    if str(view.dtype) == "DataType.double":
        pk.parallel_for(view.shape[0], log1p_impl_1d_double, view=view)
    elif str(view.dtype) == "DataType.float":
        pk.parallel_for(view.shape[0], log1p_impl_1d_float, view=view)
    return view


@pk.workunit
def sign_impl_1d_double(tid: int, view: pk.View1D[pk.double]):
    if view[tid] > 0:
        view[tid] = 1
    elif view[tid] == 0:
        view[tid] = 0
    elif view[tid] < 0:
        view[tid] = -1
    else:
        view[tid] = nan("")


@pk.workunit
def sign_impl_1d_float(tid: int, view: pk.View1D[pk.float]):
    if view[tid] > 0:
        view[tid] = 1
    elif view[tid] == 0:
        view[tid] = 0
    elif view[tid] < 0:
        view[tid] = -1
    else:
        view[tid] = nan("")


def sign(view):
    if str(view.dtype) == "DataType.double":
        pk.parallel_for(view.shape[0], sign_impl_1d_double, view=view)
    elif str(view.dtype) == "DataType.float":
        pk.parallel_for(view.shape[0], sign_impl_1d_float, view=view)
    return view


@pk.workunit
def add_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double], ):
    out[tid] = viewA[tid] + viewB[tid]


@pk.workunit
def add_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = viewA[tid] + viewB[tid]


def add(viewA, viewB):
    """
    Sums positionally corresponding elements
    of viewA with elements of viewB

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """

    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            add_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
            viewA.shape[0],
            add_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


@pk.workunit
def multiply_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = viewA[tid] * viewB[tid]


@pk.workunit
def multiply_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = viewA[tid] * viewB[tid]


def multiply(viewA, viewB):
    """
    Multiplies positionally corresponding elements
    of viewA with elements of viewB

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """

    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            multiply_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
            viewA.shape[0],
            multiply_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


@pk.workunit
def subtract_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = viewA[tid] - viewB[tid]


@pk.workunit
def subtract_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = viewA[tid] - viewB[tid]


def subtract(viewA, viewB):
    """
    Subtracts positionally corresponding elements
    of viewA with elements of viewB

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            subtract_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
            viewA.shape[0],
            subtract_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out

@pk.workunit
def matmul_impl_1d_double(tid: int, acc: pk.Acc[pk.double], viewA: pk.View1D[pk.double], viewB: pk.View2D[pk.double]):
    acc += viewA[tid] * viewB[0][tid]

@pk.workunit
def matmul_impl_1d_float(tid: int, acc: pk.Acc[pk.float], viewA: pk.View1D[pk.float], viewB: pk.View2D[pk.float]):
    acc += viewA[tid] * viewB[0][tid]

def matmul(viewA, viewB):
    """
    1D Matrix Multiplication of compatible views

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    Float/Double
        1D Matmul result

    """
    if len(viewA.shape) != 1 or viewA.shape[0] != viewB.shape[0]:
        raise RuntimeError(
            "Input operand 1 has a mismatch in its core dimension (Size {} is different from {})".format(viewA.shape[0], viewB.shape[0]))

    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        return pk.parallel_reduce(
            viewA.shape[0],
            matmul_impl_1d_double,
            viewA=viewA,
            viewB=viewB)
    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
        return pk.parallel_reduce(
            viewA.shape[0],
            matmul_impl_1d_float,
            viewA=viewA,
            viewB=viewB)
    else:
        raise RuntimeError("Incompatible Types")

@pk.workunit
def divide_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = viewA[tid] / viewB[tid]

@pk.workunit
def divide_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = viewA[tid] / viewB[tid]


def divide(viewA, viewB):
    """
    Divides positionally corresponding elements
    of viewA with elements of viewB

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            divide_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
            viewA.shape[0],
            divide_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out

@pk.workunit
def negative_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = view[tid] * -1

@pk.workunit
def negative_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = view[tid] * -1

def negative(view):
    """
    Element-wise negative of the view

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if str(view.dtype) == "DataType.double":
        out = pk.View([view.shape[0]], pk.double)
        pk.parallel_for(view.shape[0], negative_impl_1d_double, view=view, out=out)
    elif str(view.dtype) == "DataType.float":
        out = pk.View([view.shape[0]], pk.float)
        pk.parallel_for(view.shape[0], negative_impl_1d_float, view=view, out=out)
    return out

@pk.workunit
def positive_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = view[tid]

@pk.workunit
def positive_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = view[tid]

def positive(view):
    """
    Element-wise positive of the view;
    Essentially returns a copy of the view

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if str(view.dtype) == "DataType.double":
        out = pk.View([view.shape[0]], pk.double)
        pk.parallel_for(view.shape[0], positive_impl_1d_double, view=view, out=out)
    elif str(view.dtype) == "DataType.float":
        out = pk.View([view.shape[0]], pk.float)
        pk.parallel_for(view.shape[0], positive_impl_1d_float, view=view, out=out)
    return out


@pk.workunit
def power_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = pow(viewA[tid], viewB[tid])

@pk.workunit
def power_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = pow(viewA[tid], viewB[tid])


def power(viewA, viewB):
    """
    Returns a view with each val in viewA raised
    to the positionally corresponding power in viewB

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            power_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
            viewA.shape[0],
            power_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


@pk.workunit
def fmod_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = fmod(viewA[tid], viewB[tid])

@pk.workunit
def fmod_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = fmod(viewA[tid], viewB[tid])


def fmod(viewA, viewB):
    """
    Element-wise remainder of division when element of viewA is
    divided by positionally corresponding element of viewB

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """

    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            fmod_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
            viewA.shape[0],
            fmod_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


@pk.workunit
def square_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = view[tid] * view[tid]

@pk.workunit
def square_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = view[tid] * view[tid]

def square(view):
    """
    Squares argument element-wise

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if str(view.dtype) == "DataType.double":
        out = pk.View([view.shape[0]], pk.double)
        pk.parallel_for(
            view.shape[0],
            square_impl_1d_double,
            view=view,
            out=out)

    elif str(view.dtype) == "DataType.float":
        out = pk.View([view.shape[0]], pk.float)
        pk.parallel_for(
            view.shape[0],
            square_impl_1d_float,
            view=view,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out

@pk.workunit
def greater_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.uint16]):
    out[tid] = viewA[tid] > viewB[tid]

@pk.workunit
def greater_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.uint16]):
    out[tid] = viewA[tid] > viewB[tid]


def greater(viewA, viewB):
    """
    Return the truth value of viewA > viewB element-wise.

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view (uint16)
           Output view.

    """
    out = pk.View([viewA.shape[0]], pk.uint16)
    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        pk.parallel_for(
            viewA.shape[0],
            greater_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
        pk.parallel_for(
            viewA.shape[0],
            greater_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out

@pk.workunit
def logaddexp_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double],):
    out[tid] = log(exp(viewA[tid]) + exp(viewB[tid]))

@pk.workunit
def logaddexp_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float],):
    out[tid] = log(exp(viewA[tid]) + exp(viewB[tid]))


def logaddexp(viewA, viewB):
    """
    Return a view with log(exp(a) + exp(b)) calculate for
    positionally corresponding elements in viewA and viewB

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            logaddexp_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
            viewA.shape[0],
            logaddexp_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out
=======
def equal_impl_1d_double(tid: int,
                         view1: pk.View1D[pk.double],
                         view2: pk.View1D[pk.double],
                         view2_size: int,
                         view_result: pk.View1D[pk.uint16]):
    view2_idx: int = 0
    if view2_size == 1:
        view2_idx = 0
    else:
        view2_idx = tid
    if view1[tid] == view2[view2_idx]:
        view_result[tid] = 1
    else:
        view_result[tid] = 0


@pk.workunit
def equal_impl_1d_uint16(tid: int,
                         view1: pk.View1D[pk.uint16],
                         view2: pk.View1D[pk.uint16],
                         view2_size: int,
                         view_result: pk.View1D[pk.uint16]):
    view2_idx: int = 0
    if view2_size == 1:
        view2_idx = 0
    else:
        view2_idx = tid
    if view1[tid] == view2[view2_idx]:
        view_result[tid] = 1
    else:
        view_result[tid] = 0


@pk.workunit
def equal_impl_1d_int16(tid: int,
                         view1: pk.View1D[pk.int16],
                         view2: pk.View1D[pk.int16],
                         view2_size: int,
                         view_result: pk.View1D[pk.uint16]):
    view2_idx: int = 0
    if view2_size == 1:
        view2_idx = 0
    else:
        view2_idx = tid
    if view1[tid] == view2[view2_idx]:
        view_result[tid] = 1
    else:
        view_result[tid] = 0


@pk.workunit
def equal_impl_1d_int32(tid: int,
                         view1: pk.View1D[pk.int32],
                         view2: pk.View1D[pk.int32],
                         view2_size: int,
                         view_result: pk.View1D[pk.uint16]):
    view2_idx: int = 0
    if view2_size == 1:
        view2_idx = 0
    else:
        view2_idx = tid
    if view1[tid] == view2[view2_idx]:
        view_result[tid] = 1
    else:
        view_result[tid] = 0


@pk.workunit
def equal_impl_1d_int64(tid: int,
                         view1: pk.View1D[pk.int64],
                         view2: pk.View1D[pk.int64],
                         view2_size: int,
                         view_result: pk.View1D[pk.uint16]):
    view2_idx: int = 0
    if view2_size == 1:
        view2_idx = 0
    else:
        view2_idx = tid
    if view1[tid] == view2[view2_idx]:
        view_result[tid] = 1
    else:
        view_result[tid] = 0

def equal(view1, view2):
    # TODO: write even more dispatching for cases where view1 and view2
    # have different, but comparable, types (like float32 vs. float64?)
    # this may "explode" without templating

    if sum(view1.shape) == 0 or sum(view2.shape) == 0:
        return np.empty(shape=(0,))

    if view1.shape != view2.shape:
        if not view1.size <= 1 and not view2.size <= 1:
            # TODO: supporting __eq__ over broadcasted shapes beyond
            # scalar (i.e., matching number of columns)
            raise ValueError("view1 and view2 have incompatible shapes")

    # TODO: something more appropriate than uint16 as a proxy
    # for the bool type? (a shorter integer like uint8
    # at least?)
    view_result = pk.View([*view1.shape], dtype=pk.uint16)

    # NOTE: the blocks below are asymmetric on view1 vs view2,
    # and also quite awkward--they evolved from making the array API
    # test_ones() test pass, but need refinement or removal eventually
    try:
        if isinstance(view2.array, np.ndarray):
            if view2.size <= 1:
                new_shape = (1,)
            else:
                new_shape = view2.shape
            view2r = pk.View([*new_shape], dtype=view2.dtype)
            view2r[:] = view2.array
            view2 = view2r
    except AttributeError:
        pass
    try:
        if isinstance(view1.array, np.ndarray):
            if view1.shape == () or view1.shape == (0,):
                view1r = pk.View([1], dtype=view1.dtype)
                view1r[:] = view1.array
                view1 = view1r
    except AttributeError:
        pass

    if ("double" in str(view1.dtype) or "float64" in str(view1.dtype) and
       ("double" in str(view2.dtype) or "float64" in str(view2.dtype))):
        pk.parallel_for(view1.size,
                        equal_impl_1d_double,
                        view1=view1,
                        view2=view2,
                        view2_size=view2.size,
                        view_result=view_result)
    elif (("uint16" in str(view1.dtype) or "bool" in str(view1.dtype)) and
          ("uint16" in str(view2.dtype) or "bool" in str(view2.dtype))):
        pk.parallel_for(view1.size,
                        equal_impl_1d_uint16,
                        view1=view1,
                        view2=view2,
                        view2_size=view2.size,
                        view_result=view_result)
    elif "int16" in str(view1.dtype) and "int16" in str(view1.dtype):
        pk.parallel_for(view1.size,
                        equal_impl_1d_int16,
                        view1=view1,
                        view2=view2,
                        view2_size=view2.size,
                        view_result=view_result)
    elif "int32" in str(view1.dtype) and "int32" in str(view1.dtype):
        pk.parallel_for(view1.size,
                        equal_impl_1d_int32,
                        view1=view1,
                        view2=view2,
                        view2_size=view2.size,
                        view_result=view_result)
    elif "int64" in str(view1.dtype) and "int64" in str(view1.dtype):
        pk.parallel_for(view1.size,
                        equal_impl_1d_int64,
                        view1=view1,
                        view2=view2,
                        view2_size=view2.size,
                        view_result=view_result)
    else:
        # TODO: include the view types in the error message
        raise NotImplementedError("equal ufunc not implemented for this comparison")

    return view_result
