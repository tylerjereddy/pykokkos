import pykokkos as pk


@pk.workunit
def arange_impl_1d_float(tid: int,
                          start: float,
                          stop: float,
                          step: float,
                          out: pk.View1D[pk.float]):
    counter: int = 0
    for i in range(start, stop, step):
        if tid == counter:
            out[tid] = i
            break
        counter += 1


@pk.workunit
def arange_impl_1d_double(tid: int,
                          start: float,
                          stop: float,
                          step: float,
                          out: pk.View1D[pk.double]):
    counter: int = 0
    for i in range(start, stop, step):
        if tid == counter:
            out[tid] = i
            break
        counter += 1



@pk.workunit
def arange_impl_1d_int8(tid: int,
                         start: int,
                         stop: int,
                         step: int,
                         out: pk.View1D[pk.int8]):
    counter: int = 0
    for i in range(start, stop, step):
        if tid == counter:
            out[tid] = i
            break
        counter += 1


@pk.workunit
def arange_impl_1d_int16(tid: int,
                         start: int,
                         stop: int,
                         step: int,
                         out: pk.View1D[pk.int16]):
    counter: int = 0
    for i in range(start, stop, step):
        if tid == counter:
            out[tid] = i
            break
        counter += 1


@pk.workunit
def arange_impl_1d_int32(tid: int,
                         start: int,
                         stop: int,
                         step: int,
                         out: pk.View1D[pk.int32]):
    counter: int = 0
    for i in range(start, stop, step):
        if tid == counter:
            out[tid] = i
            break
        counter += 1


@pk.workunit
def arange_impl_1d_int64(tid: int,
                         start: int,
                         stop: int,
                         step: int,
                         out: pk.View1D[pk.int64]):
    counter: int = 0
    for i in range(start, stop, step):
        printf("tid, i: %d, %d\n", tid, i);
        if tid == counter:
            out[tid] = i
            break
        counter += 1


@pk.workunit
def arange_impl_1d_uint8(tid: int,
                         start: int,
                         stop: int,
                         step: int,
                         out: pk.View1D[pk.uint8]):
    counter: int = 0
    for i in range(start, stop, step):
        if tid == counter:
            out[tid] = i
            break
        counter += 1


@pk.workunit
def arange_impl_1d_uint16(tid: int,
                         start: int,
                         stop: int,
                         step: int,
                         out: pk.View1D[pk.uint16]):
    counter: int = 0
    for i in range(start, stop, step):
        if tid == counter:
            out[tid] = i
            break
        counter += 1


@pk.workunit
def arange_impl_1d_uint32(tid: int,
                         start: int,
                         stop: int,
                         step: int,
                         out: pk.View1D[pk.uint32]):
    counter: int = 0
    for i in range(start, stop, step):
        if tid == counter:
            out[tid] = i
            break
        counter += 1


@pk.workunit
def arange_impl_1d_uint64(tid: int,
                         start: int,
                         stop: int,
                         step: int,
                         out: pk.View1D[pk.uint64]):
    counter: int = 0
    for i in range(start, stop, step):
        if tid == counter:
            out[tid] = i
            break
        counter += 1
