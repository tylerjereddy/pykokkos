"""
Record sqrt() ufunc performance.
"""

import os

import pykokkos as pk

import numpy as np
import cupy as cp
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    scenario_name = "pk_gp160_sqrt"
    current_sqrt_fn = "pk.sqrt"
    space = pk.ExecutionSpace.OpenMP
    #space = pk.ExecutionSpace.Cuda
    pk.set_default_space(space)

    num_global_repeats = 50
    num_repeats = 5000
    array_size_1d = int(1e4)


    import timeit
    rng = np.random.default_rng(18898787)
    arr = rng.random(array_size_1d).astype(float)
    view = pk.from_numpy(arr)
    arr = view
    #arr = cp.array(arr)

    num_threads = os.environ.get("OMP_NUM_THREADS")
    if num_threads is None:
        raise ValueError("must set OMP_NUM_THREADS for benchmarks!")

    df = pd.DataFrame(np.full(shape=(num_global_repeats, 2), fill_value=np.nan),
                      columns=["scenario", "time (s)"])
    df["scenario"] = df["scenario"].astype(str)
    print("df before trials:\n", df)

    counter = 0
    for global_repeat in tqdm(range(1, num_global_repeats + 1)):
        sqrt_time_sec = timeit.timeit(f"{current_sqrt_fn}(arr)",
                                      globals=globals(),
                                      number=num_repeats)
        df.iloc[counter, 0] = f"{scenario_name}"
        df.iloc[counter, 1] = sqrt_time_sec
        counter += 1

    print("df after trials:\n", df)

    filename = f"{scenario_name}_array_size_1d_{array_size_1d}_{num_global_repeats}_tials_{num_repeats}_calls_per_trial.parquet.gzip"
    df.to_parquet(filename,
                  engine="pyarrow",
                  compression="gzip")
