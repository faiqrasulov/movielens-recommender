import os

# Keep BLAS single-threaded in tests: the matrices are tiny and the machine may be shared.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from recsys.synthetic import synthetic_interactions  # noqa: E402


@pytest.fixture(scope="session")
def toy_data():
    return synthetic_interactions(n_users=40, n_items=60, seed=1)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
