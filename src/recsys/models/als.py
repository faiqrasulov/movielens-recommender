"""Implicit-feedback ALS (Hu, Koren & Volinsky, 2008) with exact or conjugate-gradient solves.

Objective over *all* user-item pairs::

    L = sum_{u,i} c_ui (p_ui - x_u . y_i)^2 + reg (sum_u |x_u|^2 + sum_i |y_i|^2)

with preference ``p_ui = 1`` for observed pairs (0 otherwise) and confidence
``c_ui = 1 + alpha * r_ui``. Fixing one side, every row is a weighted ridge regression.
The ``Y^T Y`` trick makes each solve cost O(n_u f^2 + f^3) instead of O(n_items f^2):
``Y^T C_u Y = Y^T Y + alpha * sum_{i in u} r_ui y_i y_i^T``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recsys.models.base import EpochCallback, Recommender
from recsys.split import TrainData


def _row_ids(R: sp.csr_matrix) -> np.ndarray:
    return np.repeat(np.arange(R.shape[0]), np.diff(R.indptr))


def _with_data(R: sp.csr_matrix, data: np.ndarray) -> sp.csr_matrix:
    return sp.csr_matrix((data, R.indices, R.indptr), shape=R.shape)


def solve_exact(
    R: sp.csr_matrix, F: np.ndarray, reg: float, alpha: float, chunk: int = 256
) -> np.ndarray:
    """Exact least-squares update of every row of ``R`` given the fixed factors ``F``.

    ``sum_{i in u} r_ui y_i y_i^T`` is computed for all rows at once as ``R @ vec(y_i y_i^T)``
    and the small ``f x f`` systems are solved in batches.
    """
    n, f = R.shape[0], F.shape[1]
    base = F.T @ F + reg * np.eye(f)
    b = _with_data(R, 1.0 + alpha * R.data) @ F
    outer = (F[:, :, None] * F[:, None, :]).reshape(F.shape[0], f * f)
    out = np.empty((n, f))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        A = np.asarray(alpha * R[start:stop] @ outer).reshape(stop - start, f, f) + base
        out[start:stop] = np.linalg.solve(A, b[start:stop, :, None])[:, :, 0]
    return out


def solve_cg(
    R: sp.csr_matrix, F: np.ndarray, X0: np.ndarray, reg: float, alpha: float, steps: int = 3
) -> np.ndarray:
    """A few conjugate-gradient steps on every row's system, warm-started at ``X0``
    (Takacs, Pilaszy & Tikk, 2011). All rows are iterated in lockstep with per-row step sizes;
    the products ``A_u v`` never form ``A_u``: ``(Y^T Y + reg I) v + alpha Y_u^T (r_u * Y_u v)``.
    """
    base = F.T @ F + reg * np.eye(F.shape[1])
    rows = _row_ids(R)
    F_obs = F[R.indices]
    weights = alpha * R.data

    def apply(V: np.ndarray) -> np.ndarray:
        s = np.einsum("nf,nf->n", V[rows], F_obs)
        return V @ base + _with_data(R, weights * s) @ F

    b = _with_data(R, 1.0 + alpha * R.data) @ F
    x = X0.copy()
    r = b - apply(x)
    p = r.copy()
    rs = np.einsum("nf,nf->n", r, r)
    for _ in range(steps):
        Ap = apply(p)
        pAp = np.einsum("nf,nf->n", p, Ap)
        step = np.divide(rs, pAp, out=np.zeros_like(rs), where=pAp > 1e-30)
        x += step[:, None] * p
        r -= step[:, None] * Ap
        rs_new = np.einsum("nf,nf->n", r, r)
        beta = np.divide(rs_new, rs, out=np.zeros_like(rs), where=rs > 1e-30)
        p = r + beta[:, None] * p
        rs = rs_new
    return x


def implicit_objective(
    R: sp.csr_matrix, U: np.ndarray, V: np.ndarray, reg: float, alpha: float
) -> float:
    """Full HKV objective, including every unobserved pair, in O(nnz f + (n + m) f^2).

    ``sum_all (x.y)^2 = trace(U^T U V^T V)``; observed pairs add
    ``c (1 - s)^2 - s^2 = alpha r s^2 + c (1 - 2 s)``.
    """
    s = np.einsum("nf,nf->n", U[_row_ids(R)], V[R.indices])
    c = 1.0 + alpha * R.data
    loss = float(np.sum((U.T @ U) * (V.T @ V)))
    loss += float(np.sum(alpha * R.data * s**2 + c * (1.0 - 2.0 * s)))
    return loss + reg * float(np.sum(U**2) + np.sum(V**2))


class ImplicitALS(Recommender):
    """Weighted matrix factorisation for implicit feedback, fitted by alternating least squares.

    ``solver="cg"`` (default) runs ``cg_steps`` warm-started conjugate-gradient steps per
    half-iteration; ``solver="exact"`` solves every ``f x f`` system exactly. Both decrease the
    objective monotonically; ``loss_history_`` records it after every iteration.
    """

    name = "iALS"

    def __init__(
        self,
        factors: int = 64,
        reg: float = 10.0,
        alpha: float = 10.0,
        iterations: int = 15,
        solver: str = "cg",
        cg_steps: int = 3,
        seed: int = 0,
    ) -> None:
        if solver not in ("cg", "exact"):
            raise ValueError("solver must be 'cg' or 'exact'")
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.solver = solver
        self.cg_steps = cg_steps
        self.seed = seed

    def _half(self, R: sp.csr_matrix, F: np.ndarray, X0: np.ndarray) -> np.ndarray:
        if self.solver == "exact":
            return solve_exact(R, F, self.reg, self.alpha)
        return solve_cg(R, F, X0, self.reg, self.alpha, self.cg_steps)

    def fit(self, train: TrainData, callback: EpochCallback | None = None) -> ImplicitALS:
        R = train.X.tocsr().astype(np.float64)
        Rt = R.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.user_factors_ = rng.normal(0.0, 0.01, (R.shape[0], self.factors))
        self.item_factors_ = rng.normal(0.0, 0.01, (R.shape[1], self.factors))
        self.loss_history_ = [implicit_objective(R, self.user_factors_, self.item_factors_,
                                                 self.reg, self.alpha)]
        for it in range(self.iterations):
            self.user_factors_ = self._half(R, self.item_factors_, self.user_factors_)
            self.item_factors_ = self._half(Rt, self.user_factors_, self.item_factors_)
            self.loss_history_.append(implicit_objective(
                R, self.user_factors_, self.item_factors_, self.reg, self.alpha))
            if callback is not None:
                callback(it + 1, self)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.user_factors_[users] @ self.item_factors_.T
