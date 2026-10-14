"""
ADMM Solver for JAXopt
======================
Solves constrained optimisation problems of the form:

    min  f(x)
    s.t. h(x) = 0          (equality constraints)
         g(x) <= 0         (inequality constraints)

via the Alternating Direction Method of Multipliers (ADMM).

Splitting strategy
------------------
Introduce auxiliary variables z_eq and z_ineq and rewrite as:

    min  f(x)
    s.t. h(x) = z_eq          z_eq  = 0       (equality target)
         g(x) = z_ineq        z_ineq <= 0     (inequality target)

Each ADMM iteration has three steps:

  x-update  (inner solver):
      x <- argmin_x  f(x)
                   + (rho/2) ||h(x) + u_eq||^2
                   + (rho/2) ||max(0, g(x) + u_ineq)||^2

      The inequality term is the closed-form result of substituting
      z_ineq = min(0, g(x) + u_ineq) into the generic ADMM penalty,
      so that the x-step and z-step are always consistent.

  z-update  (closed-form):
      z_eq   <- 0                          (fixed equality target)
      z_ineq <- min(0, g(x) + u_ineq)     (project onto (-inf, 0])

  u-update  (scaled dual ascent):
      u_eq   <- u_eq   + h(x)
      u_ineq <- u_ineq + g(x) - z_ineq

Convergence is measured by the primal residual:
      error = max( max|h(x)|, max(0, g(x)) )

Demo: constrained quadratic — same problem as the augmented Lagrangian demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Optional

import jax
jax.config.update("jax_enable_x64", True)  # double precision for better stability
import jax.numpy as jnp
import jax.lax as lax

from jaxopt import LBFGS


# ── Default no-op constraints ─────────────────────────────────────────────────

def _eq_none(params, *args, **kwargs):
    return jnp.zeros(1)

def _ineq_none(params, *args, **kwargs):
    return jnp.zeros(1)


# ── ADMM state ────────────────────────────────────────────────────────────────

class ADMMState(NamedTuple):
    """State carried across ADMM iterations.

    Follows scaled-form ADMM convention: u = dual / rho (scaled dual).
    """
    iter_num:    jnp.ndarray   # scalar — iteration counter
    inner_state: Any           # inner jaxopt solver state
    z_ineq:      jnp.ndarray   # auxiliary for inequality: z_ineq <= 0
    u_eq:        jnp.ndarray   # scaled dual for equality
    u_ineq:      jnp.ndarray   # scaled dual for inequality
    error:       jnp.ndarray   # primal constraint violation


# ── Solver ────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class ADMMSolver:
    """ADMM wrapper for constrained optimisation over a jaxopt inner solver.

    Parameters
    ----------
    fun:
        Objective f(params, *args, **kwargs) to minimise.
    eq_constr:
        Equality constraint h(params, *args, **kwargs) -> array.
        Satisfied when h = 0.
    ineq_constr:
        Inequality constraint g(params, *args, **kwargs) -> array.
        Satisfied when g <= 0.
    BaseSolverCls:
        jaxopt solver class for the x-update subproblem.  Defaults to LBFGS.
    rho:
        ADMM penalty parameter.  Controls convergence speed.
        Too small -> slow primal convergence.
        Too large -> slow dual convergence / ill-conditioned x-update.
        Typical range: 1 -- 100 depending on problem scaling.
    maxiter:
        Maximum number of ADMM iterations.
    tol:
        Convergence tolerance on the primal constraint violation.
    inner_kwargs:
        Extra keyword arguments forwarded to BaseSolverCls (e.g. maxiter).
    """

    fun:           Callable
    eq_constr:     Callable       = field(default_factory=lambda: _eq_none)
    ineq_constr:   Callable       = field(default_factory=lambda: _ineq_none)
    BaseSolverCls: type           = LBFGS
    rho:           float          = 10.0
    maxiter:       int            = 200
    tol:           float          = 1e-6
    inner_kwargs:  Optional[dict] = None

    def __post_init__(self):
        if self.inner_kwargs is None:
            self.inner_kwargs = {}

        # x-update objective (scaled-form ADMM):
        #
        #   L_rho(x; u) = f(x)
        #               + (rho/2) ||h(x) + u_eq||^2
        #               + (rho/2) ||max(0, g(x) + u_ineq)||^2
        #
        # Equality: standard scaled ADMM with z_eq = 0 fixed, so the
        # penalty is simply (rho/2)||h(x) + u_eq||^2.
        #
        # Inequality: derived by substituting the z-update
        # z_ineq = min(0, g(x) + u_ineq) into (rho/2)||g(x) - z_ineq + u_ineq||^2:
        #   g+u <= 0 (feasible): z = g+u  -> penalty = 0
        #   g+u >  0 (violated): z = 0    -> penalty = (rho/2)(g+u)^2
        # = (rho/2) max(0, g(x) + u_ineq)^2
        #
        # z_ineq is passed as an arg to keep the call signature stable
        # across iterations but is not used in the objective.
        def _x_obj(params, z_ineq, u_eq, u_ineq, rho, *args, **kwargs):
            f_val    = self.fun(params, *args, **kwargs)
            eq_val   = self.eq_constr(params,   *args, **kwargs)
            ineq_val = self.ineq_constr(params, *args, **kwargs)

            eq_aug   = (rho / 2.0) * jnp.sum((eq_val + u_eq) ** 2)
            ineq_aug = (rho / 2.0) * jnp.sum(jnp.maximum(0.0, ineq_val + u_ineq) ** 2)

            return f_val + eq_aug + ineq_aug

        self._x_obj = _x_obj
        self._inner_solver = self.BaseSolverCls(
            fun=self._x_obj,
            **self.inner_kwargs,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def init_state(self, init_params: Any, *args, **kwargs) -> ADMMState:
        """Initialise z and u to zero; build inner solver state."""
        eq_shape   = self.eq_constr(init_params,   *args, **kwargs).shape
        ineq_shape = self.ineq_constr(init_params, *args, **kwargs).shape

        z_ineq = jnp.zeros(ineq_shape)
        u_eq   = jnp.zeros(eq_shape)
        u_ineq = jnp.zeros(ineq_shape)
        rho    = self.rho

        inner_state = self._inner_solver.init_state(
            init_params, z_ineq, u_eq, u_ineq, rho, *args, **kwargs
        )

        return ADMMState(
            iter_num=0,
            inner_state=inner_state,
            z_ineq=z_ineq,
            u_eq=u_eq,
            u_ineq=u_ineq,
            error=jnp.inf,
        )

    def update(
        self, params: Any, state: ADMMState, *args, **kwargs
    ) -> tuple[Any, ADMMState]:
        """One ADMM iteration: x-update -> z-update -> u-update."""
        z_ineq = state.z_ineq
        u_eq   = state.u_eq
        u_ineq = state.u_ineq
        rho    = self.rho

        # ── Step 1: x-update ─────────────────────────────────────────────────
        params_new, inner_state_new = self._inner_solver.run(
            params, z_ineq, u_eq, u_ineq, rho, *args, **kwargs
        )

        eq_val   = self.eq_constr(params_new,   *args, **kwargs)
        ineq_val = self.ineq_constr(params_new, *args, **kwargs)

        # ── Step 2: z-update ─────────────────────────────────────────────────
        # z_eq = 0 always (fixed equality target, no state needed)
        z_ineq_new = jnp.minimum(0.0, ineq_val + u_ineq)

        # ── Step 3: u-update ─────────────────────────────────────────────────
        u_eq_new   = u_eq   + eq_val
        u_ineq_new = u_ineq + ineq_val - z_ineq_new

        # ── Primal residual ───────────────────────────────────────────────────
        error = jnp.maximum(
            jnp.max(jnp.abs(eq_val)),
            jnp.max(jnp.maximum(0.0, ineq_val)),
        )

        return params_new, ADMMState(
            iter_num=state.iter_num + 1,
            inner_state=inner_state_new,
            z_ineq=z_ineq_new,
            u_eq=u_eq_new,
            u_ineq=u_ineq_new,
            error=error,
        )

    def run(
        self, init_params: Any, *args, **kwargs
    ) -> tuple[Any, ADMMState]:
        """Fully JIT-compiled outer loop via jax.lax.while_loop."""
        state = self.init_state(init_params, *args, **kwargs)

        tol     = self.tol
        maxiter = self.maxiter

        def cond_fun(carry):
            _, s = carry
            return (s.error > tol) & (s.iter_num < maxiter)

        def body_fun(carry):
            p, s = carry
            return self.update(p, s, *args, **kwargs)

        return lax.while_loop(cond_fun, body_fun, (init_params, state))

    def jit_run(
        self, init_params: Any, *args, **kwargs
    ) -> tuple[Any, ADMMState]:
        """Compile run() once, cache, and reuse for repeated solves."""
        if not hasattr(self, "_jit_run_fn"):
            self._jit_run_fn = jax.jit(self.run)
        return self._jit_run_fn(init_params, *args, **kwargs)


# =============================================================================
# Demo: constrained quadratic (same problem as augmented Lagrangian demo)
# =============================================================================

def run_demo():
    """
        min  13x₀² + 10x₀x₁ + 7x₁² + x₀ + x₁
        s.t. x₀ + x₁ = 1          (equality)
             x₀      >= 0.2       (inequality: -x₀ + 0.2 <= 0)

    Both constraints are active at the solution: x₀=0.2, x₁=0.8.
    """
    import time

    def f(params):
        x = params["x"]
        return 13*x[0]**2 + 10*x[0]*x[1] + 7*x[1]**2 + x[0] + x[1]

    def eq_c(params):
        x = params["x"]
        return jnp.array([x[0] + x[1] - 1.0])

    def ineq_c(params):
        x = params["x"]
        return jnp.array([-x[0] + 0.2])        # g(x) <= 0  =>  x₀ >= 0.2

    solver = ADMMSolver(
        fun=f,
        eq_constr=eq_c,
        ineq_constr=ineq_c,
        BaseSolverCls=LBFGS,
        rho=10.0,
        maxiter=200,
        tol=1e-6,
        inner_kwargs=dict(maxiter=200),
    )

    init_params = {"x": jnp.array([0.5, 0.5])}

    print("=" * 50)
    print("  ADMM — Constrained Quadratic Demo")
    print("=" * 50)

    t0 = time.perf_counter()
    params, state = solver.jit_run(init_params)
    jax.block_until_ready(params)
    t1 = time.perf_counter()
    print(f"First call  (compile + solve): {t1-t0:.2f}s")

    t0 = time.perf_counter()
    params, state = solver.jit_run(init_params)
    jax.block_until_ready(params)
    t1 = time.perf_counter()
    print(f"Second call (solve only):      {t1-t0:.4f}s\n")

    x = params["x"]
    print(f"Solution:    x₀={float(x[0]):.6f},  x₁={float(x[1]):.6f}")
    print(f"             (expect x₀=0.2, x₁=0.8)")
    print(f"f(x):        {float(f(params)):.6f}  (expect 7.6)")
    print(f"x₀ + x₁:     {float(x[0]+x[1]):.6f}  (target 1.0)")
    print(f"x₀:          {float(x[0]):.6f}  (lower bound 0.2)")
    print(f"Iterations:  {int(state.iter_num)}")
    print(f"Error:       {float(state.error):.2e}")


if __name__ == "__main__":
    run_demo()