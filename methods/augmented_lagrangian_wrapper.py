"""
Augmented Lagrangian Solver for JAXopt
=======================================
Solves constrained optimisation problems of the form:

    min  f(x)
    s.t. h(x) = 0          (equality constraints)
         g(x) <= 0         (inequality constraints)

using the augmented Lagrangian / multiplier method with:
  - LBFGS (or any jaxopt solver) as the inner minimiser
  - Bertsekas selective penalty update to avoid ill-conditioning
  - Fully JIT-compiled outer loop via jax.lax.while_loop
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


# ── Outer-loop state ──────────────────────────────────────────────────────────

class AugLagState(NamedTuple):
    """State carried across outer augmented Lagrangian iterations."""
    iter_num:    jnp.ndarray   # int32  scalar — outer iteration counter
    inner_state: Any           # state owned by the inner jaxopt solver
    dual_eq:     jnp.ndarray   # equality multipliers λ
    dual_ineq:   jnp.ndarray   # inequality multipliers μ  (≥ 0)
    c:           jnp.ndarray   # float32 scalar — current penalty coefficient
    error:       jnp.ndarray   # float32 scalar — current constraint violation
    prev_error:  jnp.ndarray   # float32 scalar — violation at previous iterate


# ── Solver ────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class AugmentedLagrangianSolver:
    """Augmented Lagrangian method wrapping any jaxopt iterative solver.

    Parameters
    ----------
    fun:
        Objective f(params, *args, **kwargs) to minimise.
    eq_constr:
        Equality constraint h(params, *args, **kwargs) → array.
        Satisfied when h = 0.  Defaults to a zero vector (unconstrained).
    ineq_constr:
        Inequality constraint g(params, *args, **kwargs) → array.
        Satisfied when g ≤ 0.  Defaults to a zero vector (unconstrained).
    BaseSolverCls:
        jaxopt solver class used for inner minimisation.  Defaults to LBFGS.
    c_init:
        Initial penalty coefficient.
    c_max:
        Maximum penalty coefficient.  Caps growth to avoid ill-conditioning.
    alpha:
        Penalty growth factor (> 1).  Applied only when the Bertsekas
        criterion determines that c needs to increase.
    eta:
        Bertsekas criterion threshold.  If the current violation exceeds
        eta * previous violation, c is grown; otherwise it is held fixed.
        Smaller eta -> more aggressive penalty growth.
    maxiter:
        Maximum number of outer iterations.
    tol:
        Convergence tolerance on the constraint violation.
    inner_kwargs:
        Extra keyword arguments forwarded to BaseSolverCls (e.g. maxiter).
    """

    fun:           Callable
    eq_constr:     Callable        = field(default_factory=lambda: _eq_none)
    ineq_constr:   Callable        = field(default_factory=lambda: _ineq_none)
    BaseSolverCls: type            = LBFGS
    c_init:        float           = 1.0
    c_max:         float           = 1e6
    alpha:         float           = 10.0
    eta:           float           = 0.25
    maxiter:       int             = 50
    tol:           float           = 1e-7
    inner_kwargs:  Optional[dict]  = None

    def __post_init__(self):
        if self.inner_kwargs is None:
            self.inner_kwargs = {}

        # Augmented Lagrangian (scaled form):
        #
        #   L(x; lambda, mu, c) = f(x)
        #                       + (c/2) ||h(x) + lambda/c||^2
        #                       + (c/2) ||max(0, g(x) + mu/c)||^2
        #
        # Dual variables are passed as extra positional args so that
        # jaxopt's run/update machinery can forward them transparently.
        def _auglag(params, dual_eq, dual_ineq, c, *args, **kwargs):
            f_val    = self.fun(params, *args, **kwargs)
            eq_val   = self.eq_constr(params,   *args, **kwargs)
            ineq_val = self.ineq_constr(params, *args, **kwargs)

            eq_penalty   = (c / 2.0) * jnp.sum(
                (eq_val + dual_eq / c) ** 2
            )
            ineq_penalty = (c / 2.0) * jnp.sum(
                jnp.maximum(0.0, ineq_val + dual_ineq / c) ** 2
            )
            return f_val + eq_penalty + ineq_penalty

        self._auglag = _auglag
        self._inner_solver = self.BaseSolverCls(
            fun=self._auglag,
            **self.inner_kwargs,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def init_state(self, init_params: Any, *args, **kwargs) -> AugLagState:
        """Initialise dual variables to zero and build the inner solver state."""
        eq_shape   = self.eq_constr(init_params,   *args, **kwargs).shape
        ineq_shape = self.ineq_constr(init_params, *args, **kwargs).shape

        dual_eq   = jnp.zeros(eq_shape)
        dual_ineq = jnp.zeros(ineq_shape)
        c         = self.c_init

        inner_state = self._inner_solver.init_state(
            init_params, dual_eq, dual_ineq, c, *args, **kwargs
        )

        return AugLagState(
            iter_num=0,
            inner_state=inner_state,
            dual_eq=dual_eq,
            dual_ineq=dual_ineq,
            c=c,
            error=jnp.inf,
            prev_error=jnp.inf,
        )

    def update(
        self, params: Any, state: AugLagState, *args, **kwargs
    ) -> tuple[Any, AugLagState]:
        """One outer iteration:

        1. Inner minimisation  (nested while_loop via inner solver's run())
        2. Bertsekas selective penalty update
        3. Dual variable updates
        """
        dual_eq   = state.dual_eq
        dual_ineq = state.dual_ineq
        c         = state.c

        # 1. Inner minimisation — full convergence via nested while_loop.
        #    JAX supports this; it compiles each loop independently.
        params_new, inner_state_new = self._inner_solver.run(
            params, dual_eq, dual_ineq, c, *args, **kwargs
        )

        # 2. Constraint violations at the new primal point.
        eq_val   = self.eq_constr(params_new,   *args, **kwargs)
        ineq_val = self.ineq_constr(params_new, *args, **kwargs)

        error = jnp.maximum(
            jnp.max(jnp.abs(eq_val)),
            jnp.max(jnp.maximum(0.0, ineq_val)),
        )

        # 3. Bertsekas criterion: only grow c if violation is not decreasing
        #    fast enough.  Avoids ill-conditioning when dual updates alone
        #    are sufficient to drive feasibility.
        c_new = jnp.where(
            error > self.eta * state.prev_error,
            jnp.minimum(self.alpha * c, self.c_max),
            c,
        )

        # 4. Dual updates:
        #    equality:   lambda <- lambda + c * h(x*)
        #    inequality: mu     <- max(0, mu + c * g(x*))
        dual_eq_new   = dual_eq + c_new * eq_val
        dual_ineq_new = jnp.maximum(0.0, dual_ineq + c_new * ineq_val)

        return params_new, AugLagState(
            iter_num=state.iter_num + 1,
            inner_state=inner_state_new,
            dual_eq=dual_eq_new,
            dual_ineq=dual_ineq_new,
            c=c_new,
            error=error,
            prev_error=error,
        )

    def run(
        self, init_params: Any, *args, **kwargs
    ) -> tuple[Any, AugLagState]:
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

        print('Solving...')

        return lax.while_loop(cond_fun, body_fun, (init_params, state))

    def jit_run(
        self, init_params: Any, *args, **kwargs
    ) -> tuple[Any, AugLagState]:
        """Compile run() once, cache, and reuse for repeated solves."""
        if not hasattr(self, "_jit_run_fn"):
            self._jit_run_fn = jax.jit(self.run)
        return self._jit_run_fn(init_params, *args, **kwargs)