import jax
from functools import partial
from jax import value_and_grad, grad, jacfwd, vmap, jit, hessian
from jax.flatten_util import ravel_pytree
import jaxopt
import jax.numpy as np
import sys

class AugmentedLagrangeWrapper(object):
    def __init__(self, Solver, x0, loss, eq_constr, ineq_constr,
                    opt_args=None, step_size=1e-3, c=1.0, *args, **kwargs):

        self.opt_args = opt_args
        self.loss = loss
        self._c_def = c
        self.c = c
        self.eq_constr = eq_constr
        self.ineq_constr = ineq_constr
        _eq_constr = eq_constr(x0, opt_args)
        _ineq_constr = ineq_constr(x0, opt_args)
        lam = np.zeros(_eq_constr.shape)
        mu = np.zeros(_ineq_constr.shape)
        self.solution = x0
        self.dual_solution = {'lam' : lam, 'mu' : mu}

        def lagrangian(solution, dual_solution, opt_args, c):
            lam = dual_solution['lam']
            mu  = dual_solution['mu']
            _eq_constr = eq_constr(solution, opt_args)
            _ineq_constr = ineq_constr(solution, opt_args)
            _eq_penalty = c * 0.5 * np.sum((_eq_constr + lam/c)**2)
            _ineq_penalty = c * 0.5 * np.sum(np.maximum(0., mu/c + _ineq_constr)**2)
            return loss(solution, opt_args) \
                + _eq_penalty \
                + _ineq_penalty

        self._unc_solver = Solver(lagrangian, *args, **kwargs)
        self._solver_state = self._unc_solver.init_state(self.solution, self.dual_solution, opt_args, c)

        val_dldx = jit(value_and_grad(lagrangian))

        @jit
        def step(solution, solver_state, dual_solution, opt_args, c):
            (solution, solver_state) = self._unc_solver.run(
                init_params=solution,
                dual_solution=dual_solution,
                opt_args=opt_args,
                c=c
            )
            _val, _dldx = val_dldx(solution, dual_solution, opt_args, c)

            dual_solution['lam'] = np.clip(dual_solution['lam'] + c*eq_constr(solution, opt_args), -10,10)
            dual_solution['mu'] = np.clip(np.maximum(0., dual_solution['mu'] + c*ineq_constr(solution, opt_args)), 0,10)
            return solution, dual_solution, solver_state, _val, _dldx

        self.lagrangian = lagrangian
        self.grad_lagrangian = val_dldx
        self.step = step

    def get_solution(self):
        return self.solution

    def update_progress(self, k, _grad_total):
        sys.stdout.write('\033[1A')
        sys.stdout.write('\033[K')
        sys.stdout.write(f"{k}, {_grad_total}\n")
        sys.stdout.flush()

    def solve(self, opt_args=None, max_iter=100, eps=1e-5, alpha=1.00001):
        print(' ')
        if opt_args is None:
            opt_args = self.opt_args

        for k in range(max_iter):
            self.solution, self.dual_solution, self._solver_state, _val, _dldx = self.step(self.solution, self._solver_state, self.dual_solution, opt_args, self.c)

            _grad_total = 0.0
            for _key in _dldx:
                _grad_total = _grad_total + np.sum(_dldx[_key]**2)
            _grad_total = np.sqrt(_grad_total)

            self.c = alpha*self.c
            eq_constr_violation = np.max(np.abs(self.eq_constr(self.solution, opt_args)))
            ineq_constr_violation = np.max(self.ineq_constr(self.solution, opt_args))
            self.update_progress(k, _grad_total)
            if _grad_total < eps:
                print('done in ', k, ' iterations', _grad_total)
                return self.solution

        print('unsuccessful, tol: ', _grad_total)