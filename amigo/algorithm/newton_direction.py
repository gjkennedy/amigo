"""Gradient evaluation, KKT factorization, and Newton solve.

Evaluates the gradient, assembles and factorizes the KKT matrix
(delegating inertia correction to InertiaCorrector when the solver
supports it), solves the condensed augmented system, asks the solver
to iteratively refine the result, and back-substitutes for the bound
duals.  The iterative-refinement kernel itself lives on the
LinearSolver base class.
"""

import numpy as np


class NewtonStep:
    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

        self.update = self.problem.create_vector()

    def compute_step(self, solver, evaluator, state):
        # Evalute the residual (may be required since the barrier may have changed)
        evaluator.evaluate_residual(state)

        # Solve the linear system to obtain the new step
        solver.solve(state.residual, self.update)

        # Compute the full step
        self.optimizer.compute_update(state.mu, state.current, self.update, state.step)

        # Now, compute the maximum step lengths in the primal and dual directions
        alpha_x, _, alpha_z, _ = self.optimizer.compute_max_step(
            state.tau, state.current, state.step
        )

        state.max_alpha_primal = alpha_x
        state.max_alpha_dual = alpha_z

        # Indicate that the step has been updated
        state.step_current = True
