"""Amigo's native LDL direct solver for the KKT system.

Wraps the C++ SparseLDL factorization. Supports inertia queries,
which lets it drive Algorithm IC inertia correction.
"""

from . import LinearSolver
from amigo import SolverType, SparseLDL, OrderingType


class AmigoSolver(LinearSolver):
    def __init__(self, options, state):
        self.options = options
        self.hessian = state.hessian

        ustab = 0.01
        pivot_tol = 1e-14

        self.ldl = SparseLDL(
            self.hessian,
            SolverType.LDL,
            ustab=ustab,
            pivot_tol=pivot_tol,
            order=OrderingType.DEFAULT,
        )

    def factor(self, hessian, diagonal):
        if hessian != self.hessian:
            raise ValueError("Hessian instance must be the same")

        # Copy the Hessian and diagonal entries to the host
        diagonal.copy_device_to_host()
        self.hessian.copy_data_device_to_host()

        flag = self.ldl.factor(diagonal)
        if flag != 0:
            raise RuntimeError(
                f"{self.solver_name} factorization failed with flag = {flag}"
            )

        return flag

    def solve(self, b, x):
        # Copy b to x
        x.copy(b)

        # Copy the components to the host
        x.copy_device_to_host()

        # Solve the problem on the host
        self.ldl.solve(x)

        # Copy the solution back to the device
        x.copy_host_to_device()
        return

    def inertia_enabled(self):
        return True

    def get_inertia(self):
        return self.ldl.get_inertia()

    def set_pivot_tolerance(self, pivtol):
        pass
