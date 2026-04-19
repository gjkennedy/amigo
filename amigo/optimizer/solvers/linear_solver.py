from abc import ABC, abstractmethod
import numpy as np


class LinearSolver(ABC):
    """
    The linear solver class is designed so that the KKT matrix is stored here,
    and the optimizer class does not have direct access to it. The interface defines
    the methods needed to evaluate, factor and solve the KKT system. Some solvers
    can compute the inertia of the KKT matrix. Those solvers should also implement
    get_inertia().
    """

    # @abstractmethod
    def eval_hessian(self, alpha, x):
        """
        Evaluate and save the Hessian at the given design point.

        This saves the Hessian matrix, but does not perform the factorization step.

        Args:
            alpha (float) : Scaling factor for the objective function
            x (am.Vector) : Vector of the design variables
        """
        pass

    @abstractmethod
    def factor(self, diag):
        """
        Factor the Hessian plus the diagonal matrix whose entries are in the diag vector.

        The factored matrix is K = (H + D)

        Args:
            diag (am.Vector) : Vector consisting of the diagonal elements

        Return:
            flag (int) : Index indicating success if flag = 0
        """
        pass

    @abstractmethod
    def solve(self, b, p):
        """
        Solve the system of equations K * p = b

        Args:
            b (am.Vector) : The right hand side vector
            p (am.Vector) : The solution vector
        """
        pass

    @abstractmethod
    def supports_inertia(self):
        """Does this class support inertia computation from a matrix factorization"""
        return False

    def get_inertia(self):
        raise NotImplementedError("This solver does not support inertia queries")

    def assemble_hessian(self, alpha, x):
        """Assemble Lagrangian Hessian and return its diagonal.

        Leaves the assembled matrix in self.hess for a subsequent
        add_diagonal_and_factor() call.  Cost: one Hessian evaluation +
        one device-to-host copy.  No factorization.
        """
        self.problem.hessian(alpha, x, self.hess)
        self.hess.copy_data_device_to_host()
        return self.hess.get_data()[self._diag_indices].copy()

    def get_hessian_diagonal(self, alpha, x):
        """Evaluate Hessian and return its diagonal. O(n), no factorization."""
        self.problem.hessian(alpha, x, self.hess)
        self.hess.copy_data_device_to_host()
        return self.hess.get_data()[self._diag_indices]

    @staticmethod
    def _find_diag_indices(rowp, cols, nrows):
        """Find the CSR data-array index of each diagonal entry (row == col)."""
        diag_idx = np.empty(nrows, dtype=np.intp)
        for i in range(nrows):
            start, end = rowp[i], rowp[i + 1]
            row_cols = cols[start:end]
            pos = np.searchsorted(row_cols, i)
            if pos < len(row_cols) and row_cols[pos] == i:
                diag_idx[i] = start + pos
            else:
                diag_idx[i] = start  # fallback (should not happen)
        return diag_idx
