import amigo as am
import numpy as np


class ExternalSMTComponent:
    def __init__(self, input_names, output_name, model, smt_model):
        """Evaluate the SMT model to compute outputs as a function of inputs

        inputs is of dimension (num_points, num_inputs) and the output of the model
        is always a scalar.

        Compute the constraints:

        for i in range(num_points):
            con[i] = SMT(inputs[i, :]) - outputs[i] = 0
        """

        self.input_indices = []
        for input_name in input_names:
            self.input_indices.append(model.get_indices(input_name))

        self.output_indices = model.get_indices(output_name)

        self.num_points = len(self.output_indices)
        self.num_inputs = len(self.input_indices)
        self.smt = smt_model

        # Set the number of variables and constraints
        self.nvars = self.num_points * (self.num_inputs + 1)
        self.ncon = self.num_points

        # Set the nonzero pattern for the constraint Jacobian
        self.rowp = np.arange(
            0,
            self.num_points * (self.num_inputs + 1) + 1,
            self.num_inputs + 1,
            dtype=int,
        )

        # Total number of inputs over all points
        nv = self.num_points * self.num_inputs

        self.cols = np.zeros(self.num_points * (self.num_inputs + 1), dtype=int)
        for i in range(self.num_inputs):
            self.cols[i :: self.num_inputs + 1] = np.arange(
                i * self.num_points, (i + 1) * self.num_points, dtype=int
            )
        self.cols[self.num_inputs :: self.num_inputs + 1] = np.arange(
            self.num_inputs * self.num_points, self.nvars, dtype=int
        )

        return

    def get_constraint_jacobian_csr(self):
        print(self.ncon, self.nvars, self.rowp, self.cols)
        return self.ncon, self.nvars, self.rowp, self.cols

    def evaluate(self, x, con, grad, jac):
        inputs = np.zeros((self.num_points, self.num_inputs))
        for i, index in enumerate(self.input_indices):
            inputs[:, i] = x[index]

        # Evaluate the interpolation constraint
        con[:] = self.smt.predict_values(inputs) - x[self.output_indices]

        # Compute the interpolation constraint Jacobian
        for i in range(self.num_inputs):
            jac[i :: self.num_inputs + 1] = self.smt.predict_derivatives(inputs, i)
        jac[self.num_inputs :: self.num_inputs + 1] = -1

        fobj = 0
        return fobj
