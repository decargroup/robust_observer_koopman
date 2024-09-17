"""Calculate the optimal transfer function to bound residual magnitudes.

Thanks to Jonathan Eid for providing the initial version of this function.
"""

import control
import numpy as np
import scipy.optimize


def tf_cover(
    omega: np.array,
    upper_bound: np.array,
    degree: int,
) -> control.TransferFunction:
    """Calculate the optimal upper bound transfer function of residuals.

    Form of W2 is prescribed to be
        W2(s) = a(s) / b(s),
    where a and b are a polynomial and monomial of given degree, respectively.

    WARNING: Perfomance is poor due to hard-coded intial guess.

    Parameters
    ----------
    omega : np.array
        Frequency domain over which W2 should bound the residuals.
    upper_bound : np.array
        Maximum magnitude of residuals over frequency domain. Absolute units,
        not decibels. Length of array must equal to that of array omega.
    degree : int
        Degree of numerator and denominator polynomials of biproper rational
        function that is W2.

    Returns
    -------
    W2 : control.TransferFunction
        Optimal upper bound transfer function of residuals.
    """

    # Error function.
    def _e(c: np.array) -> np.array:
        """Calculate the error over the frequency domain.

        The error at a particular frequency is defined as the difference between
        the maximum magnitude of residuals at that frequency and the magnitude
        of _W2 with parameters c evaluated at that frequency:
            error(w) = upper_bound(w) - |W2(c)(w)|.
        This function returns the error over each point in the frequency domain.

        Parameters
        ----------
        c : np.array
            Parameters of W2 for which the error is calculated.
            Form of c is
                [a_n, ..., a_0, b_n, ..., b_0].

        Returns
        -------
        e : np.array
            Error over the frequency domain.
        """
        num_W2 = np.polyval(c[: degree + 1], 1j * omega)
        den_W2 = np.polyval(np.insert(c[degree + 1 + 1 :], 0, 1.0), 1j * omega)
        W2 = num_W2 / den_W2
        mag_W2 = np.abs(W2)
        e = mag_W2 - upper_bound
        return e

    # Optimization objective function.
    def _J(c: np.array) -> np.double:
        """Calculate the optimization objective.

        The optimization objective is defined as the sum over each frequency
        point of the absolute error at that frequency point.

        Parameters
        ----------
        c : np.array
            Parameters of W2 for which the optimization objective is calculated.
            Form of c is
                [a_n, ..., a_0, b_n, ..., b_0].

        Returns
        -------
        J : np.double
            Optimization objective.
        """
        err = _e(c)
        J = np.sum(np.abs(err), dtype=np.double)
        return J

    # Initial guess at W2 is a constant transfer function with value equal to
    # peak of upper bound.
    c0 = np.zeros(2 * degree + 2)
    c0[degree] = upper_bound.max() + 1e-6
    c0[-1] = 1

    # Optimization problem and solution
    constraint = {"type": "ineq", "fun": _e}
    result = scipy.optimize.minimize(
        fun=_J,
        x0=list(c0),
        method="SLSQP",
        constraints=constraint,
        options={"maxiter": 100000},
    )

    c_opt = result.x

    # Replace real parts of poles and zeros with their absolute values to ensure
    # that W2 is asymptotically stable and minimum phase.
    num_c_opt = c_opt[: degree + 1]
    num_roots = np.roots(num_c_opt)
    new_num_roots = -np.abs(np.real(num_roots)) + 1j * np.imag(num_roots)

    # den_c_opt = c_opt[degree + 1:]
    den_c_opt = np.insert(c_opt[degree + 1 + 1 :], 0, 1.0)
    den_roots = np.roots(den_c_opt)
    new_den_roots = -np.abs(np.real(den_roots)) + 1j * np.imag(den_roots)

    gain = num_c_opt[0] / den_c_opt[0]

    # Form asymptotically stable, minimum phase optimal upper bound.
    W2_opt = control.zpk(new_num_roots, new_den_roots, gain)
    W2_opt_min_real = control.minreal(W2_opt, verbose=False)

    return W2_opt_min_real
