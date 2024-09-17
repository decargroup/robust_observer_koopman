"""One sine lifting function."""

from typing import Optional, Tuple

import numpy as np
import pykoop


class OneSineLiftingFn(pykoop.koopman_pipeline.EpisodeIndependentLiftingFn):
    """Lifting function with one sine wave with phase offset.

    This class implements the ``EpisodeIndependentLiftingFn`` interface from
    ``pykoop``. Documentation for each implemented function can be found at
    https://pykoop.readthedocs.io/en/stable/_autosummary/pykoop.EpisodeIndependentLiftingFn.html
    """

    def __init__(self, f: float = 1, i: int = 0, phi: float = 0) -> None:
        """Instantiate :class:`SingleFreqLiftingFn`.

        Lifting function has the form::

            [x[0], x[1], x[2], sin(f * x[i] + phi)]

        Parameters
        ----------
        f : float
            Sinusoid frequency (rad/<unit of state>)
        i : int
            Index of state to put inside sinusoid.
        phi : float
            Sinusoid phase shift (rad).
        """
        self.f = f
        self.i = i
        self.phi = phi

    def _fit_one_ep(self, X: np.ndarray) -> Tuple[int, int]:
        return (self.n_states_in_ + 1, self.n_inputs_in_)

    def _transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        states = X[:, : self.n_states_in_]
        inputs = X[:, self.n_states_in_ :]
        features = [states]
        features.append(np.sin(self.f * states[:, [self.i]] + self.phi))
        features.append(inputs)
        Xt = np.hstack(features)
        return Xt

    def _inverse_transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        Xt = np.hstack(
            (
                X[:, : self.n_states_in_],
                X[:, self.n_states_in_ : self.n_states_in_ + self.n_inputs_in_],
            )
        )
        return Xt

    def _validate_parameters(self) -> None:
        if self.f <= 0:
            raise ValueError("Parameter `f` must be positive.")
        if self.i < 0:
            raise ValueError("Parameter `i` must be positive or zero.")

    def _transform_feature_names(
        self,
        feature_names: np.ndarray,
        format: Optional[str] = None,
    ) -> np.ndarray:
        names_out = []
        # Deal with episode feature
        if self.episode_feature_:
            names_in = feature_names[1:]
            names_out.append(feature_names[0])
        else:
            names_in = feature_names
        # Add states and inputs
        for ft in range(self.n_states_in_):
            names_out.append(names_in[ft])
        names_out.append(f"sin({self.f} * x_{self.i})")
        for ft in range(self.n_states_in_, self.n_states_in_ + self.n_inputs_in_):
            names_out.append(names_in[ft])
        feature_names_out = np.array(names_out, dtype=object)
        return feature_names_out
