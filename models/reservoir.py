import numpy as np
from typing import List, Optional, Union


class Reservoir(object):
    r"""
    Reservoir with grouped random mappings and temperature-dependent states.

    All groups share the active temperature-dependent or fixed update rate.
    Providing fixed_lambda replaces the temperature-dependent rate; an
    explicit temperature can still produce a state bias when Ea is nonzero.
    The vibration baseline uses fixed_lambda with Ea=0.0.
    Reserved parameters do not affect the numerical state calculation:
    spectral_radius, connectivity, circle, leak, multi_timescale,
    thermal_min, and thermal_max.
    """

    def __init__(
        self,
        n_internal_units: int = 100,
        spectral_radius: float = 0.99,
        leak: Union[float, List[float], None] = None,
        connectivity: float = 0.3,
        input_scaling: Union[float, List[float]] = 0.2,
        noise_level: float = 0.0,
        circle: bool = False,
        n_timescales: int = 3,
        multi_timescale: bool = True,
        Ea: float = 0.7,
        k_B: float = 8.617e-5,
        temperature: Union[float, np.ndarray] = 300.0,
        T_ref: Optional[float] = None,
        fixed_lambda: Optional[float] = None,
        thermal_min: Optional[float] = None,
        thermal_max: Optional[float] = None,
    ) -> None:
        self.multi_timescale = multi_timescale
        self._n_timescales = n_timescales
        self._n_internal_units = n_internal_units
        self._noise_level = noise_level

        self._Ea = float(Ea)
        self._k_B = float(k_B)
        self._temperature = temperature
        self._thermal_min = thermal_min
        self._thermal_max = thermal_max
        self._fixed_lambda = None if fixed_lambda is None else float(fixed_lambda)

        if self._fixed_lambda is not None and not (0.0 <= self._fixed_lambda <= 1.0):
            raise ValueError("fixed_lambda must be in [0, 1].")

        if np.isscalar(temperature):
            self._temp_mode = "scalar"
            self._temp_array = None
        else:
            temp_arr = np.asarray(temperature)
            if temp_arr.ndim == 1:
                self._temp_mode = "per_sample"
                self._temp_array = temp_arr
            elif temp_arr.ndim == 2:
                self._temp_mode = "per_timestep"
                self._temp_array = temp_arr
            else:
                raise ValueError(f"temperature ndim error: {temp_arr.ndim}")

        if T_ref is not None:
            self._T_ref = float(T_ref)
        else:
            if np.isscalar(temperature):
                self._T_ref = float(temperature)
            else:
                self._T_ref = float(np.mean(np.asarray(temperature)))

        print(f"[Reservoir-NoLimit] Init T_ref={self._T_ref:.2f} K.")
        if self._fixed_lambda is not None:
            print(
                f"[Reservoir-NoLimit] Using fixed lambda={self._fixed_lambda:.6f}; "
                "temperature dynamics disabled."
            )

        if self._n_timescales <= 0:
            raise ValueError("n_timescales error")
        self._sub_units = n_internal_units // self._n_timescales

        if leak is None:
            self._leak = [0.1] * self._n_timescales
        elif isinstance(leak, (float, int)):
            self._leak = [float(leak)] * self._n_timescales
        else:
            leak_list = list(leak)
            self._leak = [float(v) for v in leak_list] + [float(leak_list[-1])] * (
                self._n_timescales - len(leak_list)
            )

        if isinstance(input_scaling, (float, int)):
            self._input_scaling = [float(input_scaling)] * self._n_timescales
        else:
            self._input_scaling = [float(v) for v in list(input_scaling)]

        self._input_weights: List[np.ndarray] = []

    def _compute_state_matrix(
        self,
        X: np.ndarray,
        multi_timescale: bool = True,
        n_drop: int = 0,
        previous_state: Optional[np.ndarray] = None,
        current_temp: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        N, T_len, V = X.shape
        current_temp = self._resolve_temperature(X, current_temp)
        if (not isinstance(n_drop, (int, np.integer)) or
                isinstance(n_drop, (bool, np.bool_)) or not 0 <= n_drop < T_len):
            raise ValueError("n_drop must be an integer in [0, window_length).")

        if previous_state is None:
            state_prev_list = [
                np.zeros((N, self._sub_units), dtype=float)
                for _ in range(self._n_timescales)
            ]
        else:
            state_prev_list = []
            start_idx = 0
            for _ in range(self._n_timescales):
                end_idx = start_idx + self._sub_units
                if end_idx <= previous_state.shape[1]:
                    state_prev_list.append(previous_state[:, start_idx:end_idx])
                else:
                    state_prev_list.append(np.zeros((N, self._sub_units), dtype=float))
                start_idx = end_idx

        if isinstance(self._input_weights, list) and len(self._input_weights) == 0:
            for i in range(self._n_timescales):
                scaling = self._input_scaling[i]
                U = (
                    2.0 * np.random.binomial(1, 0.5, [self._sub_units, V]) - 1.0
                ) * scaling
                self._input_weights.append(U)

        state_matrix = np.empty(
            (N, T_len - n_drop, self._sub_units * self._n_timescales), dtype=float
        )

        for t in range(T_len):
            current_input = X[:, t, :]

            if self._fixed_lambda is not None:
                lambda_t = np.full(N, self._fixed_lambda, dtype=float)
            else:
                if current_temp.ndim == 2:
                    temp_now = current_temp[:, t]
                else:
                    temp_now = current_temp

                temp_safe = np.maximum(temp_now, 1e-9)
                tau_dynamic = 0.0048 * np.exp(3242.0 / temp_safe)
                lambda_t = 10.0 / (tau_dynamic + 10.0)

            lambda_expanded = lambda_t[:, np.newaxis]

            state_all_scales = []
            for i in range(self._n_timescales):
                U = self._input_weights[i]
                input_current = np.dot(current_input, U.T)
                activation_term = 1.0 + np.tanh(input_current)

                if self._noise_level > 0:
                    activation_term += np.random.normal(
                        0.0, self._noise_level, activation_term.shape
                    )

                state_memory = (1.0 - lambda_expanded) * state_prev_list[i]
                state_input = lambda_expanded * activation_term
                state_t = state_memory + state_input
                state_t = np.nan_to_num(state_t, nan=0.0)
                state_prev_list[i] = state_t
                state_all_scales.append(state_t)

            state_combined = np.hstack(state_all_scales)

            if t >= n_drop:
                state_matrix[:, t - n_drop, :] = state_combined

        if current_temp is not None and self._Ea != 0.0:
            T_full = current_temp
            if T_full.ndim == 1:
                T_full = T_full[:, np.newaxis]
            elif n_drop > 0:
                T_full = T_full[:, n_drop:]

            T_safe_full = np.maximum(T_full, 1e-9)
            delta = self._Ea / self._k_B * (1.0 / self._T_ref - 1.0 / T_safe_full)
            sigma_raw = np.exp(delta)
            phys_bias_log = np.log(sigma_raw + 1e-9)

            if phys_bias_log.ndim == 2:
                phys_bias_log = phys_bias_log[:, :, np.newaxis]
            elif phys_bias_log.ndim == 1:
                phys_bias_log = phys_bias_log[:, np.newaxis, np.newaxis]

            return state_matrix + phys_bias_log

        return state_matrix

    def _resolve_temperature(self, X, current_temp):
        """Resolve Kelvin temperatures, giving explicit values priority."""
        # A fixed-rate call without an explicit temperature has no thermal bias.
        if current_temp is None:
            if self._fixed_lambda is not None:
                return None
            current_temp = self._temperature
        temp = np.asarray(current_temp, dtype=float)
        n, steps = X.shape[:2]
        if temp.ndim == 0:
            temp = np.full(n, float(temp))
        elif temp.shape == (n, 1):
            temp = np.broadcast_to(temp, (n, steps))
        elif temp.shape not in ((n,), (n, steps)):
            raise ValueError("Temperature must be scalar, (N,), (N, 1), or (N, window_length).")
        if not np.isfinite(temp).all() or np.any(temp <= 0):
            raise ValueError("Temperature must be finite and positive, in Kelvin.")
        return temp

    def get_states(
        self,
        X: np.ndarray,
        multi_timescale: bool = True,
        n_drop: int = 0,
        bidir: bool = True,
        initial_state: Optional[np.ndarray] = None,
        current_temp: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        current_temp = self._resolve_temperature(X, current_temp)
        state_fwd = self._compute_state_matrix(
            X,
            multi_timescale=multi_timescale,
            n_drop=n_drop,
            previous_state=initial_state,
            current_temp=current_temp,
        )

        if bidir:
            X_r = X[:, ::-1, :]
            temp_r = None if current_temp is None else (
                current_temp[:, ::-1] if current_temp.ndim == 2 else current_temp
            )

            state_bwd = self._compute_state_matrix(
                X_r,
                multi_timescale=multi_timescale,
                n_drop=n_drop,
                previous_state=initial_state,
                current_temp=temp_r,
            )
            return np.concatenate((state_fwd, state_bwd), axis=2)

        return state_fwd
