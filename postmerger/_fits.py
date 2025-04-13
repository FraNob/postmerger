from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin

import numpy as np
import joblib
from . import qnm_Kerr, final_mass, final_spin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import os

dir_path = os.path.dirname(os.path.realpath(__file__))


allowed_fits = ["3dq8_20M", "Prec6dq10_20M", "Prec7dq10_20M"]

fit_descr = {
    "3dq8_20M": "3dq8_20M models amplitudes and phases of the ringdown from a quasi-circular, non-precessing black-hole binary.\nIt is calibrated up to mass ratio 8 and at a starting time 20M from the peak of the (2,2) strain.",
    "Prec6dq10_20M": "Prec6dq10_20M models amplitudes of the ringdown from a quasi-circular, precessing black-hole binary.\nIt is calibrated up to mass ratio 10 and at a starting time 20M from t_emop of the simulation.\nIt also provides an uncertainty estimate calibrated on the cross-validation absolute error on the amplitude values.",
    "Prec7dq10_20M": "Prec7dq10_20M models amplitudes of the ringdown from a quasi-circular, precessing black-hole binary.\nIt is calibrated up to mass ratio 10 and at a starting time 20M from t_emop of the simulation.\nIt also provides an uncertainty estimate calibrated on the cross-validation absolute error on the amplitude values.",
}


def download_if_missing(filename: str, folder: str):
    """
    Check if a file exists in `folder` with name `filename`.
    If not, download it from the given URL.
    """
    file_path = os.path.join(folder, filename)
    url = f"https://zenodo.org/record/13286798/files/{filename}?download=1"

    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return
    else:
        print(f"File not found locally. Downloading from {url}...")

    # Requests session with retry logic for robustness.
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        with session.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()  # Raise an error for bad status codes
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
        print(f"Download complete: {file_path}")
    except requests.RequestException as e:
        print(f"Error occurred during download: {e}")
        # Optionally remove the partially downloaded file
        if os.path.exists(file_path):
            os.remove(file_path)
        raise


def load_fit(name, download=False):
    """
    Load surrogate model.

    Parameters
    ----------
    name : str
        Name of the model. Bust be one of allowed_fits.
    """
    if name not in allowed_fits:
        raise ValueError("name must be one of " + str(allowed_fits))
    trained_models_path = dir_path + "/data/trained_models/"
    if not os.path.exists(trained_models_path):
        os.makedirs(trained_models_path)
    filename = f"{name}_gpr.pkl"

    if download:
        download_if_missing(filename, trained_models_path)

    fit_dict = joblib.load()
    if "3dq8" in name:
        model = AmplitudeFit3dq8(fit_dict)
        model._descr = fit_descr[name]
    elif "Prec6dq10" in name:
        model = AmplitudeFitPrec6dq10(fit_dict)
        model._descr = fit_descr[name]
    elif "Prec7dq10" in name:
        model = AmplitudeFitPrec7dq10(fit_dict)
        model._descr = fit_descr[name]
    else:
        raise ValueError("name must be one of " + str(allowed_fits))
    return model


def _shift_amp_general(
    self, amp, final_mass_val, final_spin_val, lm, mode, start_time, qnm_method="interp"
):
    """
    Shift the amplitude of the ringdown waveform to a different time.
    Note that start_time depends on the model used.
    For spin-aligned models, start_time is the time from the peak of |h_{22}|. Default is 20.
    For 6dq10 models, start_time is the time from t_emop. Default is 20."""

    if start_time == self.t0:
        return amp
    DT = start_time - self.t0

    mf = final_mass_val
    sf = final_spin_val
    if hasattr(mode[0], "__len__"):
        ## handle quadratic mode
        inv_tau = 0.0
        for linear_mode in mode:
            inv_tau += (
                1.0
                / qnm_Kerr(mf, sf, linear_mode, qnm_method=qnm_method, SI_units=False)[
                    1
                ]
            )
    else:
        ## handle linear mode
        inv_tau = 1.0 / qnm_Kerr(mf, sf, mode, qnm_method=qnm_method, SI_units=False)[1]
    if lm != (2, 2) or mode != (2, 2, 0):
        inv_tau -= (
            1.0 / qnm_Kerr(mf, sf, (2, 2, 0), qnm_method=qnm_method, SI_units=False)[1]
        )
    out = amp * np.exp(-DT * inv_tau)
    return out


def _make_stacked_array(*args):
    # Convert all inputs to at least 1D arrays
    arrays = [np.atleast_1d(arg) for arg in args]

    # Determine target length
    lengths = [arr.shape[0] for arr in arrays]
    target_len = max(lengths)

    # Broadcast scalars or shorter arrays to target length
    broadcasted = []
    for arr in arrays:
        if arr.shape[0] == 1 and target_len > 1:
            arr = np.full(target_len, arr.item())
        elif arr.shape[0] != target_len:
            raise ValueError(
                f"Incompatible shapes: expected {target_len}, got {arr.shape[0]}"
            )
        broadcasted.append(arr)

    return np.vstack(broadcasted).T


class AmplitudeFitPrec6dq10:
    def __init__(self, fit_dict):
        self._fit_amps = fit_dict["amps"]
        self._fit_abs_err = fit_dict["abs_err"]
        self._fit_t_emop = fit_dict["t_emop"]
        self._fit_t_emop_abs_err = fit_dict["t_emop_abs_err"]
        self.t0 = fit_dict["time_from_temop"]
        self.feature_set = fit_dict["feature_set"]

        ## deduce modes
        self.modes = {}
        for k in self._fit_amps.keys():
            if k[1] >= 0:
                self.modes[k] = [mode for mode in self._fit_amps[k].keys()]

    def __print__(self):
        out = self._descr
        return out

    def __repr__(self):
        out = self._descr
        return out

    def predict_amp(
        self,
        delta,
        chi_s,
        chi_a,
        rem_spin_angle,
        kick_angle,
        kick_vel,
        lm,
        mode,
        return_std=False,
        start_time=None,
        final_mass_val=None,
        final_spin_val=None,
    ):
        """
        Predict the values of A_lm corresponding to the query points.
        Mode (2,0) is not circularly polarized. lm=(2,0) corresponds to the real part, while lm=(2,10) corresponds to the imaginary part.

        Parameters
        ----------
        delta : array_like of shape (n_samples,) or float
            Asymmetric mass ratio of the query points. delta = (q-1)/(q+1) where q is the mass ratio. q = m1/m2 >= 1.

        chi_s : array_like of shape (n_samples,) or float
            Symmetric combination of spin components parallel to orbital angular momentum at ISCO.
            chi_s = (q*chi1z + chi2z)/(q+1) where q = m1/m2 >= 1.

        chi_a : array_like of shape (n_samples,) or float
            Antisymmetric combination of spin components parallel to orbital angular momentum at ISCO.
            chi_a = (q*chi1z - chi2z)/(q+1) where q = m1/m2 >= 1.

        rem_spin_angle : array_like of shape (n_samples,) or float
            Angle between the remnant spin and the orbital angular momentum at ISCO.

        kick_angle : array_like of shape (n_samples,) or float
            Angle between the remnant spin and the recoil kick velocity.

        kick_vel : array_like of shape (n_samples,) or float
            Magnitude of the recoil kick velocity.

        lm : tuple_like object
            Ordered couple (l,m) specifying the angular number l and azimuthal number m of the harmonic.

        mode : tuple_like object
            Ordered tuple specifying the queried quasi-normal mode.
            For linear modes, ordered triple (l,m,n).
            For quadratic modes, ordered couple of the form ((l1,m1,n1),(l2,m2,n2)).

        return_std : bool. Default=False.
            Whether or not to return the standard deviation of the predictive distribution at the query points.
            **WARNING**: As explained in the paper, this is not a good measure of uncertainty for the surrogate model. Use the predict_abs_err instead.

        start_time : float or None. Default=None.
            Start time of the ringdown waveform, relative to t_emop of the simulation.
            If None (default), assume that start_time=self.t0.

        final_mass_val : float or None. Default=None.
            Final mass of the remnant black hole. Only used if start_time is not None for retrodiction.

        final_spin_val : float or None. Default=None.
            Final spin of the remnant black hole. Only used if start_time is not None for retrodiction.

        """
        X = _make_stacked_array(
            delta,
            chi_s,
            chi_a,
            rem_spin_angle,
            kick_angle,
            kick_vel,
        )

        if start_time is not None:
            assert (
                final_mass_val is not None
            ), "final_mass_val must be provided if start_time is not None"
            assert (
                final_spin_val is not None
            ), "final_spin_val must be provided if start_time is not None"

        if return_std:
            amp_abs, std_abs = self._fit_amps[lm][mode].predict(
                X, return_std=return_std
            )
            if start_time is not None:
                amp_abs = self._shift_amp(
                    amp_abs,
                    final_mass_val,
                    final_spin_val,
                    lm,
                    mode,
                    start_time,
                )
                std_abs = self._shift_amp(
                    std_abs,
                    final_mass_val,
                    final_spin_val,
                    lm,
                    mode,
                    start_time,
                )
            return amp_abs, std_abs

        else:
            amp_abs = self._fit_amps[lm][mode].predict(X, return_std=return_std)
            if start_time is not None:
                amp_abs = self._shift_amp(
                    amp_abs,
                    final_mass_val,
                    final_spin_val,
                    lm,
                    mode,
                    start_time,
                )
            return amp_abs

    def predict_abs_err(
        self,
        delta,
        chi_s,
        chi_a,
        rem_spin_angle,
        kick_angle,
        kick_vel,
        lm,
        mode,
        return_std=False,
        start_time=None,
        final_mass_val=None,
        final_spin_val=None,
    ):
        """
        Predict the cross-validation absolute error on A_lm corresponding to the query points.
        Mode (2,0) is not circularly polarized. lm=(2,0) corresponds to the real part, while lm=(2,10) corresponds to the imaginary part.

        Parameters
        ----------
        mass_ratio : array_like of shape (n_samples,) or float
            Mass ratio of the query points.

        chi1z : array_like of shape (n_samples,) or float
            Projection along the z axis of the primary spin.

        chi2z : array_like of shape (n_samples,) or float
            Projection along the z axis of the secondary spin.

        lm : tuple_like object
            Ordered couple (l,m) specifying the angular number l and azimuthal number m of the harmonic.

        mode : tuple_like object
            Ordered tuple specifying the queried quasi-normal mode.
            For linear modes, ordered triple (l,m,n).
            For quadratic modes, ordered couple of the form ((l1,m1,n1),(l2,m2,n2)).

        return_std : bool. Default=False.
            Whether or not to return the standard deviation of the predictive distribution at the query points.

        start_time : float or None. Default=None.
            Start time of the ringdown waveform, relative to t_emop of the simulation.
            If None (default), assume that start_time=self.t0.

        final_mass_val : float or None. Default=None.
            Final mass of the remnant black hole. Only used if start_time is not None for retrodiction.

        final_spin_val : float or None. Default=None.
            Final spin of the remnant black hole. Only used if start_time is not None for retrodiction.

        """
        X = _make_stacked_array(
            delta,
            chi_s,
            chi_a,
            rem_spin_angle,
            kick_angle,
            kick_vel,
        )

        if start_time is not None:
            assert (
                final_mass_val is not None
            ), "final_mass_val must be provided if start_time is not None"
            assert (
                final_spin_val is not None
            ), "final_spin_val must be provided if start_time is not None"

        if return_std:
            abs_err, abs_err_std = self._fit_abs_err[lm][mode].predict(
                X, return_std=return_std
            )
            if start_time is not None:
                abs_err = self._shift_amp(
                    abs_err,
                    final_mass_val,
                    final_spin_val,
                    lm,
                    mode,
                    start_time,
                )
                abs_err_std = self._shift_amp(
                    abs_err_std,
                    final_mass_val,
                    final_spin_val,
                    lm,
                    mode,
                    start_time,
                )
            return abs_err, abs_err_std

        else:
            abs_err = self._fit_abs_err[lm][mode].predict(X, return_std=return_std)
            if start_time is not None:
                abs_err = self._shift_amp(
                    abs_err,
                    final_mass_val,
                    final_spin_val,
                    lm,
                    mode,
                    start_time,
                )
            return abs_err

    def sample_amp(
        self,
        delta,
        chi_s,
        chi_a,
        rem_spin_angle,
        kick_angle,
        kick_vel,
        lm,
        mode,
        n_samples=1,
        start_time=None,
        final_mass_val=None,
        final_spin_val=None,
    ):
        """
        Sample the values of A_lm corresponding to the query points.
        Mode (2,0) is not circularly polarized. lm=(2,0) corresponds to the real part, while lm=(2,10) corresponds to the imaginary part.

        Parameters
        ----------
        delta : array_like of shape (n_samples,) or float
            Asymmetric mass ratio of the query points. delta = (q-1)/(q+1) where q is the mass ratio. q = m1/m2 >= 1.

        chi_s : array_like of shape (n_samples,) or float
            Symmetric combination of spin components parallel to orbital angular momentum at ISCO.
            chi_s = (q*chi1z + chi2z)/(q+1) where q = m1/m2 >= 1.

        chi_a : array_like of shape (n_samples,) or float
            Antisymmetric combination of spin components parallel to orbital angular momentum at ISCO.
            chi_a = (q*chi1z - chi2z)/(q+1) where q = m1/m2 >= 1.

        rem_spin_angle : array_like of shape (n_samples,) or float
            Angle between the remnant spin and the orbital angular momentum at ISCO.

        kick_angle : array_like of shape (n_samples,) or float
            Angle between the remnant spin and the recoil kick velocity.

        kick_vel : array_like of shape (n_samples,) or float
            Magnitude of the recoil kick velocity.

        lm : tuple_like object
            Ordered couple (l,m) specifying the angular number l and azimuthal number m of the harmonic.

        mode : tuple_like object
            Ordered tuple specifying the queried quasi-normal mode.
            For linear modes, ordered triple (l,m,n).
            For quadratic modes, ordered couple of the form ((l1,m1,n1),(l2,m2,n2)).

        return_std : bool. Default=False.
            Whether or not to return the standard deviation of the predictive distribution at the query points.
            **WARNING**: As explained in the paper, this is not a good measure of uncertainty for the surrogate model. Use the predict_abs_err instead.

        start_time : float or None. Default=None.
            Start time of the ringdown waveform, relative to t_emop of the simulation.
            If None (default), assume that start_time=self.t0.

        final_mass_val : float or None. Default=None.
            Final mass of the remnant black hole. Only used if start_time is not None for retrodiction.

        final_spin_val : float or None. Default=None.
            Final spin of the remnant black hole. Only used if start_time is not None for retrodiction.

        """
        X = _make_stacked_array(
            delta,
            chi_s,
            chi_a,
            rem_spin_angle,
            kick_angle,
            kick_vel,
        )

        if start_time is not None:
            assert (
                final_mass_val is not None
            ), "final_mass_val must be provided if start_time is not None"
            assert (
                final_spin_val is not None
            ), "final_spin_val must be provided if start_time is not None"

        amp_abs = self._fit_amps[lm][mode].sample_y(X, n_samples=n_samples)
        if start_time is not None:
            amp_abs = self._shift_amp(
                amp_abs,
                final_mass_val,
                final_spin_val,
                lm,
                mode,
                start_time,
            )
        return amp_abs

    def predict_t_emop(
        self,
        delta,
        chi_s,
        chi_a,
        rem_spin_angle,
        kick_angle,
        kick_vel,
        return_std=False,
    ):
        """
        Predict the value of t_emop with respect to the peak of the L2 norm of the waveform, corresponding to the query points.
        Mode (2,0) is not circularly polarized. lm=(2,0) corresponds to the real part, while lm=(2,10) corresponds to the imaginary part.

        Parameters
        ----------
        delta : array_like of shape (n_samples,) or float
            Asymmetric mass ratio of the query points. delta = (q-1)/(q+1) where q is the mass ratio. q = m1/m2 >= 1.

        chi_s : array_like of shape (n_samples,) or float
            Symmetric combination of spin components parallel to orbital angular momentum at ISCO.
            chi_s = (q*chi1z + chi2z)/(q+1) where q = m1/m2 >= 1.

        chi_a : array_like of shape (n_samples,) or float
            Antisymmetric combination of spin components parallel to orbital angular momentum at ISCO.
            chi_a = (q*chi1z - chi2z)/(q+1) where q = m1/m2 >= 1.

        rem_spin_angle : array_like of shape (n_samples,) or float
            Angle between the remnant spin and the orbital angular momentum at ISCO.

        kick_angle : array_like of shape (n_samples,) or float
            Angle between the remnant spin and the recoil kick velocity.

        kick_vel : array_like of shape (n_samples,) or float
            Magnitude of the recoil kick velocity.

        return_std : bool. Default=False.
            Whether or not to return the standard deviation of the predictive distribution at the query points.

        """
        X = _make_stacked_array(
            delta,
            chi_s,
            chi_a,
            rem_spin_angle,
            kick_angle,
            kick_vel,
        )

        if return_std:
            t_emop, t_emop_std = self._fit_t_emop.predict(X, return_std=return_std)
            return t_emop, t_emop_std

        else:
            t_emop = self._fit_t_emop.predict(X, return_std=return_std)
            return t_emop

    def predict_t_emop_abs_err(
        self,
        delta,
        chi_s,
        chi_a,
        rem_spin_angle,
        kick_angle,
        kick_vel,
        return_std=False,
    ):
        """
        Predict the value of t_emop with respect to the peak of the L2 norm of the waveform, corresponding to the query points.
        Mode (2,0) is not circularly polarized. lm=(2,0) corresponds to the real part, while lm=(2,10) corresponds to the imaginary part.

        Parameters
        ----------
        delta : array_like of shape (n_samples,) or float
            Asymmetric mass ratio of the query points. delta = (q-1)/(q+1) where q is the mass ratio. q = m1/m2 >= 1.

        chi_s : array_like of shape (n_samples,) or float
            Symmetric combination of spin components parallel to orbital angular momentum at ISCO.
            chi_s = (q*chi1z + chi2z)/(q+1) where q = m1/m2 >= 1.

        chi_a : array_like of shape (n_samples,) or float
            Antisymmetric combination of spin components parallel to orbital angular momentum at ISCO.
            chi_a = (q*chi1z - chi2z)/(q+1) where q = m1/m2 >= 1.

        rem_spin_angle : array_like of shape (n_samples,) or float
            Angle between the remnant spin and the orbital angular momentum at ISCO.

        kick_angle : array_like of shape (n_samples,) or float
            Angle between the remnant spin and the recoil kick velocity.

        kick_vel : array_like of shape (n_samples,) or float
            Magnitude of the recoil kick velocity.

        return_std : bool. Default=False.
            Whether or not to return the standard deviation of the predictive distribution at the query points.

        """
        X = _make_stacked_array(
            delta,
            chi_s,
            chi_a,
            rem_spin_angle,
            kick_angle,
            kick_vel,
        )

        if return_std:
            t_emop_abs_err, t_emop_abs_err_std = self._fit_t_emop_abs_err.predict(
                X, return_std=return_std
            )
            return t_emop_abs_err, t_emop_abs_err_std

        else:
            t_emop_abs_err = self._fit_t_emop_abs_err.predict(X, return_std=return_std)
            return t_emop_abs_err

    def sample_t_emop(
        self,
        delta,
        chi_s,
        chi_a,
        rem_spin_angle,
        kick_angle,
        kick_vel,
        n_samples=1,
    ):
        """
        Sample the value of t_emop with respect to the peak of the L2 norm of the waveform, corresponding to the query points.
        Mode (2,0) is not circularly polarized. lm=(2,0) corresponds to the real part, while lm=(2,10) corresponds to the imaginary part.

        Parameters
        ----------
        delta : array_like of shape (n_samples,) or float
            Asymmetric mass ratio of the query points. delta = (q-1)/(q+1) where q is the mass ratio. q = m1/m2 >= 1.

        chi_s : array_like of shape (n_samples,) or float
            Symmetric combination of spin components parallel to orbital angular momentum at ISCO.
            chi_s = (q*chi1z + chi2z)/(q+1) where q = m1/m2 >= 1.

        chi_a : array_like of shape (n_samples,) or float
            Antisymmetric combination of spin components parallel to orbital angular momentum at ISCO.
            chi_a = (q*chi1z - chi2z)/(q+1) where q = m1/m2 >= 1.

        rem_spin_angle : array_like of shape (n_samples,) or float
            Angle between the remnant spin and the orbital angular momentum at ISCO.

        kick_angle : array_like of shape (n_samples,) or float
            Angle between the remnant spin and the recoil kick velocity.

        kick_vel : array_like of shape (n_samples,) or float
            Magnitude of the recoil kick velocity.

        n_samples : int
            Number of samples to draw from the predictive distribution.
        """
        X = _make_stacked_array(
            delta,
            chi_s,
            chi_a,
            rem_spin_angle,
            kick_angle,
            kick_vel,
        )

        t_emop = self._fit_t_emop.sample_y(X, n_samples=n_samples)
        return t_emop

    def _shift_amp(
        self,
        amp,
        final_mass_val,
        final_spin_val,
        lm,
        mode,
        start_time,
        qnm_method="interp",
    ):
        return _shift_amp_general(
            self, amp, final_mass_val, final_spin_val, lm, mode, start_time, qnm_method
        )


class AmplitudeFitPrec7dq10:
    def __init__(self, fit_dict):
        self._fit_amps = fit_dict["amps"]
        self._fit_abs_err = fit_dict["abs_err"]
        self._fit_t_emop = fit_dict["t_emop"]
        self._fit_t_emop_abs_err = fit_dict["t_emop_abs_err"]
        self.t0 = fit_dict["time_from_temop"]
        self.feature_set = fit_dict["feature_set"]

        ## deduce modes
        self.modes = {}
        for k in self._fit_amps.keys():
            if k[1] >= 0:
                self.modes[k] = [mode for mode in self._fit_amps[k].keys()]

    def __print__(self):
        out = self._descr
        return out

    def __repr__(self):
        out = self._descr
        return out

    def predict_amp(
        self,
        delta,
        chi1_x,
        chi1_y,
        chi1_z,
        chi2_x,
        chi2_y,
        chi2_z,
        lm,
        mode,
        return_std=False,
        start_time=None,
        final_mass_val=None,
        final_spin_val=None,
    ):
        """
        Predict the values of A_lm corresponding to the query points.
        Mode (2,0) is not circularly polarized. lm=(2,0) corresponds to the real part, while lm=(2,10) corresponds to the imaginary part.

        Parameters
        ----------
        delta : array_like of shape (n_samples,) or float
            Asymmetric mass ratio of the query points. delta = (q-1)/(q+1) where q is the mass ratio. q = m1/m2 >= 1.

        chi1_x : array_like of shape (n_samples,) or float
            x-component of the primary spin at ISCO.

        chi1_y : array_like of shape (n_samples,) or float
            y-component of the primary spin at ISCO.

        chi1_z : array_like of shape (n_samples,) or float
            z-component of the primary spin at ISCO.

        chi2_x : array_like of shape (n_samples,) or float
            x-component of the secondary spin at ISCO.

        chi2_y : array_like of shape (n_samples,) or float
            y-component of the secondary spin at ISCO.

        chi2_z : array_like of shape (n_samples,) or float
            z-component of the secondary spin at ISCO.

        lm : tuple_like object
            Ordered couple (l,m) specifying the angular number l and azimuthal number m of the harmonic.

        mode : tuple_like object
            Ordered tuple specifying the queried quasi-normal mode.
            For linear modes, ordered triple (l,m,n).
            For quadratic modes, ordered couple of the form ((l1,m1,n1),(l2,m2,n2)).

        return_std : bool. Default=False.
            Whether or not to return the standard deviation of the predictive distribution at the query points.
            **WARNING**: As explained in the paper, this is not a good measure of uncertainty for the surrogate model. Use the predict_abs_err instead.

        start_time : float or None. Default=None.
            Start time of the ringdown waveform, relative to t_emop of the simulation.
            If None (default), assume that start_time=self.t0.

        final_mass_val : float or None. Default=None.
            Final mass of the remnant black hole. Only used if start_time is not None for retrodiction.

        final_spin_val : float or None. Default=None.
            Final spin of the remnant black hole. Only used if start_time is not None for retrodiction.

        """

        X = _make_stacked_array(
            delta,
            chi1_x,
            chi1_y,
            chi1_z,
            chi2_x,
            chi2_y,
            chi2_z,
        )

        if start_time is not None:
            assert (
                final_mass_val is not None
            ), "final_mass_val must be provided if start_time is not None"
            assert (
                final_spin_val is not None
            ), "final_spin_val must be provided if start_time is not None"

        if return_std:
            amp_abs, std_abs = self._fit_amps[lm][mode].predict(
                X, return_std=return_std
            )
            if start_time is not None:
                amp_abs = self._shift_amp(
                    amp_abs,
                    final_mass_val,
                    final_spin_val,
                    lm,
                    mode,
                    start_time,
                )
                std_abs = self._shift_amp(
                    std_abs,
                    final_mass_val,
                    final_spin_val,
                    lm,
                    mode,
                    start_time,
                )
            return amp_abs, std_abs

        else:
            amp_abs = self._fit_amps[lm][mode].predict(X, return_std=return_std)
            if start_time is not None:
                amp_abs = self._shift_amp(
                    amp_abs,
                    final_mass_val,
                    final_spin_val,
                    lm,
                    mode,
                    start_time,
                )
            return amp_abs

    def predict_abs_err(
        self,
        delta,
        chi1_x,
        chi1_y,
        chi1_z,
        chi2_x,
        chi2_y,
        chi2_z,
        lm,
        mode,
        return_std=False,
        start_time=None,
        final_mass_val=None,
        final_spin_val=None,
    ):
        """
        Predict the values of the cross-validation absolute error on A_lm corresponding to the query points.
        Mode (2,0) is not circularly polarized. lm=(2,0) corresponds to the real part, while lm=(2,10) corresponds to the imaginary part.

        Parameters
        ----------
        delta : array_like of shape (n_samples,) or float
            Asymmetric mass ratio of the query points. delta = (q-1)/(q+1) where q is the mass ratio. q = m1/m2 >= 1.

        chi1_x : array_like of shape (n_samples,) or float
            x-component of the primary spin at ISCO.

        chi1_y : array_like of shape (n_samples,) or float
            y-component of the primary spin at ISCO.

        chi1_z : array_like of shape (n_samples,) or float
            z-component of the primary spin at ISCO.

        chi2_x : array_like of shape (n_samples,) or float
            x-component of the secondary spin at ISCO.

        chi2_y : array_like of shape (n_samples,) or float
            y-component of the secondary spin at ISCO.

        chi2_z : array_like of shape (n_samples,) or float
            z-component of the secondary spin at ISCO.

        lm : tuple_like object
            Ordered couple (l,m) specifying the angular number l and azimuthal number m of the harmonic.

        mode : tuple_like object
            Ordered tuple specifying the queried quasi-normal mode.
            For linear modes, ordered triple (l,m,n).
            For quadratic modes, ordered couple of the form ((l1,m1,n1),(l2,m2,n2)).

        return_std : bool. Default=False.
            Whether or not to return the standard deviation of the predictive distribution at the query points.

        start_time : float or None. Default=None.
            Start time of the ringdown waveform, relative to t_emop of the simulation.
            If None (default), assume that start_time=self.t0.

        final_mass_val : float or None. Default=None.
            Final mass of the remnant black hole. Only used if start_time is not None for retrodiction.

        final_spin_val : float or None. Default=None.
            Final spin of the remnant black hole. Only used if start_time is not None for retrodiction.

        """

        X = _make_stacked_array(
            delta,
            chi1_x,
            chi1_y,
            chi1_z,
            chi2_x,
            chi2_y,
            chi2_z,
        )

        if start_time is not None:
            assert (
                final_mass_val is not None
            ), "final_mass_val must be provided if start_time is not None"
            assert (
                final_spin_val is not None
            ), "final_spin_val must be provided if start_time is not None"

        if return_std:
            abs_err, abs_err_std = self._fit_abs_err[lm][mode].predict(
                X, return_std=return_std
            )
            if start_time is not None:
                abs_err = self._shift_amp(
                    abs_err,
                    final_mass_val,
                    final_spin_val,
                    lm,
                    mode,
                    start_time,
                )
                abs_err_std = self._shift_amp(
                    abs_err_std,
                    final_mass_val,
                    final_spin_val,
                    lm,
                    mode,
                    start_time,
                )
            return abs_err, abs_err_std

        else:
            abs_err = self._fit_abs_err[lm][mode].predict(X, return_std=return_std)
            if start_time is not None:
                abs_err = self._shift_amp(
                    abs_err,
                    final_mass_val,
                    final_spin_val,
                    lm,
                    mode,
                    start_time,
                )
            return abs_err

    def sample_amp(
        self,
        delta,
        chi1_x,
        chi1_y,
        chi1_z,
        chi2_x,
        chi2_y,
        chi2_z,
        lm,
        mode,
        n_samples=1,
        start_time=None,
        final_mass_val=None,
        final_spin_val=None,
    ):
        """
        Draw samples of A_lm from the predictive distribution corresponding to the query points.
        Mode (2,0) is not circularly polarized. lm=(2,0) corresponds to the real part, while lm=(2,10) corresponds to the imaginary part.

        Parameters
        ----------
        delta : array_like of shape (n_samples,) or float
            Asymmetric mass ratio of the query points. delta = (q-1)/(q+1) where q is the mass ratio. q = m1/m2 >= 1.

        chi1_x : array_like of shape (n_samples,) or float
            x-component of the primary spin at ISCO.

        chi1_y : array_like of shape (n_samples,) or float
            y-component of the primary spin at ISCO.

        chi1_z : array_like of shape (n_samples,) or float
            z-component of the primary spin at ISCO.

        chi2_x : array_like of shape (n_samples,) or float
            x-component of the secondary spin at ISCO.

        chi2_y : array_like of shape (n_samples,) or float
            y-component of the secondary spin at ISCO.

        chi2_z : array_like of shape (n_samples,) or float
            z-component of the secondary spin at ISCO.

        lm : tuple_like object
            Ordered couple (l,m) specifying the angular number l and azimuthal number m of the harmonic.

        mode : tuple_like object
            Ordered tuple specifying the queried quasi-normal mode.
            For linear modes, ordered triple (l,m,n).
            For quadratic modes, ordered couple of the form ((l1,m1,n1),(l2,m2,n2)).

        n_samples : int
            Number of samples to draw from the predictive distribution.

        start_time : float or None. Default=None.
            Start time of the ringdown waveform, relative to t_emop of the simulation.
            If None (default), assume that start_time=self.t0.

        final_mass_val : float or None. Default=None.
            Final mass of the remnant black hole. Only used if start_time is not None for retrodiction.

        final_spin_val : float or None. Default=None.
            Final spin of the remnant black hole. Only used if start_time is not None for retrodiction.

        """

        X = _make_stacked_array(
            delta,
            chi1_x,
            chi1_y,
            chi1_z,
            chi2_x,
            chi2_y,
            chi2_z,
        )

        if start_time is not None:
            assert (
                final_mass_val is not None
            ), "final_mass_val must be provided if start_time is not None"
            assert (
                final_spin_val is not None
            ), "final_spin_val must be provided if start_time is not None"

        amp_abs = self._fit_amps[lm][mode].sample_y(X, n_samples=n_samples)
        if start_time is not None:
            amp_abs = self._shift_amp(
                amp_abs,
                final_mass_val,
                final_spin_val,
                lm,
                mode,
                start_time,
            )
        return amp_abs

    def predict_t_emop(
        self,
        delta,
        chi1_x,
        chi1_y,
        chi1_z,
        chi2_x,
        chi2_y,
        chi2_z,
        return_std=False,
    ):
        """
        Predict the value of t_emop with respect to the peak of the L2 norm of the waveform, corresponding to the query points.
        Mode (2,0) is not circularly polarized. lm=(2,0) corresponds to the real part, while lm=(2,10) corresponds to the imaginary part.

        Parameters
        ----------
        delta : array_like of shape (n_samples,) or float
            Asymmetric mass ratio of the query points. delta = (q-1)/(q+1) where q is the mass ratio. q = m1/m2 >= 1.

        chi1_x : array_like of shape (n_samples,) or float
            x-component of the primary spin at ISCO.

        chi1_y : array_like of shape (n_samples,) or float
            y-component of the primary spin at ISCO.

        chi1_z : array_like of shape (n_samples,) or float
            z-component of the primary spin at ISCO.

        chi2_x : array_like of shape (n_samples,) or float
            x-component of the secondary spin at ISCO.

        chi2_y : array_like of shape (n_samples,) or float
            y-component of the secondary spin at ISCO.

        chi2_z : array_like of shape (n_samples,) or float
            z-component of the secondary spin at ISCO.

        return_std : bool. Default=False.
            Whether or not to return the standard deviation of the predictive distribution at the query points.

        """

        X = _make_stacked_array(
            delta,
            chi1_x,
            chi1_y,
            chi1_z,
            chi2_x,
            chi2_y,
            chi2_z,
        )

        if return_std:
            t_emop, t_emop_std = self._fit_t_emop.predict(X, return_std=return_std)
            return t_emop, t_emop_std

        else:
            t_emop = self._fit_t_emop.predict(X, return_std=return_std)
            return t_emop

    def predict_t_emop_abs_err(
        self,
        delta,
        chi1_x,
        chi1_y,
        chi1_z,
        chi2_x,
        chi2_y,
        chi2_z,
        return_std=False,
    ):
        """
        Predict the absolute error on t_emop with respect to the peak of the L2 norm of the waveform, corresponding to the query points.
        Mode (2,0) is not circularly polarized. lm=(2,0) corresponds to the real part, while lm=(2,10) corresponds to the imaginary part.

        Parameters
        ----------
        delta : array_like of shape (n_samples,) or float
            Asymmetric mass ratio of the query points. delta = (q-1)/(q+1) where q is the mass ratio. q = m1/m2 >= 1.

        chi1_x : array_like of shape (n_samples,) or float
            x-component of the primary spin at ISCO.

        chi1_y : array_like of shape (n_samples,) or float
            y-component of the primary spin at ISCO.

        chi1_z : array_like of shape (n_samples,) or float
            z-component of the primary spin at ISCO.

        chi2_x : array_like of shape (n_samples,) or float
            x-component of the secondary spin at ISCO.

        chi2_y : array_like of shape (n_samples,) or float
            y-component of the secondary spin at ISCO.

        chi2_z : array_like of shape (n_samples,) or float
            z-component of the secondary spin at ISCO.

        return_std : bool. Default=False.
            Whether or not to return the standard deviation of the predictive distribution at the query points.

        """
        X = _make_stacked_array(
            delta,
            chi1_x,
            chi1_y,
            chi1_z,
            chi2_x,
            chi2_y,
            chi2_z,
        )

        if return_std:
            t_emop_abs_err, t_emop_abs_err_std = self._fit_t_emop_abs_err.predict(
                X, return_std=return_std
            )
            return t_emop_abs_err, t_emop_abs_err_std

        else:
            t_emop_abs_err = self._fit_t_emop_abs_err.predict(X, return_std=return_std)
            return t_emop_abs_err

    def sample_t_emop(
        self,
        delta,
        chi1_x,
        chi1_y,
        chi1_z,
        chi2_x,
        chi2_y,
        chi2_z,
        n_samples=1,
    ):
        """
        Sample the value of t_emop with respect to the peak of the L2 norm of the waveform, corresponding to the query points.
        Mode (2,0) is not circularly polarized. lm=(2,0) corresponds to the real part, while lm=(2,10) corresponds to the imaginary part.

        Parameters
        ----------
        delta : array_like of shape (n_samples,) or float
            Asymmetric mass ratio of the query points. delta = (q-1)/(q+1) where q is the mass ratio. q = m1/m2 >= 1.

        chi1_x : array_like of shape (n_samples,) or float
            x-component of the primary spin at ISCO.

        chi1_y : array_like of shape (n_samples,) or float
            y-component of the primary spin at ISCO.

        chi1_z : array_like of shape (n_samples,) or float
            z-component of the primary spin at ISCO.

        chi2_x : array_like of shape (n_samples,) or float
            x-component of the secondary spin at ISCO.

        chi2_y : array_like of shape (n_samples,) or float
            y-component of the secondary spin at ISCO.

        chi2_z : array_like of shape (n_samples,) or float
            z-component of the secondary spin at ISCO.

        n_samples : int
            Number of samples to draw from the predictive distribution.
        """
        X = _make_stacked_array(
            delta,
            chi1_x,
            chi1_y,
            chi1_z,
            chi2_x,
            chi2_y,
            chi2_z,
        )

        t_emop = self._fit_t_emop.sample_y(X, n_samples=n_samples)
        return t_emop

    def _shift_amp(
        self,
        amp,
        final_mass_val,
        final_spin_val,
        lm,
        mode,
        start_time,
        qnm_method="interp",
    ):
        return _shift_amp_general(
            self, amp, final_mass_val, final_spin_val, lm, mode, start_time, qnm_method
        )


class AmplitudeFit3dq8:

    def __init__(self, fit_dict):
        self._fit_amps = fit_dict["amps"]
        self.t0 = fit_dict["time_from_peak"]
        ## deduce modes
        self.modes = {}
        for k in self._fit_amps.keys():
            if k[1] >= 0:
                self.modes[k] = [mode for mode in self._fit_amps[k].keys()]

    def __print__(self):
        out = self._descr
        return out

    def __repr__(self):
        out = self._descr
        return out

    def predict_xy_amp(self, mass_ratio, chi1z, chi2z, lm, mode, return_std=False):
        """
        Predict the values of A^x and A^y corresponding to the query points (mass_ratio,chi1z,chi2z).

        Parameters
        ----------
        mass_ratio : array_like of shape (n_samples,) or float
            Mass ratio of the query points.

        chi1z : array_like of shape (n_samples,) or float
            Projection along the z axis of the primary spin.

        chi2z : array_like of shape (n_samples,) or float
            Projection along the z axis of the secondary spin.

        lm : tuple_like object
            Ordered couple (l,m) specifying the angular number l and azimuthal number m of the harmonic.

        mode : tuple_like object
            Ordered tuple specifying the queried quasi-normal mode.
            For linear modes, ordered triple (l,m,n).
            For quadratic modes, ordered couple of the form ((l1,m1,n1),(l2,m2,n2)).

        return_std : bool. Default=False.
            Whether or not to return the standard deviation of the predictive distribution at the query points.
        """
        X = self._transform_X(mass_ratio, chi1z, chi2z)
        if lm == (2, 2) and mode == (2, 2, 0):
            amp_x = self._fit_amps[lm][mode].predict(X, return_std=return_std)
            amp_y = np.zeros_like(mass_ratio)
            if return_std:
                mu_x, std_x = amp_x
                mu_y = amp_y
                std_y = np.zeros_like(mu_y)
                return mu_x, mu_y, std_x, std_y
            else:
                return amp_x, amp_y
        else:
            if return_std:
                out_mu, out_std = self._fit_amps[lm][mode].predict(
                    X, return_std=return_std
                )
                mu_x, mu_y = out_mu.T
                std_x, std_y = out_std.T
                return mu_x, mu_y, std_x, std_y
            else:
                out_mu = self._fit_amps[lm][mode].predict(X, return_std=return_std)
                mu_x, mu_y = out_mu.T
                return mu_x, mu_y

    def predict_amp(
        self, mass_ratio, chi1z, chi2z, lm, mode, return_std=False, start_time=None
    ):
        """
        Predict the value of abs(A^x+iA^y) corresponding to the query points (mass_ratio,chi1z,chi2z).

        Parameters
        ----------
        mass_ratio : array_like of shape (n_samples,) or float
            Mass ratio of the query points.

        chi1z : array_like of shape (n_samples,) or float
            Projection along the z axis of the primary spin.

        chi2z : array_like of shape (n_samples,) or float
            Projection along the z axis of the secondary spin.

        lm : tuple_like object
            Ordered couple (l,m) specifying the angular number l and azimuthal number m of the harmonic.

        mode : tuple_like object
            Ordered tuple specifying the queried quasi-normal mode.
            For linear modes, ordered triple (l,m,n).
            For quadratic modes, ordered couple of the form ((l1,m1,n1),(l2,m2,n2)).

        return_std : bool. Default=False.
            Whether or not to return the standard deviation of the predictive distribution at the query points.

        start_time : float or None. Default=None.
            Start time of the ringdown waveform, relative to the peak of |h_{22}|.
            If None (default), assume that start_time=self.t0.
        """
        if return_std:
            mu_x, mu_y, std_x, std_y = self.predict_xy_amp(
                mass_ratio, chi1z, chi2z, lm, mode, return_std=return_std
            )
            mu_abs = np.sqrt(mu_x**2 + mu_y**2)
            std_abs = np.sqrt((mu_x * std_x) ** 2 + (mu_y * std_y) ** 2) / mu_abs
            if start_time is not None:
                mu_abs = self._shift_amp(
                    mu_abs, mass_ratio, chi1z, chi2z, lm, mode, start_time
                )
                std_abs = self._shift_amp(
                    std_abs, mass_ratio, chi1z, chi2z, lm, mode, start_time
                )
            return mu_abs, std_abs
        else:
            mu_x, mu_y = self.predict_xy_amp(
                mass_ratio, chi1z, chi2z, lm, mode, return_std=return_std
            )
            mu_abs = np.sqrt(mu_x**2 + mu_y**2)
            if start_time is not None:
                mu_abs = self._shift_amp(
                    mu_abs, mass_ratio, chi1z, chi2z, lm, mode, start_time
                )
            return mu_abs

    def predict_phase(
        self, mass_ratio, chi1z, chi2z, lm, mode, return_std=False, start_time=None
    ):
        """
        Predict the value of angle(A^x+iA^y)/beta corresponding to the query points (mass_ratio,chi1z,chi2z).
        beta is a correction factor introduced in CITE-THE-PAPER to help fitting of A^x and A^y.

        Parameters
        ----------
        mass_ratio : array_like of shape (n_samples,) or float
            Mass ratio of the query points.

        chi1z : array_like of shape (n_samples,) or float
            Projection along the z axis of the primary spin.

        chi2z : array_like of shape (n_samples,) or float
            Projection along the z axis of the secondary spin.

        lm : tuple_like object
            Ordered couple (l,m) specifying the angular number l and azimuthal number m of the harmonic.

        mode : tuple_like object
            Ordered tuple specifying the queried quasi-normal mode.
            For linear modes, ordered triple (l,m,n).
            For quadratic modes, ordered couple of the form ((l1,m1,n1),(l2,m2,n2)).

        return_std : bool. Default=False.
            Whether or not to return the standard deviation of the predictive distribution at the query points.

        start_time : float or None. Default=None.
            Start time of the ringdown waveform, relative to the peak of |h_{22}|.
            If None (default), assume that start_time=self.t0.
        """
        beta = 1 + np.mod(lm[1], 2)
        if return_std:
            mu_x, mu_y, std_x, std_y = self.predict_xy_amp(
                mass_ratio, chi1z, chi2z, lm, mode, return_std=return_std
            )
            mu_abs = np.sqrt(mu_x**2 + mu_y**2)
            mu_phi = np.angle(mu_x + 1j * mu_y) / beta
            std_phi = (
                np.sqrt((mu_x * std_y) ** 2 + (mu_y * std_y) ** 2) / mu_abs**2 / beta
            )
            if start_time is not None:
                mu_phi = self._shift_phase(
                    mu_phi, mass_ratio, chi1z, chi2z, lm, mode, start_time
                )
            return mu_phi, std_phi
        else:
            mu_x, mu_y = self.predict_xy_amp(
                mass_ratio, chi1z, chi2z, lm, mode, return_std=return_std
            )
            mu_phi = np.angle(mu_x + 1j * mu_y) / beta
            if start_time is not None:
                mu_phi = self._shift_phase(
                    mu_phi, mass_ratio, chi1z, chi2z, lm, mode, start_time
                )
            return mu_phi

    def sample_xy_amp(self, mass_ratio, chi1z, chi2z, lm, mode, n_samples=1):
        """
        Draw samples of of A^x and A^y from the predictive distribution corresponding to the query points (mass_ratio,chi1z,chi2z).

        Parameters
        ----------
        mass_ratio : array_like of shape (n_samples_X,) or float
            Mass ratio of the query points.

        chi1z : array_like of shape (n_samples_X,) or float
            Projection along the z axis of the primary spin.

        chi2z : array_like of shape (n_samples_X,) or float
            Projection along the z axis of the secondary spin.

        lm : tuple-like object
            Ordered couple (l,m) specifying the angular number l and azimuthal number m of the harmonic.

        mode : tuple-like object
            Ordered tuple specifying the queried quasi-normal mode.
            For linear modes, ordered triple (l,m,n).
            For quadratic modes, ordered couple of the form ((l1,m1,n1),(l2,m2,n2)).

        n_samples : int. Default=1.
            Number of samples to be drawn per query point.

        Returns
        -------
        amp_x : array-like of shape (n_samples_X, n_samples)
        amp_y : array-like of shape (n_samples_X, n_samples)
        """
        X = self._transform_X(mass_ratio, chi1z, chi2z)
        if lm == (2, 2) and mode == (2, 2, 0):
            amp_x = self._fit_amps[lm][mode].sample_y(X, n_samples=n_samples)
            amp_y = np.zeros_like(amp_x)
        else:
            out = self._fit_amps[lm][mode].sample_y(X, n_samples=n_samples)
            amp_x, amp_y = out[:, 0, :], out[:, 1, :]
        return amp_x, amp_y

    def sample_amp(
        self, mass_ratio, chi1z, chi2z, lm, mode, n_samples=1, start_time=None
    ):
        """
        Draw samples of abs(A^x+iA^y) from the predictive distribution corresponding to the query points (mass_ratio,chi1z,chi2z).

        Parameters
        ----------
        mass_ratio : array_like of shape (n_samples_X,) or float
            Mass ratio of the query points.

        chi1z : array_like of shape (n_samples_X,) or float
            Projection along the z axis of the primary spin.

        chi2z : array_like of shape (n_samples_X,) or float
            Projection along the z axis of the secondary spin.

        lm : tuple-like object
            Ordered couple (l,m) specifying the angular number l and azimuthal number m of the harmonic.

        mode : tuple-like object
            Ordered tuple specifying the queried quasi-normal mode.
            For linear modes, ordered triple (l,m,n).
            For quadratic modes, ordered couple of the form ((l1,m1,n1),(l2,m2,n2)).

        n_samples : int. Default=1.
            Number of samples to be drawn per query point.

        start_time : float or None. Default=None.
            Start time of the ringdown waveform, relative to the peak of |h_{22}|.
            If None (default), assume that start_time=self.t0.

        Returns
        -------
        amp : array-like of shape (n_samples_X, n_samples)
        """
        amp_x, amp_y = self.sample_xy_amp(
            mass_ratio, chi1z, chi2z, lm, mode, n_samples=n_samples
        )
        amp = np.sqrt(amp_x**2 + amp_y**2)
        if start_time is not None:
            amp = np.array(
                [
                    self._shift_amp(
                        amp_k, mass_ratio, chi1z, chi2z, lm, mode, start_time
                    )
                    for amp_k in amp.T
                ]
            )
        return amp

    def sample_phase(
        self, mass_ratio, chi1z, chi2z, lm, mode, n_samples=1, start_time=None
    ):
        """
        Draw samples of angle(A^x+iA^y)/beta from the predictive distribution corresponding to the query points (mass_ratio,chi1z,chi2z).
        beta is a correction factor introduced in CITE-THE-PAPER to help fitting of A^x and A^y.

        Parameters
        ----------
        mass_ratio : array_like of shape (n_samples,) or float
            Mass ratio of the query points.

        chi1z : array_like of shape (n_samples,) or float
            Projection along the z axis of the primary spin.

        chi2z : array_like of shape (n_samples,) or float
            Projection along the z axis of the secondary spin.

        lm : tuple-like object
            Ordered couple (l,m) specifying the angular number l and azimuthal number m of the harmonic.

        mode : tuple-like object
            Ordered tuple specifying the queried quasi-normal mode.
            For linear modes, ordered triple (l,m,n).
            For quadratic modes, ordered couple of the form ((l1,m1,n1),(l2,m2,n2)).

        n_samples : int. Default=1.
            Number of samples to be drawn per query point.

        start_time : float or None. Default=None
            Start time of the ringdown waveform, relative to the peak of |h_{22}|.
            If None (default), assume that start_time=self.t0.

        Returns
        -------
        phase : array-like of shape (n_samples_X, n_samples)
        """
        beta = 1 + np.mod(lm[1], 2)
        amp_x, amp_y = self.sample_amp_xy_amp(
            mass_ratio, chi1z, chi2z, lm, mode, n_samples=n_samples
        )
        phi = np.angle(amp_x + 1j * amp_y) / beta
        if start_time is not None:
            phi = np.array(
                [
                    self._shift_phase(
                        phi_k, mass_ratio, chi1z, chi2z, lm, mode, start_time
                    )
                    for phi_k in phi.T
                ]
            )
        return phi

    def _transform_X(self, mass_ratio, chi1z, chi2z):
        mass1 = mass_ratio / (1 + mass_ratio)
        mass2 = 1 / (1 + mass_ratio)
        chip = chi1z * mass1 + chi2z * mass2
        chim = chi1z * mass1 - chi2z * mass2
        eta = np.clip(mass_ratio / (1 + mass_ratio) ** 2, 0, 0.25)
        delta = np.sqrt(1 - 4 * eta)
        out = np.vstack((delta, chip, chim)).T
        print("X shape", out.shape)
        return out

    def _shift_amp(
        self, amp, mass_ratio, chi1z, chi2z, lm, mode, start_time, qnm_method="interp"
    ):
        if start_time == self.t0:
            return amp
        DT = start_time - self.t0
        mass1 = mass_ratio / (1 + mass_ratio)
        mass2 = 1 / (1 + mass_ratio)
        mf = final_mass(mass1, mass2, chi1z, chi2z, aligned_spins=True, method="B12")
        sf = final_spin(mass1, mass2, chi1z, chi2z, aligned_spins=True, method="H16")
        if hasattr(mode[0], "__len__"):
            ## handle quadratic mode
            inv_tau = 0.0
            for linear_mode in mode:
                inv_tau += (
                    1.0
                    / qnm_Kerr(
                        mf, sf, linear_mode, qnm_method=qnm_method, SI_units=False
                    )[1]
                )
        else:
            ## handle linear mode
            inv_tau = (
                1.0 / qnm_Kerr(mf, sf, mode, qnm_method=qnm_method, SI_units=False)[1]
            )
        if lm != (2, 2) or mode != (2, 2, 0):
            inv_tau -= (
                1.0
                / qnm_Kerr(mf, sf, (2, 2, 0), qnm_method=qnm_method, SI_units=False)[1]
            )
        out = amp * np.exp(-DT * inv_tau)
        return out

    def _shift_phase(
        self, phase, mass_ratio, chi1z, chi2z, lm, mode, start_time, qnm_method="interp"
    ):
        if start_time == self.t0:
            return phase
        DT = start_time - self.t0
        mass1 = mass_ratio / (1 + mass_ratio)
        mass2 = 1 / (1 + mass_ratio)
        mf = final_mass(mass1, mass2, chi1z, chi2z, aligned_spins=True, method="B12")
        sf = final_spin(mass1, mass2, chi1z, chi2z, aligned_spins=True, method="H16")
        if hasattr(mode[0], "__len__"):
            ## handle quadratic mode
            freq = 0.0
            for linear_mode in mode:
                freq += qnm_Kerr(
                    mf, sf, linear_mode, qnm_method=qnm_method, SI_units=False
                )[0]
        else:
            ## handle linear mode
            freq = (
                1.0 / qnm_Kerr(mf, sf, mode, qnm_method=qnm_method, SI_units=False)[0]
            )
        if lm != (2, 2) or mode != (2, 2, 0):
            freq -= (
                0.5
                * lm[1]
                * qnm_Kerr(mf, sf, (2, 2, 0), qnm_method=qnm_method, SI_units=False)[0]
            )
        out = phase + 2 * np.pi * freq * DT
        out = np.angle(np.exp(1j * out))
        return out


class CustomGPR(RegressorMixin, BaseEstimator):

    def __init__(self):
        return None

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        normalize_y=True,
        linear_fit: bool = True,
        precessing=False,
        **kwargs,
    ):
        """
        Fit GPR by first subtracting a linear fit.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Training data. Prior to fit, they are normalized to zero mean and unit variance.

        y : array_like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        sample_weight : array_like of shape (n_samples). Default=None.
            Individual weights for each sample.
            None (default) is equivalent to 1-D sample_weight filled with ones.
            If not None:
                (a) the training data for the linear fit are weighted by sample_weights, and
                (b) the inverse of sample_weights is added to the diagonal of the kernel matrix during GPR fitting.

        normalize_y : bool. Default=True.
            Whether or not to normalize the target values y by removing the mean and scaling to unit-variance.
            If True (default), it only applies to the GPR fit.
        """
        if sample_weight is None:
            sample_weight = np.ones((X.shape[0],))
            alpha = 1e-10
        else:
            alpha = 1 / sample_weight

        self.linear_fit = (
            linear_fit  ## Need this because precessing are done without linear fit
        )
        self.precessing = precessing  ## used in _prepare_kernel(). Need this because precessing are initialised with different base kernel length_scale

        if linear_fit:
            self._linear_fit(X, y, sample_weight=sample_weight)
            self._gpr_fit(
                X,
                y,
                alpha=alpha,
                normalize_y=normalize_y,
                n_restarts_optimizer=0,
                **kwargs,
            )
        else:
            self._gpr_fit(
                X,
                y,
                alpha=alpha,
                normalize_y=normalize_y,
                n_restarts_optimizer=0,
                **kwargs,
            )
        return None

    def predict(self, X, return_std=False):
        """
        Predict the target corresponding to the query points X.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The query points.

        return_std : bool. Default=False.
            Whether or not to return the standard deviation of the predictive distribution at the query points X.

        Returns
        -------
        y_mean : array_like of shape (n_samples,) or (n_samples, n_targets)
            Mean of the predictive distribution at the query points X.

        y_std : array_like of shape (n_samples,) or (n_samples, n_targets), optional
            Standard deviation of the predictive distribution at the query points X.
            Only returned when return_std=True.
        """
        ## If self.linear_fit is not found, create and set to True (means that these are the old models)
        if not hasattr(self, "linear_fit"):
            self.linear_fit = True

        if self.linear_fit:
            if return_std:
                out, std = self.gpr.predict(X, return_std=True)
                out += self.lin.predict(X)
                return out, std
            else:
                out = self.gpr.predict(X, return_std=False)
                out += self.lin.predict(X)
                return out
        else:
            if return_std:
                out, std = self.gpr.predict(X, return_std=True)
                return out, std
            else:
                out = self.gpr.predict(X, return_std=False)
                return out

    def rms_score(self, X, y_true, sample_weight=None):
        """
        Return the root mean squared error of the prediction.

        Prameters
        ---------
        X : array_like of shape (n_samples, n_features)
            The query points.

        y_true : array_like of shape (n_samples,) or (n_samples, n_targets)
            Correct target values.

        sample_weight : array_like of shape (n_samples,). Default=None.
            Individual weights for each sample.
            None (default) is equivalent to 1-D sample_weight filled with ones.
        """
        y_pred = self.predict(X, return_std=False)
        out = (
            mean_squared_error(
                y_true,
                y_pred,
                sample_weight=sample_weight,
                multioutput="uniform_average",
            )
            ** 0.5
        )
        return out

    def std_score(self, X, y_true, sample_weight=None):
        """
        Return the root mean squared error of the prediction, where each sample is weighted by the inverse square of the standard deviation of the predcitve distribution.
        It can be thought as the average distance of the mean predictions from y_true, in units of GPR standard deviation.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The query points.

        y_true : array_like of shape (n_samples,) or (n_samples, n_targets)
            Correct target values.

        sample_weight : array_like of shape (n_samples,). Default=None.
            Individual weights for each sample.
            None (default) is equivalent to 1-D sample_weight filled with ones.
        """
        y_pred, std_pred = self.predict(X, return_std=True)
        if sample_weight is not None:
            out = (
                np.sum((y_true - y_pred) ** 2 / std_pred**2 * sample_weight)
                / np.sum(sample_weight)
            ) ** 0.5
        else:
            out = np.mean((y_true - y_pred) ** 2 / std_pred**2) ** 0.5
        return out

    def r2_score(self, X, y_true, sample_weight=None):
        """
        Return the R2 (coefficient of determination) regression score.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The query points.

        y_true : array_like of shape (n_samples,) or (n_samples, n_targets)
            Correct target values.

        sample_weight : array_like of shape (n_samples,). Default=None.
            Individual weights for each sample.
            None (default) is equivalent to 1-D sample_weight filled with ones.
        """
        y_pred = self.predict(X, return_std=False)
        out = r2_score(
            y_true, y_pred, sample_weight=sample_weight, multioutput="uniform_average"
        )
        return out

    def sample_y(self, X, n_samples=1, random_state=42):
        """
        Draw samples from the GPR and evaluate at X.

        Parameters
        ----------
        X : array_like of shape (n_samples_X, n_features)
            The query points.

        n_samples : int. Default=1.
            Number of samples to be drawn per query point.

        Returns
        -------
        y_samples : array-like of shape (n_samples_X, n_samples) or (n_samples_X, n_targets, n_samples)
        """
        X_transformed = self.gpr["scaler"].transform(X)
        out = self.gpr["gpr"].sample_y(
            X_transformed, n_samples=n_samples, random_state=random_state
        )

        ## If self.linear_fit is not found, create and set to True (means that these are the old models)
        if not hasattr(self, "linear_fit"):
            self.linear_fit = True
        if self.linear_fit:
            if len(out.shape) == 2:
                out += self.lin.predict(X)[:, np.newaxis]
            elif len(out.shape) == 3:
                out += self.lin.predict(X)[:, :, np.newaxis]
        return out

    def _linear_fit(self, X, y, sample_weight=None):
        self.lin = Pipeline(
            [("scaler", StandardScaler()), ("lin", LinearRegression())],
        )
        self.lin.fit(X, y, lin__sample_weight=sample_weight)
        return None

    def _gpr_fit(self, X, y, **kwargs):
        kernel = self._prepare_kernel(X_dim=X.shape[1], white_kernel=True)
        self.gpr = Pipeline(
            [("scaler", StandardScaler()), ("gpr", GPR(kernel=kernel, **kwargs))]
        )
        if self.linear_fit:
            y_lin = self.lin.predict(X)
            self.gpr.fit(X, y - y_lin)
        else:
            self.gpr.fit(X, y)

        return None

    def _prepare_kernel(self, X_dim, white_kernel=False):
        if self.precessing:  ## I used different initial length_scale for precessing
            kernel1 = RBF(
                length_scale=np.ones((X_dim,)) * 0.2, length_scale_bounds=(1e-10, 1e5)
            )
            kernel2 = ConstantKernel()
            kernel = kernel1 * kernel2
            if white_kernel:
                kernel3 = WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e2))
                kernel += kernel3
        else:
            kernel1 = RBF(
                length_scale=np.ones((X_dim,)), length_scale_bounds=(1e-10, 1e5)
            )
            kernel2 = ConstantKernel(constant_value_bounds=(1e-5, 1e5))
            kernel = kernel1 * kernel2
            if white_kernel:
                kernel3 = WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e2))
                kernel += kernel3
        return kernel
