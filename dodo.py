"""Define and automate tasks with ``doit``."""

import itertools
import pathlib
import re

import control
import doit
import joblib
import numpy as np
import pandas
import pykoop
import scipy.linalg
from matplotlib import pyplot as plt

import onesine
import tf_cover

# Directory containing ``dodo.py``
WD = pathlib.Path(__file__).parent.resolve()


def task_preprocess_experiments():
    """Preprocess raw data into pickle containing Pandas ``DataFrame``."""
    raw_dataset = WD.joinpath("dataset", "raw", "batch_b")
    preprocessed_dataset = WD.joinpath("build", "dataset.pickle")
    return {
        "actions": [
            (
                action_preprocess_experiments,
                (
                    raw_dataset,
                    preprocessed_dataset,
                ),
            )
        ],
        "targets": [preprocessed_dataset],
        "uptodate": [doit.tools.check_timestamp_unchanged(str(raw_dataset))],
        "clean": True,
    }


def task_compute_phase():
    """Compute phase offset."""
    preprocessed_dataset = WD.joinpath("build", "dataset.pickle")
    phase = WD.joinpath("build", "phase.pickle")
    return {
        "actions": [
            (
                action_compute_phase,
                (
                    preprocessed_dataset,
                    phase,
                ),
            )
        ],
        "file_dep": [preprocessed_dataset],
        "targets": [phase],
        "clean": True,
    }


def task_id_models():
    """Identify linear and Koopman models."""
    preprocessed_dataset = WD.joinpath("build", "dataset.pickle")
    phase = WD.joinpath("build", "phase.pickle")
    models_linear = WD.joinpath("build", "models_linear.pickle")
    yield {
        "name": "linear",
        "actions": [
            (
                action_id_models,
                (preprocessed_dataset, phase, models_linear, "linear"),
            )
        ],
        "file_dep": [preprocessed_dataset, phase],
        "targets": [models_linear],
        "clean": True,
    }
    models_koopman = WD.joinpath("build", "models_koopman.pickle")
    yield {
        "name": "koopman",
        "actions": [
            (
                action_id_models,
                (preprocessed_dataset, phase, models_koopman, "koopman"),
            )
        ],
        "file_dep": [preprocessed_dataset, phase],
        "targets": [models_koopman],
        "clean": True,
    }


def task_compute_residuals():
    """Compute residuals from linear and Koopman models."""
    models_linear = WD.joinpath("build", "models_linear.pickle")
    residuals_linear = WD.joinpath("build", "residuals_linear.pickle")
    yield {
        "name": "linear",
        "actions": [
            (
                action_compute_residuals,
                (models_linear, residuals_linear),
            )
        ],
        "file_dep": [models_linear],
        "targets": [residuals_linear],
        "clean": True,
    }
    models_koopman = WD.joinpath("build", "models_koopman.pickle")
    residuals_koopman = WD.joinpath("build", "residuals_koopman.pickle")
    yield {
        "name": "koopman",
        "actions": [
            (
                action_compute_residuals,
                (models_koopman, residuals_koopman),
            )
        ],
        "file_dep": [models_koopman],
        "targets": [residuals_koopman],
        "clean": True,
    }


def task_generate_uncertainty_weights():
    """Generate uncertainty weights models."""
    residuals_koopman = WD.joinpath("build", "residuals_koopman.pickle")
    orders_koopman = np.array(
        [
            [1, 3],
            [3, 3],
        ]
    )
    uncertainty_koopman_noload = WD.joinpath(
        "build", "uncertainty_koopman_noload.pickle"
    )
    uncertainty_mimo_koopman_noload = WD.joinpath(
        "build", "uncertainty_mimo_koopman_noload.png"
    )
    uncertainty_msv_koopman_noload = WD.joinpath(
        "build", "uncertainty_msv_koopman_noload.png"
    )
    nominal_noload = WD.joinpath("build", "nominal_noload.txt")
    yield {
        "name": "koopman_noload",
        "actions": [
            (
                action_generate_uncertainty_weights,
                (
                    residuals_koopman,
                    nominal_noload,
                    uncertainty_koopman_noload,
                    uncertainty_mimo_koopman_noload,
                    uncertainty_msv_koopman_noload,
                    orders_koopman,
                    "koopman",
                    "noload",
                ),
            )
        ],
        "file_dep": [residuals_koopman],
        "targets": [
            uncertainty_koopman_noload,
            nominal_noload,
            uncertainty_mimo_koopman_noload,
            uncertainty_msv_koopman_noload,
        ],
        "clean": True,
    }
    uncertainty_koopman_load = WD.joinpath("build", "uncertainty_koopman_load.pickle")
    uncertainty_mimo_koopman_load = WD.joinpath(
        "build", "uncertainty_mimo_koopman_load.png"
    )
    uncertainty_msv_koopman_load = WD.joinpath(
        "build", "uncertainty_msv_koopman_load.png"
    )
    nominal_load = WD.joinpath("build", "nominal_load.txt")
    yield {
        "name": "koopman_load",
        "actions": [
            (
                action_generate_uncertainty_weights,
                (
                    residuals_koopman,
                    nominal_load,
                    uncertainty_koopman_load,
                    uncertainty_mimo_koopman_load,
                    uncertainty_msv_koopman_load,
                    orders_koopman,
                    "koopman",
                    "load",
                ),
            )
        ],
        "file_dep": [residuals_koopman],
        "targets": [
            uncertainty_koopman_load,
            nominal_load,
            uncertainty_mimo_koopman_load,
            uncertainty_msv_koopman_load,
        ],
        "clean": True,
    }
    residuals_linear = WD.joinpath("build", "residuals_linear.pickle")
    orders_linear = np.array(
        [
            [1, 2],
            [1, 2],
        ]
    )
    uncertainty_linear_noload = WD.joinpath("build", "uncertainty_linear_noload.pickle")
    uncertainty_mimo_linear_noload = WD.joinpath(
        "build", "uncertainty_mimo_linear_noload.png"
    )
    uncertainty_msv_linear_noload = WD.joinpath(
        "build", "uncertainty_msv_linear_noload.png"
    )
    yield {
        "name": "linear_noload",
        "actions": [
            (
                action_generate_uncertainty_weights,
                (
                    residuals_linear,
                    nominal_noload,
                    uncertainty_linear_noload,
                    uncertainty_mimo_linear_noload,
                    uncertainty_msv_linear_noload,
                    orders_linear,
                    "linear",
                    "noload",
                ),
            )
        ],
        "file_dep": [residuals_linear, nominal_noload],
        "targets": [
            uncertainty_linear_noload,
            uncertainty_mimo_linear_noload,
            uncertainty_msv_linear_noload,
        ],
        "clean": True,
    }
    uncertainty_linear_load = WD.joinpath("build", "uncertainty_linear_load.pickle")
    uncertainty_mimo_linear_load = WD.joinpath(
        "build", "uncertainty_mimo_linear_load.png"
    )
    uncertainty_msv_linear_load = WD.joinpath(
        "build", "uncertainty_msv_linear_load.png"
    )
    yield {
        "name": "linear_load",
        "actions": [
            (
                action_generate_uncertainty_weights,
                (
                    residuals_linear,
                    nominal_load,
                    uncertainty_linear_load,
                    uncertainty_mimo_linear_load,
                    uncertainty_msv_linear_load,
                    orders_linear,
                    "linear",
                    "load",
                ),
            )
        ],
        "file_dep": [residuals_linear, nominal_load],
        "targets": [
            uncertainty_linear_load,
            uncertainty_mimo_linear_load,
            uncertainty_msv_linear_load,
        ],
        "clean": True,
    }


def action_preprocess_experiments(
    raw_dataset_path: pathlib.Path,
    preprocessed_dataset_path: pathlib.Path,
):
    """Preprocess raw data into pickle containing Pandas ``DataFrame``."""
    preprocessed_dataset = WD.joinpath("build/dataset.pickle")
    gear_ratio = 1 / 100
    rad_per_deg = 2 * np.pi / 360
    t_step = 1e-3
    dfs = []
    for path in itertools.chain(
        raw_dataset_path.joinpath("population").iterdir(),
        raw_dataset_path.joinpath("outliers").iterdir(),
    ):
        # Skip path that's not a directory
        if not path.is_dir():
            continue
        # Set up regex (e.g., 20221220T101041_001002_noload)
        pattern = re.compile(r"^(\d\d\d\d\d\d\d\dT\d\d\d\d\d\d)_(\d\d\d\d\d\d)_(.*)$")
        match = pattern.match(path.stem)
        # Skip if no match
        if match is None:
            continue
        # Get metadata
        timestamp = match[1]
        serial_no = match[2]
        load = match[3] == "load"
        # Parse episodes
        for file in sorted(path.glob("*.csv")):
            # Get episode
            ep = int(str(file)[-7:-4])
            # Load data
            array = np.loadtxt(
                file,
                delimiter=",",
                skiprows=1,
            )
            # Select and shift columns to align them in time, due to bug in
            # data acquisition software
            joint_posvel_raw = array[:-1, 1:3] * gear_ratio * rad_per_deg
            joint_trq_raw = array[:-1, [3]] / 100  # Percent to normalized
            target_joint_posvel_raw = array[1:, 4:6] * gear_ratio * rad_per_deg
            # Calibrate for initial position offset due to bug in data
            # acquisition software
            error_raw = target_joint_posvel_raw - joint_posvel_raw
            error_offset = np.mean(error_raw[500:1000, :], axis=0)
            # Apply offset and remove first second of recording where velocity
            # is zero
            joint_posvel = joint_posvel_raw[1000:, :] + error_offset
            error_offset[1] = 0
            joint_trq = joint_trq_raw[1000:, :]
            target_joint_posvel = target_joint_posvel_raw[1000:, :]
            # Create ``DataFrame``
            df_dict = {
                "k": np.arange(target_joint_posvel.shape[0]),
                "t": np.arange(target_joint_posvel.shape[0]) * t_step,
                "joint_pos": joint_posvel[:, 0],
                "joint_vel": joint_posvel[:, 1],
                "joint_trq": joint_trq[:, 0],
                "target_joint_pos": target_joint_posvel[:, 0],
                "target_joint_vel": target_joint_posvel[:, 1],
            }
            df = pandas.DataFrame(df_dict)
            df["serial_no"] = serial_no
            df["load"] = load
            df["episode"] = ep
            df["timestamp"] = timestamp
            dfs.append(df)
    merged_df = pandas.concat(dfs)
    merged_df.attrs["t_step"] = t_step
    merged_df.sort_values(
        by=["serial_no", "load", "episode", "k"],
        inplace=True,
    )
    preprocessed_dataset.parent.mkdir(exist_ok=True)
    joblib.dump(merged_df, preprocessed_dataset)


def action_compute_phase(
    dataset_path: pathlib.Path,
    phase_path: pathlib.Path,
):
    """Compute phase offset."""
    # Load dataset
    dataset = joblib.load(dataset_path)
    # Settings
    n_phase_samples = 1000
    min_length = 600
    min_vel = 3
    trim = 100
    # Iterate over episodes
    df_lst = []
    for i, dataset_ep in dataset.groupby(by=["serial_no", "load", "episode"]):
        tvel = dataset_ep["target_joint_vel"].to_numpy()
        vel = dataset_ep["joint_vel"].to_numpy()
        tpos = dataset_ep["target_joint_pos"].to_numpy()
        pos = dataset_ep["joint_pos"].to_numpy()
        # Find points where velocity changes
        vel_changes = np.ravel(np.argwhere(np.diff(tvel, prepend=0) != 0))
        # Split into constant-velocity segments
        X = np.vstack(
            [
                pos,
                vel,
                tpos,
                tvel,
            ]
        ).T
        const_vel_segments = np.split(X, vel_changes)
        # Find first segment of required length and speed
        for segment in const_vel_segments:
            X_const_vel = segment[trim:-trim, :]
            if segment.shape[0] > min_length:
                if np.all(segment[:, 3] > min_vel):
                    direction = "forward"
                elif np.all(segment[:, 3] < -1 * min_vel):
                    direction = "reverse"
                else:
                    continue
                # Compute normalized velocity error in that segment
                vel_err = X_const_vel[:, 1] - X_const_vel[:, 3]
                norm_vel_err = vel_err / np.max(np.abs(vel_err))
                # Create array of phases to test
                phases = np.linspace(0, 2 * np.pi, n_phase_samples)
                # Compute inner product of error and shifted signal for each phase
                inner_products = (
                    np.array(
                        [
                            np.sum(norm_vel_err * np.sin(100 * X_const_vel[:, 0] + p))
                            for p in phases
                        ]
                    )
                    / norm_vel_err.shape[0]
                )
                # Find best phase
                # There are two phases that will work (+ve and -ve correlations)
                optimal_phase = phases[np.argmax(inner_products)]
                df_lst.append(
                    i + (direction, optimal_phase, phases, inner_products),
                )
    df = pandas.DataFrame(
        df_lst,
        columns=[
            "serial_no",
            "load",
            "episode",
            "direction",
            "optimal_phase",
            "phases",
            "inner_products",
        ],
    )
    df.sort_values(
        by=["serial_no", "load", "episode"],
        inplace=True,
    )
    phase_path.parent.mkdir(exist_ok=True)
    joblib.dump(df, phase_path)


def action_id_models(
    dataset_path: pathlib.Path,
    phase_path: pathlib.Path,
    models_path: pathlib.Path,
    koopman: str,
):
    """Identify linear and Koopman models."""
    dataset = joblib.load(dataset_path)
    n_train = 18
    n_inputs = 2
    episode_feature = True
    t_step = dataset.attrs["t_step"]

    df_lst = []
    for i, dataset_ep in dataset.groupby(by=["serial_no", "load"]):
        X = dataset_ep[
            [
                "episode",
                "joint_pos",
                "joint_vel",
                "joint_trq",
                "target_joint_pos",
                "target_joint_vel",
            ]
        ]
        X_train = X.loc[X["episode"] < n_train].to_numpy()
        if koopman == "koopman":
            # Get optimal phase shift
            phase = joblib.load(phase_path)
            phi_all = phase.loc[(phase["serial_no"] == i[0]) & (phase["load"] == i[1])]
            optimal_phi = _circular_mean(phi_all["optimal_phase"].to_numpy())
            # Set lifting functions
            lf = [
                (
                    "sin",
                    onesine.OneSineLiftingFn(
                        f=100,
                        i=0,
                        phi=optimal_phi,
                    ),
                )
            ]
        else:
            lf = None
        # Fit Koopman model
        kp = pykoop.KoopmanPipeline(
            lifting_functions=lf,
            regressor=pykoop.Edmd(alpha=90),
        )
        kp.fit(X_train, n_inputs=n_inputs, episode_feature=episode_feature)
        # Create state-space model
        nx = kp.regressor_.coef_.T.shape[0]
        nu = kp.regressor_.coef_.T.shape[1] - nx
        A = kp.regressor_.coef_.T[:, :nx]
        B = kp.regressor_.coef_.T[:, nx:]
        ss = control.StateSpace(
            A,
            B,
            np.eye(nx),
            np.zeros((nx, nu)),
            dt=t_step,
        )
        # Check stability
        if np.any(np.abs(scipy.linalg.eigvals(A)) >= 1):
            raise RuntimeError(f"System {i} is unstable.")
        ss_mat = (ss.A, ss.B, ss.C, ss.D, ss.dt)
        df_lst.append(i + (kp, ss_mat))
    df = pandas.DataFrame(
        df_lst,
        columns=["serial_no", "load", "koopman_pipeline", "state_space"],
    )
    df.sort_values(
        by=["serial_no", "load"],
        inplace=True,
    )
    df.attrs["t_step"] = t_step
    models_path.parent.mkdir(exist_ok=True)
    joblib.dump(df, models_path)


def action_compute_residuals(
    models_path: pathlib.Path,
    residuals_path: pathlib.Path,
):
    """Compute residuals from linear and Koopman models."""
    models = joblib.load(models_path)
    t_step = models.attrs["t_step"]
    uncertainty_forms = [
        "additive",
        "input_multiplicative",
        "output_multiplicative",
        "inverse_additive",
        "inverse_input_multiplicative",
        "inverse_output_multiplicative",
    ]
    f = np.logspace(-3, np.log10(0.5 / t_step), 1000)
    outlier_sn = "000000"

    df_lst = []
    for i, model_ep in models.groupby(by=["serial_no", "load"]):
        # Create nominal and off-nominal plants
        nominal = control.StateSpace(*model_ep["state_space"].item())
        off_nominal_ = models.loc[
            (models["serial_no"] != i[0])
            & (models["serial_no"] != outlier_sn)
            & (models["load"] == i[1])
        ]
        off_nominal = [
            control.StateSpace(*on_) for on_ in off_nominal_["state_space"].to_list()
        ]
        off_nominal_sn = off_nominal_["serial_no"].to_list()
        # Compute residuals
        for uncertainty_form in uncertainty_forms:
            residual_data = _residuals(
                nominal,
                off_nominal,
                t_step,
                f,
                form=uncertainty_form,
            )
            df_lst.append(
                i
                + (
                    uncertainty_form,
                    residual_data["peak_bound"],
                    residual_data["area_bound"],
                    residual_data["bound"],
                    residual_data["magnitudes"],
                    residual_data["residuals"],
                    off_nominal_sn,
                )
            )
    # Add averaged model
    As, Bs, Cs, Ds, dts = zip(*models["state_space"].to_list())
    A_avg = np.mean(np.array(As), axis=0)
    B_avg = np.mean(np.array(Bs), axis=0)
    nominal = control.StateSpace(A_avg, B_avg, Cs[0], Ds[0], dts[0])
    off_nominal_ = models["state_space"].to_list()
    off_nominal = [control.StateSpace(*on_) for on_ in off_nominal_]
    off_nominal_sn = models["serial_no"].to_list()
    # Compute residuals
    for uncertainty_form in uncertainty_forms:
        residual_data = _residuals(
            nominal,
            off_nominal,
            t_step,
            f,
            form=uncertainty_form,
        )
        df_lst.append(
            (
                "average",
                i[1],
                uncertainty_form,
                residual_data["peak_bound"],
                residual_data["area_bound"],
                residual_data["bound"],
                residual_data["magnitudes"],
                residual_data["residuals"],
                off_nominal_sn,
            )
        )
    df = pandas.DataFrame(
        df_lst,
        columns=[
            "nominal_serial_no",
            "load",
            "uncertainty_form",
            "peak_bound",
            "area_bound",
            "bound",
            "magnitudes",
            "residuals",
            "off_nominal_serial_no",
        ],
    )
    df.sort_values(
        by=["nominal_serial_no", "load", "uncertainty_form"],
        inplace=True,
    )
    df.attrs["t_step"] = t_step
    df.attrs["f"] = f
    residuals_path.parent.mkdir(exist_ok=True)
    joblib.dump(df, residuals_path)


def action_generate_uncertainty_weights(
    residuals_path: pathlib.Path,
    nominal_path: pathlib.Path,
    uncertainty_path: pathlib.Path,
    uncertainty_mimo_path: pathlib.Path,
    uncertainty_msv_path: pathlib.Path,
    orders: np.ndarray,
    koopman: str,
    load: str,
):
    """Generate uncertainty weights models."""
    residuals = joblib.load(residuals_path)
    t_step = residuals.attrs["t_step"]
    f = residuals.attrs["f"]
    omega = 2 * np.pi * f
    load_bool = load == "load"

    if koopman == "linear":
        # nominal = joblib.load(nominal_path)
        nominal = nominal_path.read_text()
        residuals_ia = residuals.loc[
            (residuals["uncertainty_form"] == "inverse_input_multiplicative")
            & (residuals["load"] == load_bool)
            & (residuals["nominal_serial_no"] == nominal)
        ]
    else:
        residuals_ia = residuals.loc[
            (residuals["uncertainty_form"] == "inverse_input_multiplicative")
            & (residuals["load"] == load_bool)
        ]
    min_area = residuals_ia.loc[residuals_ia["peak_bound"].idxmin()]

    all = np.abs(np.array(min_area["residuals"]))
    bound = np.max(all, axis=0)

    fit_bound_arr = np.zeros(orders.shape, dtype=object)
    for i in range(fit_bound_arr.shape[0]):
        for j in range(fit_bound_arr.shape[1]):
            fit_bound_arr[i, j] = tf_cover.tf_cover(omega, bound[i, j, :], orders[i, j])
    fit_bound = _combine(fit_bound_arr)
    mag, _, _ = fit_bound.frequency_response(omega)

    fig, ax = plt.subplots(fit_bound_arr.shape[0], fit_bound_arr.shape[1])
    for i in range(fit_bound_arr.shape[0]):
        for j in range(fit_bound_arr.shape[1]):
            for residual in min_area["residuals"]:
                magnitude = 20 * np.log10(np.abs(residual))
                ax[i, j].semilogx(f, magnitude[i, j, :], ":k")
                ax[i, 0].set_ylabel(r"$|W(f)|$ (dB)")
                ax[-1, j].set_xlabel(r"$f$ (Hz)")
            ax[i, j].semilogx(f, 20 * np.log10(bound[i, j, :]), "r", lw=3, label="max")
            ax[i, j].semilogx(f, 20 * np.log10(mag[i, j, :]), "--b", lw=3, label="fit")
    fig.suptitle(f"{koopman} uncertainty bounds, {load}")
    for a in ax.ravel():
        a.grid(ls="--")
    uncertainty_mimo_path.parent.mkdir(exist_ok=True)
    fig.savefig(uncertainty_mimo_path)

    max_sv = min_area["bound"]
    max_sv_fit = np.array([scipy.linalg.svdvals(fit_bound(1j * w))[0] for w in omega])
    fig, ax = plt.subplots()
    ax.semilogx(f, 20 * np.log10(max_sv), "r", lw=3)
    ax.semilogx(f, 20 * np.log10(max_sv_fit), "--b", lw=3)
    ax.grid(ls="--")
    ax.set_xlabel(r"$f$ (Hz)")
    ax.set_ylabel(r"$\bar{\sigma}(W(f))$ (dB)")
    ax.set_title(f"{koopman} uncertainty bound, {load}")
    uncertainty_msv_path.parent.mkdir(exist_ok=True)
    fig.savefig(uncertainty_msv_path)

    nominal_serial_no = min_area["nominal_serial_no"]
    if koopman == "koopman":
        nominal_path.parent.mkdir(exist_ok=True)
        # joblib.dump(nominal_serial_no, nominal_path)
        nominal_path.write_text(nominal_serial_no)

    data = {
        "nominal_serial_no": nominal_serial_no,
        "bound": bound,
        "fit_bound": fit_bound,
        "t_step": t_step,
    }
    uncertainty_path.parent.mkdir(exist_ok=True)
    joblib.dump(data, uncertainty_path)


def _circular_mean(theta: np.ndarray) -> float:
    """Circular mean."""
    avg_sin = np.mean(np.sin(theta))
    avg_cos = np.mean(np.cos(theta))
    avg = np.arctan2(avg_sin, avg_cos)
    return avg


def _residuals(
    ss_nom,
    ss_list,
    t_step,
    f_plot,
    form="additive",
):
    """Compute residual frequency-by-frequency."""

    def _res(f, ss):
        """Compute residual at given frequency."""
        G = _transfer_matrix(f, ss_nom, t_step)
        G_p = _transfer_matrix(f, ss, t_step)
        if form == "additive":
            res = G_p - G
        elif form == "input_multiplicative":
            res = scipy.linalg.lstsq(G, G_p)[0] - np.eye(G_p.shape[1])
        elif form == "output_multiplicative":
            res = scipy.linalg.lstsq(G.T, G_p.T)[0].T - np.eye(G_p.shape[0])
        elif form == "inverse_additive":
            res = scipy.linalg.pinv(G) - scipy.linalg.pinv(G_p)
        elif form == "inverse_input_multiplicative":
            res = np.eye(G.shape[1]) - scipy.linalg.lstsq(G_p, G)[0]
        elif form == "inverse_output_multiplicative":
            res = np.eye(G.shape[0]) - scipy.linalg.lstsq(G_p.T, G.T)[0].T
        else:
            raise ValueError("Invalid `form`.")
        return res

    residuals = []
    magnitudes = []
    for ss in ss_list:
        # Generate frequency response data
        res = np.stack([_res(f_plot[k], ss) for k in range(f_plot.shape[0])], axis=-1)
        mag = np.array(
            [scipy.linalg.svdvals(res[:, :, k])[0] for k in range(res.shape[2])]
        )
        residuals.append(res)
        magnitudes.append(mag)
    # Compute max bound
    bound = np.max(np.vstack(magnitudes), axis=0)
    # Compute peak of max bound
    peak_bound = np.max(bound)
    area_bound = np.trapz(bound, x=f_plot)
    out = {
        "magnitudes": magnitudes,
        "residuals": residuals,
        "bound": bound,
        "peak_bound": peak_bound,
        "area_bound": area_bound,
    }
    return out


def _transfer_matrix(f, ss, t_step):
    """Compute a transfer matrix at a frequency."""
    z = np.exp(1j * 2 * np.pi * f * t_step)
    G = ss.C @ scipy.linalg.solve((np.diag([z] * ss.A.shape[0]) - ss.A), ss.B) + ss.D
    return G


def _combine(G):
    """Combine arraylike of transfer functions into a MIMO TF."""
    G = np.array(G)
    num = []
    den = []
    for i_out in range(G.shape[0]):
        for j_out in range(G[i_out, 0].noutputs):
            num_row = []
            den_row = []
            for i_in in range(G.shape[1]):
                for j_in in range(G[i_out, i_in].ninputs):
                    num_row.append(G[i_out, i_in].num[j_out][j_in])
                    den_row.append(G[i_out, i_in].den[j_out][j_in])
            num.append(num_row)
            den.append(den_row)
    G_tf = control.TransferFunction(num, den, dt=G[0][0].dt)
    return G_tf
