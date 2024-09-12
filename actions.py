"""Define and automate tasks with ``doit``."""

import itertools
import pathlib
import re
import shutil

import control
import joblib
import numpy as np
import pandas
import pykoop
import scipy.linalg
from cmcrameri import cm as cmc
from matplotlib import pyplot as plt

import obs_syn
import onesine
import tf_cover

# Number of training episodes
N_TRAIN = 18

# Okabe-Ito colorscheme: https://jfly.uni-koeln.de/color/
OKABE_ITO = {
    "black": (0.00, 0.00, 0.00),
    "orange": (0.90, 0.60, 0.00),
    "sky blue": (0.35, 0.70, 0.90),
    "bluish green": (0.00, 0.60, 0.50),
    "yellow": (0.95, 0.90, 0.25),
    "blue": (0.00, 0.45, 0.70),
    "vermillion": (0.80, 0.40, 0.00),
    "reddish purple": (0.80, 0.60, 0.70),
    "grey": (0.60, 0.60, 0.60),
}

# LaTeX linewidth (inches)
LW = 3.5

# ``fig.savefig()`` settings
SAVEFIG_KW = {
    "bbox_inches": "tight",
    "pad_inches": 0.05,
}

# Set gobal Matplotlib options
plt.rc("lines", linewidth=1.5)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--")
# Set LaTeX rendering only if available
usetex = True if shutil.which("latex") else False
if usetex:
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("font", size=9)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}")


def action_preprocess_experiments(
    raw_dataset_path: pathlib.Path,
    preprocessed_dataset_path: pathlib.Path,
):
    """Preprocess raw data into pickle containing Pandas ``DataFrame``."""
    preprocessed_dataset_path.parent.mkdir(parents=True, exist_ok=True)
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
    joblib.dump(merged_df, preprocessed_dataset_path)


def action_compute_phase(
    dataset_path: pathlib.Path,
    phase_path: pathlib.Path,
):
    """Compute phase offset."""
    phase_path.parent.mkdir(parents=True, exist_ok=True)
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
    joblib.dump(df, phase_path)


def action_id_models(
    dataset_path: pathlib.Path,
    phase_path: pathlib.Path,
    models_path: pathlib.Path,
    koopman: str,
):
    """Identify linear and Koopman models."""
    models_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = joblib.load(dataset_path)
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
        X_train = X.loc[X["episode"] < N_TRAIN].to_numpy()
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
    joblib.dump(df, models_path)


def action_compute_residuals(
    models_path: pathlib.Path,
    residuals_path: pathlib.Path,
):
    """Compute residuals from linear and Koopman models."""
    residuals_path.parent.mkdir(parents=True, exist_ok=True)
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
    uncertainty_path.parent.mkdir(parents=True, exist_ok=True)
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
    fig.savefig(uncertainty_msv_path)

    nominal_serial_no = min_area["nominal_serial_no"]
    if koopman == "koopman":
        # joblib.dump(nominal_serial_no, nominal_path)
        nominal_path.write_text(nominal_serial_no)

    data = {
        "nominal_serial_no": nominal_serial_no,
        "bound": bound,
        "fit_bound": fit_bound,
        "t_step": t_step,
    }
    joblib.dump(data, uncertainty_path)


def action_synthesize_observer(
    dataset_path: pathlib.Path,
    models_path: pathlib.Path,
    uncertainty_path: pathlib.Path,
    observer_path: pathlib.Path,
    weight_plot_path: pathlib.Path,
    traj_plot_path: pathlib.Path,
    err_plot_path: pathlib.Path,
    fft_plot_path: pathlib.Path,
    koopman: str,
):
    """Synthesize observer."""
    observer_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = joblib.load(dataset_path)
    models = joblib.load(models_path)
    uncertainty = joblib.load(uncertainty_path)
    t_step = models.attrs["t_step"]
    nom_sn = uncertainty["nominal_serial_no"]
    # Results dictionary
    results = {}
    # Generalized plant weights
    if koopman == "koopman":
        W_p = control.StateSpace([], [], [], np.diag([1, 1, 1, 0]), dt=t_step)
    else:
        W_p = control.StateSpace([], [], [], np.diag([1, 1, 1]), dt=t_step)
    W_u = control.StateSpace([], [], [], np.diag([1, 1]), dt=t_step)
    W_D = control.tf2ss(uncertainty["fit_bound"]).sample(t_step)
    # Save weight magnitudes
    f = np.logspace(-3, np.log10(0.5 / t_step), 1000)
    omega = 2 * np.pi * f
    mag_p = _max_sv(W_p, f, t_step)
    mag_u = _max_sv(W_u, f, t_step)
    mag_D = _max_sv(W_D, f, t_step)
    results["f"] = f
    results["omega"] = omega
    results["mag_p"] = mag_p
    results["mag_u"] = mag_u
    results["mag_D"] = mag_D
    # Plot weights
    fig, ax = plt.subplots()
    ax.semilogx(f, 20 * np.log10(mag_p), label=r"$W_\mathrm{p}$")
    ax.semilogx(f, 20 * np.log10(mag_u), "--", label=r"$W_\mathrm{u}$")
    ax.semilogx(f, 20 * np.log10(mag_D), label=r"$W_\Delta$")
    # Get nominal model and Koopman pipeline
    P_0_ = control.StateSpace(
        *models.loc[
            (models["serial_no"] == nom_sn) & (~models["load"]), "state_space"
        ].item()
    )
    kp = models.loc[
        (models["serial_no"] == nom_sn) & (~models["load"]), "koopman_pipeline"
    ].item()
    # Update ``C`` matrix
    if koopman == "koopman":
        P_0 = control.StateSpace(
            P_0_.A,
            P_0_.B,
            np.array(
                [
                    [1, 0, 0, 0],
                ]
            ),
            np.array([[0, 0]]),
            P_0_.dt,
        )
    else:
        P_0 = control.StateSpace(
            P_0_.A,
            P_0_.B,
            np.array(
                [
                    [1, 0, 0],
                ]
            ),
            np.array([[0, 0]]),
            P_0_.dt,
        )
    # Set number of inputs and outputs for generalized plant
    n_z2 = 2
    n_w2 = 2
    n_y = 1
    n_u = 2
    # Create generalized plant state-space matrices
    F_A = np.block(
        [
            [
                P_0.A,
                np.zeros((P_0.nstates, P_0.nstates)),
                np.zeros((P_0.nstates, W_p.nstates)),
                P_0.B @ W_u.C,
                P_0.B @ W_D.C,
            ],
            [
                np.zeros((P_0.nstates, P_0.nstates)),
                P_0.A,
                np.zeros((P_0.nstates, W_p.nstates)),
                P_0.B @ W_u.C,
                np.zeros((P_0.nstates, W_D.nstates)),
            ],
            [
                W_p.B,
                -1 * W_p.B,
                W_p.A,
                np.zeros((W_p.nstates, W_u.nstates)),
                np.zeros((W_p.nstates, W_D.nstates)),
            ],
            [
                np.zeros((W_u.nstates, P_0.nstates)),
                np.zeros((W_u.nstates, P_0.nstates)),
                np.zeros((W_u.nstates, W_p.nstates)),
                W_u.A,
                np.zeros((W_u.nstates, W_D.nstates)),
            ],
            [
                np.zeros((W_D.nstates, P_0.nstates)),
                np.zeros((W_D.nstates, P_0.nstates)),
                np.zeros((W_D.nstates, W_p.nstates)),
                np.zeros((W_D.nstates, W_u.nstates)),
                W_D.A,
            ],
        ]
    )
    F_B = np.block(
        [
            [
                P_0.B @ W_u.D,
                P_0.B @ W_D.D,
                np.zeros((P_0.nstates, P_0.ninputs)),
            ],
            [
                P_0.B @ W_u.D,
                np.zeros((P_0.nstates, W_D.ninputs)),
                P_0.B,
            ],
            [
                np.zeros((W_p.nstates, W_u.ninputs)),
                np.zeros((W_p.nstates, W_D.ninputs)),
                np.zeros((W_p.nstates, P_0.ninputs)),
            ],
            [
                W_u.B,
                np.zeros((W_u.nstates, W_D.ninputs)),
                np.zeros((W_u.nstates, P_0.ninputs)),
            ],
            [
                np.zeros((W_D.nstates, W_u.ninputs)),
                W_D.B,
                np.zeros((W_D.nstates, P_0.ninputs)),
            ],
        ]
    )
    F_C = np.block(
        [
            [
                W_p.D,
                -1 * W_p.D,
                W_p.C,
                np.zeros((W_p.noutputs, W_u.nstates)),
                np.zeros((W_p.noutputs, W_D.nstates)),
            ],
            [
                np.zeros((W_D.noutputs, P_0.nstates)),
                np.zeros((W_D.noutputs, P_0.nstates)),
                np.zeros((W_D.noutputs, W_p.nstates)),
                W_u.C,
                W_D.C,
            ],
            [
                P_0.C,
                -1 * P_0.C,
                np.zeros((P_0.noutputs, W_p.nstates)),
                np.zeros((P_0.noutputs, W_u.nstates)),
                np.zeros((P_0.noutputs, W_D.nstates)),
            ],
        ]
    )
    F_D = np.block(
        [
            [
                np.zeros((W_p.noutputs, W_u.ninputs)),
                np.zeros((W_p.noutputs, W_D.ninputs)),
                np.zeros((W_p.noutputs, P_0.ninputs)),
            ],
            [
                W_u.D,
                W_D.D,
                np.zeros((W_u.ninputs, P_0.ninputs)),
            ],
            [
                np.zeros((P_0.noutputs, W_u.ninputs)),
                np.zeros((P_0.noutputs, W_D.ninputs)),
                np.zeros((P_0.noutputs, P_0.ninputs)),
            ],
        ]
    )
    F = control.StateSpace(F_A, F_B, F_C, F_D, t_step)
    # Save magnitude responses of generalized plant and nominal plant
    mag_F = _max_sv(F, f, t_step)
    mag_P = _max_sv(P_0, f, t_step)
    results["mag_P"] = mag_P
    results["mag_F"] = mag_F
    results["F"] = (F_A, F_B, F_C, F_D, t_step)
    # Add magnitudes to plot and save it
    ax.semilogx(f, 20 * np.log10(mag_F), label=r"$G$")
    ax.semilogx(f, 20 * np.log10(mag_P), label=r"$P$")
    ax.grid(ls="--")
    ax.legend(loc="lower right")
    fig.savefig(weight_plot_path)
    # Synthesize controller
    K, info = obs_syn.mixed_H2_Hinf(F, n_z2, n_w2, n_y, n_u, initial_guess=None)
    if K is None:
        raise RuntimeError(
            f"Could not find solution to mixed H2-Hinf problem: {info['status']}"
        )
    # Save synthesis results
    results["K"] = (K.A, K.B, K.C, K.D, t_step)
    results["P"] = (P_0.A, P_0.B, P_0.C, P_0.D, t_step)
    results["synthesis_info"] = info
    # Load dataset to test observer
    dataset_sn_noload = dataset.loc[
        (dataset["serial_no"] == nom_sn) & (~dataset["load"])
    ]
    X = dataset_sn_noload[
        [
            "episode",
            "joint_pos",
            "joint_vel",
            "joint_trq",
            "target_joint_pos",
            "target_joint_vel",
        ]
    ]
    X_valid_ = X.loc[X["episode"] >= N_TRAIN].to_numpy()
    X_valid = pykoop.split_episodes(
        X_valid_,
        episode_feature=True,
    )[0][1]
    # Lift measurements (for linear model this will do nothing)
    meas = kp.lift_state(X_valid[:, :3], episode_feature=False).T
    inpt = X_valid[:, 3:5].T
    t = np.arange(inpt.shape[1]) * t_step
    # Form closed-loop system to check stability
    A_cl = np.block(
        [
            [
                P_0.A - P_0.B @ K.D @ P_0.C,
                P_0.B @ K.C,
            ],
            [
                -K.B @ P_0.C,
                K.A,
            ],
        ]
    )
    evs = scipy.linalg.eigvals(A_cl)
    violation = np.max(np.abs(evs)) - 1
    if np.any(np.abs(evs) >= 1):
        raise RuntimeError(f"Unstable closed-loop by {violation}.")
    # Check stability with all off-nominal models
    for i, model in models.groupby(by=["serial_no", "load"]):
        A, B, _, _, _ = model["state_space"].item()
        A_cl = np.block(
            [
                [
                    A - B @ K.D @ P_0.C,
                    B @ K.C,
                ],
                [
                    -K.B @ P_0.C,
                    K.A,
                ],
            ]
        )
        evs = scipy.linalg.eigvals(A_cl)
        violation = np.max(np.abs(evs)) - 1
        if np.any(np.abs(evs) >= 1):
            raise RuntimeError(f"{i} unstable closed-loop by {violation}.")
    # Simulate observer
    X = np.zeros((P_0.nstates, t.shape[0]))
    X[:, [0]] = kp.lift_state(np.zeros((1, 3)), episode_feature=False).T
    Xc = np.zeros((K.nstates, t.shape[0]))
    for k in range(1, t.shape[0] + 1):
        # Compute error first since 0 has no D matrix
        err = P_0.C @ meas[:, k - 1] - P_0.C @ X[:, k - 1]
        # Compute control output
        u = K.C @ Xc[:, k - 1] + K.D @ err
        if k < X.shape[1]:
            # Update plant with control input
            if koopman == "linear":
                X[:, k] = P_0.A @ X[:, k - 1] + P_0.B @ (inpt[:, k - 1] + u)
            else:
                Xt_ret = kp.retract_state(X[:, [k - 1]].T, episode_feature=False)
                X_rl = kp.lift_state(Xt_ret, episode_feature=False).T.ravel()
                X[:, k] = P_0.A @ X_rl + P_0.B @ (inpt[:, k - 1] + u)
            # Update controller
            Xc[:, k] = K.A @ Xc[:, k - 1] + K.B @ err
    # Plot trajectories
    fig, ax = plt.subplots(meas.shape[0] + inpt.shape[0], 1, sharex=True)
    for i in range(meas.shape[0]):
        ax[i].plot(meas[i, :], label="True")
        ax[i].plot(X[i, :], "--", label="Estimate")
    if koopman == "koopman":
        ax[4].plot(inpt[0, :])
        ax[5].plot(inpt[1, :])
        ax[3].set_ylabel(r"$\sin{\theta}$")
        ax[4].set_ylabel(r"$\theta$ ref. (rad)")
        ax[5].set_ylabel(r"$\dot{\theta}$ ref.(rad/s)")
        ax[5].set_xlabel(r"k")
    else:
        ax[3].plot(inpt[0, :])
        ax[4].plot(inpt[1, :])
        ax[3].set_ylabel(r"$\theta$ ref. (rad)")
        ax[4].set_ylabel(r"$\dot{\theta}$ ref.(rad/s)")
        ax[4].set_xlabel(r"k")
    ax[0].legend(loc="lower right")
    for a in ax.ravel():
        a.grid(ls="--")
    ax[0].set_ylabel(r"$\theta$ (rad)")
    ax[1].set_ylabel(r"$\dot{\theta}$ (rad/s)")
    ax[2].set_ylabel(r"$\tau$ (pct)")
    fig.suptitle(f"{koopman} prediction")
    fig.savefig(traj_plot_path)
    # Plot trajectory errors
    fig, ax = plt.subplots(meas.shape[0], 1, sharex=True)
    for i in range(meas.shape[0]):
        ax[i].plot(meas[i, :] - X[i, :])
    for a in ax.ravel():
        a.grid(ls="--")
    ax[0].set_ylabel(r"$\Delta\theta$ (rad)")
    ax[1].set_ylabel(r"$\Delta\dot{\theta}$ (rad/s)")
    ax[2].set_ylabel(r"$\Delta\tau$ (pct)")
    if koopman == "koopman":
        ax[3].set_ylabel(r"$\Delta\sin{\theta}$")
        ax[3].set_xlabel(r"k")
    else:
        ax[2].set_xlabel(r"k")
    fig.suptitle(f"{koopman} error")
    fig.savefig(err_plot_path)
    # Plot trajectory error FFTs
    f = scipy.fft.rfftfreq(meas.shape[1], t_step)
    fft_err = scipy.fft.rfft(meas - X, norm="forward")
    fig, ax = plt.subplots(meas.shape[0], 1, sharex=True)
    for i in range(meas.shape[0]):
        ax[i].plot(f, np.abs(fft_err[i, :]))
    for a in ax.ravel():
        a.grid(ls="--")
    ax[0].set_ylabel(r"$\Delta\theta$ (rad)")
    ax[1].set_ylabel(r"$\Delta\dot{\theta}$ (rad/s)")
    ax[2].set_ylabel(r"$\Delta\tau$ (pct)")
    if koopman == "koopman":
        ax[3].set_ylabel(r"$\Delta\sin{\theta}$")
        ax[3].set_xlabel(r"$f$ (Hz)")
    else:
        ax[2].set_xlabel(r"$f$ (Hz)")
    fig.savefig(fft_plot_path)
    # Save results
    joblib.dump(results, observer_path)


def action_plot_fft(
    dataset_path: pathlib.Path,
    error_fft_path: pathlib.Path,
):
    """Plot error FFT."""
    error_fft_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = joblib.load(dataset_path)
    # Settings
    t_step = dataset.attrs["t_step"]
    min_length = 600
    min_vel = 3
    trim = 100
    dataset_ep = dataset.loc[
        (dataset["serial_no"] == "009017")
        & (~dataset["load"])
        & (dataset["episode"] == 0)
    ]
    tvel = dataset_ep["target_joint_vel"].to_numpy()
    vel = dataset_ep["joint_vel"].to_numpy()
    # Find points where velocity changes
    vel_changes = np.ravel(np.argwhere(np.diff(tvel, prepend=0) != 0))
    # Split into constant-velocity segments
    X = np.vstack(
        [
            vel,
            tvel,
        ]
    ).T
    const_vel_segments = np.split(X, vel_changes)
    # Find first segment of required length and speed
    for segment in const_vel_segments:
        X_const_vel = segment[trim:-trim, :]
        if segment.shape[0] > min_length:
            if np.all(segment[:, 1] > min_vel):
                # Compute normalized velocity error in that segment
                vel_err = X_const_vel[:, 1] - X_const_vel[:, 0]
                f, pos_err_spec = scipy.signal.welch(
                    vel_err,
                    fs=(1 / t_step),
                    nperseg=512,
                )
                fig, ax = plt.subplots(
                    constrained_layout=True,
                    figsize=(LW, LW),
                )
                ax.semilogy(f, pos_err_spec, color=OKABE_ITO["blue"])
                ax.set_yticks([10**i for i in range(-9, -3)])
                ax.set_xticks(np.arange(0, 550, 50))
                ax.set_xlabel(r"$f$ (Hz)")
                ax.set_ylabel(
                    r"$S_{\dot{\theta}^\mathrm{e}\dot{\theta}^\mathrm{e}}(f)$ "
                    r"($\mathrm{rad}^2/\mathrm{s}^2/\mathrm{Hz}$)"
                )
                break
    fig.savefig(
        error_fft_path,
        **SAVEFIG_KW,
    )


def action_plot_model_predictions(
    dataset_path: pathlib.Path,
    models_linear_path: pathlib.Path,
    models_koopman_path: pathlib.Path,
    pred_ref_path: pathlib.Path,
    pred_traj_path: pathlib.Path,
    pred_err_path: pathlib.Path,
    pred_fft_path: pathlib.Path,
):
    """Plot model predictions."""
    pred_ref_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = joblib.load(dataset_path)
    models_linear = joblib.load(models_linear_path)
    models_koopman = joblib.load(models_koopman_path)

    sn = "009017"
    t_step = dataset.attrs["t_step"]

    dataset_sn_noload = dataset.loc[(dataset["serial_no"] == sn) & (~dataset["load"])]
    X = dataset_sn_noload[
        [
            "episode",
            "joint_pos",
            "joint_vel",
            "joint_trq",
            "target_joint_pos",
            "target_joint_vel",
        ]
    ]
    X_test = X.loc[X["episode"] == N_TRAIN].to_numpy()
    t = np.arange(X_test.shape[0]) * t_step
    kp_linear = models_linear.loc[
        (models_linear["serial_no"] == sn) & (~models_linear["load"]),
        "koopman_pipeline",
    ].item()
    kp_koopman = models_koopman.loc[
        (models_koopman["serial_no"] == sn) & (~models_koopman["load"]),
        "koopman_pipeline",
    ].item()
    Xp_linear = kp_linear.predict_trajectory(X_test)
    Xp_koopman = kp_koopman.predict_trajectory(X_test)

    fig, ax = plt.subplots(
        2,
        1,
        constrained_layout=True,
        figsize=(LW, LW),
        sharex=True,
    )
    ax[0].plot(t, X_test[:, 4], color=OKABE_ITO["black"], label="Measured")
    ax[1].plot(t, X_test[:, 5], color=OKABE_ITO["black"])
    ax[0].set_ylabel(r"$\theta^\mathrm{r}(t)$ (rad)")
    ax[1].set_ylabel(r"$\dot{\theta}^\mathrm{r}(t)$ (rad/s)")
    ax[1].set_xlabel(r"$t$ (s)")
    fig.align_ylabels()
    fig.savefig(
        pred_ref_path,
        **SAVEFIG_KW,
    )

    fig, ax = plt.subplots(
        3,
        1,
        constrained_layout=True,
        figsize=(LW, LW),
        sharex=True,
    )
    ax[0].plot(t, X_test[:, 1], color=OKABE_ITO["black"], label="Measured")
    ax[1].plot(t, X_test[:, 2], color=OKABE_ITO["black"])
    ax[2].plot(t, X_test[:, 3], color=OKABE_ITO["black"])
    ax[0].plot(t, Xp_koopman[:, 1], color=OKABE_ITO["blue"], label="Koopman")
    ax[1].plot(t, Xp_koopman[:, 2], color=OKABE_ITO["blue"])
    ax[2].plot(t, Xp_koopman[:, 3], color=OKABE_ITO["blue"])
    ax[0].plot(t, Xp_linear[:, 1], color=OKABE_ITO["vermillion"], label="Linear")
    ax[1].plot(t, Xp_linear[:, 2], color=OKABE_ITO["vermillion"])
    ax[2].plot(t, Xp_linear[:, 3], color=OKABE_ITO["vermillion"])
    # Inset axis 1
    axins1 = ax[1].inset_axes(
        [0.5, 0.5, 0.48, 0.48],
        xlim=(1.5, 1.8),
        ylim=(3.05, 3.20),
    )
    axins1.plot(t, X_test[:, 2], color=OKABE_ITO["black"])
    axins1.plot(t, Xp_koopman[:, 2], color=OKABE_ITO["blue"])
    axins1.plot(t, Xp_linear[:, 2], color=OKABE_ITO["vermillion"])
    ax[1].indicate_inset_zoom(axins1, edgecolor="black")
    # Inset axis 2
    axins2 = ax[2].inset_axes(
        [0.5, 0.5, 0.48, 0.48],
        xlim=(1.5, 1.8),
        ylim=(-0.05, 0.25),
    )
    axins2.plot(t, X_test[:, 3], color=OKABE_ITO["black"])
    axins2.plot(t, Xp_koopman[:, 3], color=OKABE_ITO["blue"])
    axins2.plot(t, Xp_linear[:, 3], color=OKABE_ITO["vermillion"])
    ax[2].indicate_inset_zoom(axins2, edgecolor="black")
    # Labels
    ax[0].set_ylabel(r"$\theta(t)$ (rad)")
    ax[1].set_ylabel(r"$\dot{\theta}(t)$ (rad/s)")
    ax[2].set_ylabel(r"$i(t)$ (unitless)")
    ax[2].set_xlabel(r"$t$ (s)")
    fig.align_ylabels()
    fig.legend(
        handles=[
            ax[0].get_lines()[0],
            ax[0].get_lines()[2],
            ax[0].get_lines()[1],
        ],
        loc="upper center",
        ncol=3,
        handlelength=1,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.savefig(
        pred_traj_path,
        **SAVEFIG_KW,
    )

    fig, ax = plt.subplots(
        3,
        1,
        constrained_layout=True,
        figsize=(LW, LW),
        sharex=True,
    )
    ax[0].plot(
        t,
        _percent_error(X_test[:, 1], Xp_linear[:, 1]),
        color=OKABE_ITO["vermillion"],
        label="Linear",
    )
    ax[1].plot(
        t,
        _percent_error(X_test[:, 2], Xp_linear[:, 2]),
        color=OKABE_ITO["vermillion"],
    )
    ax[2].plot(
        t,
        _percent_error(X_test[:, 3], Xp_linear[:, 3]),
        color=OKABE_ITO["vermillion"],
    )
    ax[0].plot(
        t,
        _percent_error(X_test[:, 1], Xp_koopman[:, 1]),
        color=OKABE_ITO["blue"],
        label="Koopman",
    )
    ax[1].plot(
        t,
        _percent_error(X_test[:, 2], Xp_koopman[:, 2]),
        color=OKABE_ITO["blue"],
    )
    ax[2].plot(
        t,
        _percent_error(X_test[:, 3], Xp_koopman[:, 3]),
        color=OKABE_ITO["blue"],
    )
    axins1 = ax[1].inset_axes(
        [0.5, 0.5, 0.48, 0.48],
        xlim=(0, 0.5),
        ylim=(-9.5, 9.5),
    )
    axins1.plot(
        t, _percent_error(X_test[:, 2], Xp_linear[:, 2]), color=OKABE_ITO["vermillion"]
    )
    axins1.plot(
        t, _percent_error(X_test[:, 2], Xp_koopman[:, 2]), color=OKABE_ITO["blue"]
    )
    ax[1].indicate_inset_zoom(axins1, edgecolor="black")
    axins2 = ax[2].inset_axes(
        [0.5, 0.5, 0.48, 0.48],
        xlim=(0, 0.5),
        ylim=(-95, 95),
    )
    axins2.plot(
        t, _percent_error(X_test[:, 3], Xp_linear[:, 3]), color=OKABE_ITO["vermillion"]
    )
    axins2.plot(
        t, _percent_error(X_test[:, 3], Xp_koopman[:, 3]), color=OKABE_ITO["blue"]
    )
    ax[2].indicate_inset_zoom(axins2, edgecolor="black")
    ax[1].set_yticks([-10, -5, 0, 5, 10])
    ax[2].set_yticks([-100, -50, 0, 50, 100])
    ax[0].set_ylabel(r"$\Delta\theta(t)$ (\%)")
    ax[1].set_ylabel(r"$\Delta\dot{\theta}(t)$ (\%)")
    ax[2].set_ylabel(r"$\Delta i(t)$ (\%)")
    ax[2].set_xlabel(r"$t$ (s)")
    fig.align_ylabels()
    fig.legend(
        handles=[
            ax[0].get_lines()[0],
            ax[0].get_lines()[1],
        ],
        loc="upper center",
        ncol=3,
        handlelength=1,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.savefig(
        pred_err_path,
        **SAVEFIG_KW,
    )

    fig, ax = plt.subplots(
        3,
        1,
        constrained_layout=True,
        figsize=(LW, LW),
        sharex=True,
    )
    ax[0].plot(
        *_psd_error(X_test[:, 1], Xp_linear[:, 1], t_step),
        color=OKABE_ITO["vermillion"],
        label="Linear",
    )
    ax[0].plot(
        *_psd_error(X_test[:, 1], Xp_koopman[:, 1], t_step),
        color=OKABE_ITO["blue"],
        label="Koopman",
    )
    ax[1].plot(
        *_psd_error(X_test[:, 2], Xp_linear[:, 2], t_step),
        color=OKABE_ITO["vermillion"],
        label="Linear",
    )
    ax[1].plot(
        *_psd_error(X_test[:, 2], Xp_koopman[:, 2], t_step),
        color=OKABE_ITO["blue"],
        label="Koopman",
    )
    ax[2].plot(
        *_psd_error(X_test[:, 3], Xp_linear[:, 3], t_step),
        color=OKABE_ITO["vermillion"],
        label="Linear",
    )
    ax[2].plot(
        *_psd_error(X_test[:, 3], Xp_koopman[:, 3], t_step),
        color=OKABE_ITO["blue"],
        label="Koopman",
    )
    ax[2].set_xlabel(r"$f$ (Hz)")
    ax[0].set_ylabel(
        r"$S_{\theta^\mathrm{e}\theta^\mathrm{e}}(f)$ "
        "\n"
        r"($\mathrm{rad}^2/\mathrm{Hz}$)"
    )
    ax[1].set_ylabel(
        r"$S_{\dot{\theta}^\mathrm{e}\dot{\theta}^\mathrm{e}}(f)$ "
        "\n"
        r"($\mathrm{rad}^2/\mathrm{s}^2/\mathrm{Hz}$)"
    )
    ax[2].set_ylabel(r"$S_{i^\mathrm{e}i^\mathrm{e}}(f)$" "\n" r"($1/\mathrm{Hz}$)")
    fig.align_ylabels()
    fig.legend(
        handles=[
            ax[0].get_lines()[0],
            ax[0].get_lines()[1],
        ],
        loc="upper center",
        ncol=3,
        handlelength=1,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.savefig(
        pred_fft_path,
        **SAVEFIG_KW,
    )


def action_plot_model_tfs(
    dataset_path: pathlib.Path,
    models_linear_path: pathlib.Path,
    models_koopman_path: pathlib.Path,
    tfs_msv_linear_path: pathlib.Path,
    tfs_msv_koopman_path: pathlib.Path,
    tfs_mimo_linear_path: pathlib.Path,
    tfs_mimo_koopman_path: pathlib.Path,
):
    """Plot model transfer functions."""
    tfs_msv_linear_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = joblib.load(dataset_path)
    models_linear = joblib.load(models_linear_path)
    models_koopman = joblib.load(models_koopman_path)
    t_step = dataset.attrs["t_step"]
    f = np.logspace(-3, np.log10(0.5 / t_step), 1000)
    color = cmc.batlowS(np.linspace(0, 1, models_linear.shape[0]))

    fig, ax = plt.subplots(
        constrained_layout=True,
        figsize=(LW, LW),
    )
    for i, sn in enumerate(models_linear["serial_no"]):
        ss_ = models_linear.loc[
            (models_linear["serial_no"] == sn) & (~models_linear["load"]), "state_space"
        ]
        if len(ss_) > 0:
            ss_ = ss_.item()
        else:
            continue
        ss = control.StateSpace(*ss_)
        G = np.array([_transfer_matrix(f_, ss, t_step) for f_ in f])
        mag = np.array([scipy.linalg.svdvals(G[k, :, :])[0] for k in range(G.shape[0])])
        ax.semilogx(f, 20 * np.log10(mag), color=color[i], label=sn)
    ax.set_xlabel(r"$f$ (Hz)")
    ax.set_ylabel(r"$\bar{\sigma}\left({\bf G}(f)\right)$ (dB)")
    ax.set_ylim(-10, 20)
    fig.savefig(
        tfs_msv_linear_path,
        **SAVEFIG_KW,
    )

    fig, ax = plt.subplots(
        constrained_layout=True,
        figsize=(LW, LW),
    )
    for i, sn in enumerate(models_koopman["serial_no"]):
        ss_ = models_koopman.loc[
            (models_koopman["serial_no"] == sn) & (~models_koopman["load"]),
            "state_space",
        ]
        if len(ss_) > 0:
            ss_ = ss_.item()
        else:
            continue
        ss = control.StateSpace(*ss_)
        G = np.array([_transfer_matrix(f_, ss, t_step) for f_ in f])
        mag = np.array([scipy.linalg.svdvals(G[k, :, :])[0] for k in range(G.shape[0])])
        ax.semilogx(f, 20 * np.log10(mag), color=color[i], label=sn)
    ax.set_xlabel(r"$f$ (Hz)")
    ax.set_ylabel(r"$\bar{\sigma}\left({\bf G}(f)\right)$ (dB)")
    ax.set_ylim(-10, 20)
    fig.savefig(
        tfs_msv_koopman_path,
        **SAVEFIG_KW,
    )

    fig, ax = plt.subplots(
        3,
        2,
        constrained_layout=True,
        figsize=(2 * LW, 1.5 * LW),
        sharex=True,
        sharey=True,
    )
    for i, sn in enumerate(models_linear["serial_no"]):
        ss_ = models_linear.loc[
            (models_linear["serial_no"] == sn) & (~models_linear["load"]), "state_space"
        ]
        if len(ss_) > 0:
            ss_ = ss_.item()
        else:
            continue
        ss = control.StateSpace(*ss_)
        mag, _, _ = ss.frequency_response(2 * np.pi * f)
        for j in range(mag.shape[0]):
            for k in range(mag.shape[1]):
                ax[j, k].semilogx(
                    f,
                    20 * np.log10(mag[j, k, :]),
                    color=color[i],
                    label=sn,
                )
                ax[-1, k].set_xlabel(r"$f$ (Hz)")
                ax[j, k].set_ylabel(rf"$|G_{{{j + 1}{k + 1}}}(f)|$ (dB)")
    fig.savefig(
        tfs_mimo_linear_path,
        **SAVEFIG_KW,
    )

    fig, ax = plt.subplots(
        4,
        2,
        constrained_layout=True,
        figsize=(2 * LW, 2 * LW),
        sharex=True,
        sharey=True,
    )
    for i, sn in enumerate(models_koopman["serial_no"]):
        ss_ = models_koopman.loc[
            (models_koopman["serial_no"] == sn) & (~models_koopman["load"]),
            "state_space",
        ]
        if len(ss_) > 0:
            ss_ = ss_.item()
        else:
            continue
        ss = control.StateSpace(*ss_)
        mag, _, _ = ss.frequency_response(2 * np.pi * f)
        for j in range(mag.shape[0]):
            for k in range(mag.shape[1]):
                ax[j, k].semilogx(
                    f,
                    20 * np.log10(mag[j, k, :]),
                    color=color[i],
                    label=sn,
                )
                ax[-1, k].set_xlabel(r"$f$ (Hz)")
                ax[j, k].set_ylabel(rf"$|G_{{{j + 1}{k + 1}}}(f)|$ (dB)")
    fig.savefig(
        tfs_mimo_koopman_path,
        **SAVEFIG_KW,
    )


def action_plot_observer(
    dataset_path: pathlib.Path,
    uncertainty_linear_path: pathlib.Path,
    uncertainty_koopman_path: pathlib.Path,
    models_linear_path: pathlib.Path,
    models_koopman_path: pathlib.Path,
    observer_linear_path: pathlib.Path,
    observer_koopman_path: pathlib.Path,
    obs_weights_linear_path: pathlib.Path,
    obs_weights_koopman_path: pathlib.Path,
    obs_nom_noload_traj_path: pathlib.Path,
    obs_nom_noload_err_path: pathlib.Path,
    obs_nom_noload_psd_path: pathlib.Path,
    obs_nom_load_traj_path: pathlib.Path,
    obs_nom_load_err_path: pathlib.Path,
    obs_nom_load_psd_path: pathlib.Path,
    obs_offnom_noload_traj_path: pathlib.Path,
    obs_offnom_noload_err_path: pathlib.Path,
    obs_offnom_noload_psd_path: pathlib.Path,
):
    """Plot observer."""
    obs_weights_linear_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = joblib.load(dataset_path)
    uncertainty_linear = joblib.load(uncertainty_linear_path)
    # Currently unused but may be used later
    # uncertainty_koopman = joblib.load(uncertainty_koopman_path)
    # models_linear = joblib.load(models_linear_path)
    models_koopman = joblib.load(models_koopman_path)
    observer_linear = joblib.load(observer_linear_path)
    observer_koopman = joblib.load(observer_koopman_path)
    t_step = dataset.attrs["t_step"]

    # Plot weights for linear and Koopman observers
    fig, ax = _plot_weights(observer_linear)
    fig.savefig(
        obs_weights_linear_path,
        **SAVEFIG_KW,
    )
    fig, ax = _plot_weights(observer_koopman)
    fig.savefig(
        obs_weights_koopman_path,
        **SAVEFIG_KW,
    )

    # Nominal, no load
    nom_sn = uncertainty_linear["nominal_serial_no"]
    X_valid_noload = dataset.loc[
        (dataset["serial_no"] == nom_sn)
        & (dataset["episode"] == N_TRAIN)
        & (~dataset["load"]),
        [
            "joint_pos",
            "joint_vel",
            "joint_trq",
            "target_joint_pos",
            "target_joint_vel",
        ],
    ].to_numpy()
    t = np.arange(X_valid_noload.shape[0]) * t_step
    x0 = np.array([[0], [0], [0]])
    X_obs_linear_noload = _simulate_linear(
        control.StateSpace(*observer_linear["P"]),
        control.StateSpace(*observer_linear["K"]),
        X_valid_noload,
        x0=x0,
    )
    kp = models_koopman.loc[
        (models_koopman["serial_no"] == nom_sn) & (~models_koopman["load"]),
        "koopman_pipeline",
    ].item()
    X_obs_koopman_noload = _simulate_koopman(
        control.StateSpace(*observer_koopman["P"]),
        control.StateSpace(*observer_koopman["K"]),
        X_valid_noload,
        kp,
        x0=x0,
    )
    fig, ax = _plot_traj(t, X_valid_noload, X_obs_linear_noload, X_obs_koopman_noload)
    fig.savefig(
        obs_nom_noload_traj_path,
        **SAVEFIG_KW,
    )
    fig, ax = _plot_err(t, X_valid_noload, X_obs_linear_noload, X_obs_koopman_noload)
    fig.savefig(
        obs_nom_noload_err_path,
        **SAVEFIG_KW,
    )
    fig, ax = _plot_psd(
        X_valid_noload, X_obs_linear_noload, X_obs_koopman_noload, t_step
    )
    fig.savefig(
        obs_nom_noload_psd_path,
        **SAVEFIG_KW,
    )

    # Nominal, load
    nom_sn = uncertainty_linear["nominal_serial_no"]
    X_valid_load = dataset.loc[
        (dataset["serial_no"] == nom_sn)
        & (dataset["episode"] == N_TRAIN)
        & (dataset["load"]),
        [
            "joint_pos",
            "joint_vel",
            "joint_trq",
            "target_joint_pos",
            "target_joint_vel",
        ],
    ].to_numpy()
    t = np.arange(X_valid_load.shape[0]) * t_step
    x0 = np.array([[0], [0], [0]])
    X_obs_linear_load = _simulate_linear(
        control.StateSpace(*observer_linear["P"]),
        control.StateSpace(*observer_linear["K"]),
        X_valid_load,
        x0=x0,
    )
    kp = models_koopman.loc[
        (models_koopman["serial_no"] == nom_sn) & (~models_koopman["load"]),
        "koopman_pipeline",
    ].item()
    X_obs_koopman_load = _simulate_koopman(
        control.StateSpace(*observer_koopman["P"]),
        control.StateSpace(*observer_koopman["K"]),
        X_valid_load,
        kp,
        x0=x0,
    )
    fig, ax = _plot_traj(t, X_valid_load, X_obs_linear_load, X_obs_koopman_load)
    fig.savefig(
        obs_nom_load_traj_path,
        **SAVEFIG_KW,
    )
    fig, ax = _plot_err(t, X_valid_load, X_obs_linear_load, X_obs_koopman_load)
    fig.savefig(
        obs_nom_load_err_path,
        **SAVEFIG_KW,
    )
    fig, ax = _plot_psd(X_valid_load, X_obs_linear_load, X_obs_koopman_load, t_step)
    fig.savefig(
        obs_nom_load_psd_path,
        **SAVEFIG_KW,
    )

    # Off-nominal, no load
    nom_sn = uncertainty_linear["nominal_serial_no"]
    offnom_sn = "011011"
    assert nom_sn != offnom_sn
    X_valid_noload = dataset.loc[
        (dataset["serial_no"] == offnom_sn)
        & (dataset["episode"] == N_TRAIN)
        & (~dataset["load"]),
        [
            "joint_pos",
            "joint_vel",
            "joint_trq",
            "target_joint_pos",
            "target_joint_vel",
        ],
    ].to_numpy()
    t = np.arange(X_valid_noload.shape[0]) * t_step
    x0 = np.array([[0], [0], [0]])
    X_obs_linear_noload = _simulate_linear(
        control.StateSpace(*observer_linear["P"]),
        control.StateSpace(*observer_linear["K"]),
        X_valid_noload,
        x0=x0,
    )
    kp = models_koopman.loc[
        (models_koopman["serial_no"] == offnom_sn) & (~models_koopman["load"]),
        "koopman_pipeline",
    ].item()
    kp.regressor_.coef_ = None  # Make super sure it's not being used
    X_obs_koopman_noload = _simulate_koopman(
        control.StateSpace(*observer_koopman["P"]),
        control.StateSpace(*observer_koopman["K"]),
        X_valid_noload,
        kp,
        x0=x0,
    )
    fig, ax = _plot_traj(t, X_valid_noload, X_obs_linear_noload, X_obs_koopman_noload)
    fig.savefig(
        obs_offnom_noload_traj_path,
        **SAVEFIG_KW,
    )
    fig, ax = _plot_err(t, X_valid_noload, X_obs_linear_noload, X_obs_koopman_noload)
    fig.savefig(
        obs_offnom_noload_err_path,
        **SAVEFIG_KW,
    )
    fig, ax = _plot_psd(
        X_valid_noload, X_obs_linear_noload, X_obs_koopman_noload, t_step
    )
    fig.savefig(
        obs_offnom_noload_psd_path,
        **SAVEFIG_KW,
    )


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


def _max_sv(ss, f, t_step):
    """Maximum singular value plot."""
    tm = np.array([_transfer_matrix(f_, ss, t_step) for f_ in f])
    mag = np.array([scipy.linalg.svdvals(tm[k, :, :])[0] for k in range(tm.shape[0])])
    return mag


def _percent_error(reference: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Calculate percent error from reference and predicted trajectories.

    Normalized using maximum amplitude of reference trajectory.

    Parameters
    ----------
    reference : np.ndarray
        Reference trajectory, witout episode feature.
    predicted : np.ndarray
        Predicted trajectory, witout episode feature.

    Returns
    -------
    np.ndarray
        Percent error.
    """
    ampl = np.max(np.abs(reference))
    percent_error = (reference - predicted) / ampl * 100
    return percent_error


def _psd_error(reference, predicted, t_step):
    """Calculate error PSD."""
    err = reference - predicted
    f, err_spec = scipy.signal.welch(
        err,
        fs=(1 / t_step),
        nperseg=512,
    )
    return f, err_spec


def _plot_traj(t, X_test, Xp_linear, Xp_koopman):
    """Plot trajectory."""
    fig, ax = plt.subplots(
        3,
        1,
        constrained_layout=True,
        figsize=(LW, LW),
        sharex=True,
    )
    ax[0].plot(t, X_test[:, 0], color=OKABE_ITO["black"], label="Measured")
    ax[1].plot(t, X_test[:, 1], color=OKABE_ITO["black"])
    ax[2].plot(t, X_test[:, 2], color=OKABE_ITO["black"])
    ax[0].plot(t, Xp_koopman[:, 0], color=OKABE_ITO["blue"], label="Koopman")
    ax[1].plot(t, Xp_koopman[:, 1], color=OKABE_ITO["blue"])
    ax[2].plot(t, Xp_koopman[:, 2], color=OKABE_ITO["blue"])
    ax[0].plot(t, Xp_linear[:, 0], color=OKABE_ITO["vermillion"], label="Linear")
    ax[1].plot(t, Xp_linear[:, 1], color=OKABE_ITO["vermillion"])
    ax[2].plot(t, Xp_linear[:, 2], color=OKABE_ITO["vermillion"])
    # Inset axis 1
    axins1 = ax[1].inset_axes(
        [0.5, 0.5, 0.48, 0.48],
        xlim=(1.5, 1.8),
        ylim=(3.05, 3.20),
    )
    axins1.plot(t, X_test[:, 1], color=OKABE_ITO["black"])
    axins1.plot(t, Xp_koopman[:, 1], color=OKABE_ITO["blue"])
    axins1.plot(t, Xp_linear[:, 1], color=OKABE_ITO["vermillion"])
    ax[1].indicate_inset_zoom(axins1, edgecolor="black")
    # Inset axis 2
    axins2 = ax[2].inset_axes(
        [0.5, 0.5, 0.48, 0.48],
        xlim=(1.5, 1.8),
        ylim=(-0.05, 0.25),
    )
    axins2.plot(t, X_test[:, 2], color=OKABE_ITO["black"])
    axins2.plot(t, Xp_koopman[:, 2], color=OKABE_ITO["blue"])
    axins2.plot(t, Xp_linear[:, 2], color=OKABE_ITO["vermillion"])
    ax[2].indicate_inset_zoom(axins2, edgecolor="black")
    # Labels
    ax[0].set_ylabel(r"$\theta(t)$ (rad)")
    ax[1].set_ylabel(r"$\dot{\theta}(t)$ (rad/s)")
    ax[2].set_ylabel(r"$i(t)$ (unitless)")
    ax[2].set_xlabel(r"$t$ (s)")
    fig.align_ylabels()
    fig.legend(
        handles=[
            ax[0].get_lines()[0],
            ax[0].get_lines()[2],
            ax[0].get_lines()[1],
        ],
        loc="upper center",
        ncol=3,
        handlelength=1,
        bbox_to_anchor=(0.5, 0.01),
    )
    return fig, ax


def _plot_err(t, X_test, Xp_linear, Xp_koopman):
    """Plot trajectory error."""
    fig, ax = plt.subplots(
        3,
        1,
        constrained_layout=True,
        figsize=(LW, LW),
        sharex=True,
    )
    ax[0].plot(
        t,
        _percent_error(X_test[:, 0], Xp_linear[:, 0]),
        color=OKABE_ITO["vermillion"],
        label="Linear",
    )
    ax[1].plot(
        t,
        _percent_error(X_test[:, 1], Xp_linear[:, 1]),
        color=OKABE_ITO["vermillion"],
    )
    ax[2].plot(
        t,
        _percent_error(X_test[:, 2], Xp_linear[:, 2]),
        color=OKABE_ITO["vermillion"],
    )
    ax[0].plot(
        t,
        _percent_error(X_test[:, 0], Xp_koopman[:, 0]),
        color=OKABE_ITO["blue"],
        label="Koopman",
    )
    ax[1].plot(
        t,
        _percent_error(X_test[:, 1], Xp_koopman[:, 1]),
        color=OKABE_ITO["blue"],
    )
    ax[2].plot(
        t,
        _percent_error(X_test[:, 2], Xp_koopman[:, 2]),
        color=OKABE_ITO["blue"],
    )
    axins2 = ax[2].inset_axes(
        [0.5, 0.5, 0.48, 0.48],
        xlim=(0, 0.5),
        ylim=(-95, 95),
    )
    axins2.plot(
        t, _percent_error(X_test[:, 2], Xp_linear[:, 2]), color=OKABE_ITO["vermillion"]
    )
    axins2.plot(
        t, _percent_error(X_test[:, 2], Xp_koopman[:, 2]), color=OKABE_ITO["blue"]
    )
    ax[2].indicate_inset_zoom(axins2, edgecolor="black")
    ax[0].set_ylabel(r"$\Delta\theta(t)$ (\%)")
    ax[1].set_ylabel(r"$\Delta\dot{\theta}(t)$ (\%)")
    ax[2].set_ylabel(r"$\Delta i(t)$ (\%)")
    ax[2].set_xlabel(r"$t$ (s)")
    ax[0].set_yticks([-0.1, -0.05, 0, 0.05, 0.1])
    ax[1].set_yticks([-20, -10, 0, 10, 20])
    ax[2].set_yticks([-100, -50, 0, 50, 100])
    ax[0].set_ylim([-0.1, 0.1])
    ax[1].set_ylim([-20, 20])
    ax[2].set_ylim([-100, 100])
    fig.align_ylabels()
    fig.legend(
        handles=[
            ax[0].get_lines()[0],
            ax[0].get_lines()[1],
        ],
        loc="upper center",
        ncol=3,
        handlelength=1,
        bbox_to_anchor=(0.5, 0.01),
    )
    return fig, ax


def _plot_psd(X_test, Xp_linear, Xp_koopman, t_step):
    """Plot trajectory error PSD."""
    fig, ax = plt.subplots(
        3,
        1,
        constrained_layout=True,
        figsize=(LW, LW),
        sharex=True,
    )
    ax[0].plot(
        *_psd_error(X_test[:, 0], Xp_linear[:, 0], t_step),
        color=OKABE_ITO["vermillion"],
        label="Linear",
    )
    ax[0].plot(
        *_psd_error(X_test[:, 0], Xp_koopman[:, 0], t_step),
        color=OKABE_ITO["blue"],
        label="Koopman",
    )
    ax[1].plot(
        *_psd_error(X_test[:, 1], Xp_linear[:, 1], t_step),
        color=OKABE_ITO["vermillion"],
        label="Linear",
    )
    ax[1].plot(
        *_psd_error(X_test[:, 1], Xp_koopman[:, 1], t_step),
        color=OKABE_ITO["blue"],
        label="Koopman",
    )
    ax[2].plot(
        *_psd_error(X_test[:, 2], Xp_linear[:, 2], t_step),
        color=OKABE_ITO["vermillion"],
        label="Linear",
    )
    ax[2].plot(
        *_psd_error(X_test[:, 2], Xp_koopman[:, 2], t_step),
        color=OKABE_ITO["blue"],
        label="Koopman",
    )
    ax[2].set_xlabel(r"$f$ (Hz)")
    ax[0].set_ylabel(
        r"$S_{\theta^\mathrm{e}\theta^\mathrm{e}}(f)$ "
        "\n"
        r"($\mathrm{rad}^2/\mathrm{Hz}$)"
    )
    ax[1].set_ylabel(
        r"$S_{\dot{\theta}^\mathrm{e}\dot{\theta}^\mathrm{e}}(f)$ "
        "\n"
        r"($\mathrm{rad}^2/\mathrm{s}^2/\mathrm{Hz}$)"
    )
    ax[2].set_ylabel(r"$S_{i^\mathrm{e}i^\mathrm{e}}(f)$" "\n" r"($1/\mathrm{Hz}$)")
    fig.align_ylabels()
    fig.legend(
        handles=[
            ax[0].get_lines()[0],
            ax[0].get_lines()[1],
        ],
        loc="upper center",
        ncol=3,
        handlelength=1,
        bbox_to_anchor=(0.5, 0.01),
    )
    return fig, ax


def _plot_weights(obs):
    """Plot weights used in synthesized observer."""
    fig, ax = plt.subplots(
        constrained_layout=True,
        figsize=(LW, LW),
    )
    ax.semilogx(
        obs["f"],
        20 * np.log10(obs["mag_p"]),
        label=r"$\boldsymbol{\mathcal{W}}_\mathrm{p}$",
        color=OKABE_ITO["sky blue"],
    )
    ax.semilogx(
        obs["f"],
        20 * np.log10(obs["mag_u"]),
        label=r"$\boldsymbol{\mathcal{W}}_\mathrm{u}$",
        color=OKABE_ITO["orange"],
        ls="--",
    )
    ax.semilogx(
        obs["f"],
        20 * np.log10(obs["mag_D"]),
        label=r"$\boldsymbol{\mathcal{W}}_{\!\Delta}$",
        color=OKABE_ITO["bluish green"],
    )
    ax.semilogx(
        obs["f"],
        20 * np.log10(obs["mag_P"]),
        label=r"$\boldsymbol{\mathcal{CG}}$",
        color=OKABE_ITO["blue"],
    )
    ax.semilogx(
        obs["f"],
        20 * np.log10(obs["mag_F"]),
        label=r"$\boldsymbol{\mathcal{K}}$",
        color=OKABE_ITO["vermillion"],
    )
    ax.set_ylabel(r"$\bar{\sigma}({\bf W}(f))$ (dB)")
    ax.set_xlabel(r"$f$ (Hz)")
    ax.set_ylim([-70, 10])
    fig.legend(
        handles=[
            ax.get_lines()[0],
            ax.get_lines()[1],
            ax.get_lines()[2],
            ax.get_lines()[3],
            ax.get_lines()[4],
        ],
        loc="upper center",
        ncol=5,
        handlelength=1,
        bbox_to_anchor=(0.5, 0.01),
    )
    return fig, ax


def _simulate_linear(P, K, X_valid, x0=None):
    if x0 is None:
        x0 = np.zeros((3, 1))
    meas = X_valid[:, :3].T
    inpt = X_valid[:, 3:5].T
    X = np.zeros((P.nstates, X_valid.shape[0]))
    X[:, [0]] = x0
    Xc = np.zeros((K.nstates, X_valid.shape[0]))
    for k in range(1, X_valid.shape[0] + 1):
        # Compute error first since 0 has no D matrix
        err = P.C @ meas[:, k - 1] - P.C @ X[:, k - 1]
        # Compute control output
        u = K.C @ Xc[:, k - 1] + K.D @ err
        if k < X.shape[1]:
            # Update plant with control input
            X[:, k] = P.A @ X[:, k - 1] + P.B @ (inpt[:, k - 1] + u)
            # Update controller
            Xc[:, k] = K.A @ Xc[:, k - 1] + K.B @ err
    return X.T


def _simulate_koopman(P, K, X_valid, kp, x0=None, linear_prediction=False):
    if x0 is None:
        x0 = np.zeros((3, 1))
    meas_ = X_valid[:, :3].T
    meas = kp.lift_state(meas_.T, episode_feature=False).T
    inpt = X_valid[:, 3:5].T
    X = np.zeros((P.nstates, X_valid.shape[0]))
    X[:, [0]] = kp.lift_state(x0.T, episode_feature=False).T
    Xc = np.zeros((K.nstates, X_valid.shape[0]))
    for k in range(1, X_valid.shape[0] + 1):
        # Compute error first since 0 has no D matrix
        err = P.C @ meas[:, k - 1] - P.C @ X[:, k - 1]
        # Compute control output
        u = K.C @ Xc[:, k - 1] + K.D @ err
        if k < X.shape[1]:
            # Update plant with control input
            if linear_prediction:
                X[:, k] = P.A @ X[:, k - 1] + P.B @ (inpt[:, k - 1] + u)
            else:
                # TODO Need to actually plot X_rl instead of X I think
                Xt_ret = kp.retract_state(X[:, [k - 1]].T, episode_feature=False)
                X_rl = kp.lift_state(Xt_ret, episode_feature=False).T.ravel()
                X[:, k] = P.A @ X_rl + P.B @ (inpt[:, k - 1] + u)
            # Update controller
            Xc[:, k] = K.A @ Xc[:, k - 1] + K.B @ err
    return X.T
