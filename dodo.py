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

from onesine import OneSineLiftingFn

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
    models_koopman = WD.joinpath("build", "models_koopman.pickle")
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
    models_koopman = WD.joinpath("build", "models_koopman.pickle")
    residuals_linear = WD.joinpath("build", "residuals_linear.pickle")
    residuals_koopman = WD.joinpath("build", "residuals_koopman.pickle")
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
    method: str,
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
        if method == "koopman":
            # Get optimal phase shift
            phase = joblib.load(phase_path)
            phi_all = phase.loc[(phase["serial_no"] == i[0]) & (phase["load"] == i[1])]
            optimal_phi = _circular_mean(phi_all["optimal_phase"].to_numpy())
            # Set lifting functions
            lf = [
                (
                    "sin",
                    OneSineLiftingFn(
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
