import sys

sys.path.append("..")
import postmerger as pm
import numpy as np
import os

repo = os.environ["RINGREPO"]  # set to use on different machines
# sys.path.append(repo + "/code")
sys.path.append(repo)
import load_data_v2 as ld
import GPR_regressor_class as gprcl
import time
import joblib
import argparse
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":

    subsample = 0
    notes = "LogWeights"

    ### ----- Parse command line arguments --------
    parser = argparse.ArgumentParser(description="Train or cross-validate GPR")

    parser.add_argument("-f", "--features", type=str, required=True, action="store")
    parser.add_argument("-y", "--y_fit", type=str, required=True, action="store")
    # parser.add_argument("-nt", "--notes", type=str, required=False, action="store")

    args = vars(parser.parse_args())

    features = args["features"]
    # notes = args["notes"]
    y_fit = args["y_fit"]

    save_filepath = "/home/fra/ringdown/postmerger/postmerger/data/trained_models/"
    if features == "X_7d_ISCO":
        save_file = save_filepath + "Prec7dq10.pkl"
    elif features == "X_6d_theta":
        save_file = save_filepath + "Prec6dq10.pkl"
    else:
        raise ValueError("Invalid 'features' argument. Choose one of the available")

    if Path(save_file).exists():
        print(f"File {save_file} already exists. Changes will be overwritten.")
        # print(
        #     f"File {save_file} already exists. Changes will be overwritten. Continue? (y/n)"
        # )
        # cont = input()
        # if cont != "y":
        #     print("Exiting...")
        #     sys.exit(0)

        with open(save_file, "rb") as f:
            dict_to_save = joblib.load(f)

    else:
        print(f"File {save_file} does not exist. Creating new file.")

        amp_models_dict = {}
        abs_err_models_dict = {}

        dict_to_save = {
            "time_from_temop": 20,
            "amps": amp_models_dict,
            "abs_err": abs_err_models_dict,
            "t_emop": None,
        }

    if y_fit == "t_emop":

        mode = (2, 2)

        sub = ld.SourcesSubset(
            cut_condition="bad_fit",
            print_info=False,
            data_for_goodness_of_fit=[20, "EMOP", "out_ROT2"],
            modes_for_goodness_of_fit=[mode],
            status_subsample=subsample,
        )

        names_array = sub.names_array

        [
            mode_list,
            new_mode_list,
            status_array,
            angle_array,  # between initial L (computed as J - S1 - S2) and rem spin vec
            omega_angle_array,  # between initial L (computed) and J (initial_ADM_angular_momentum)
            j_rem_angle,  # between J (initial_ADM_angular_momentum) and rem spin vec
            beta_angle,  # between rem_spin_vector and z-component of rem_spin
            theta2_isco_pro,  # between L (computed as J - S1 - S2) at isco_prograde and rem spin vec
            theta2_peak22,  # between L (computed as J - S1 - S2) at peak22 and rem spin vec
            remnant_spin_x,
            remnant_spin_y,
            remnant_spin_z,
            remnant_spin_vector,
            remnant_spin_array,  # Norm of the vector
            remnant_mass_array,
            chip_ISCO_array,
            chip_array,
            q_array,
            eta_array,
            delta_array,
            mass1_array,
            mass2_array,
            Lx_array,
            Ly_array,
            Lz_array,
            L_ISCO_vector,
            J_adm_ini,
            J_hor_ini,
            J_ISCO_vector,
            ecc_array,
            separation_array,
            chi1x_ISCO,
            chi1y_ISCO,
            chi1z_ISCO,
            chi1_ISCO_mag,
            chi2x_ISCO,
            chi2y_ISCO,
            chi2z_ISCO,
            chi2_ISCO_mag,
            chi_plus_ISCO,
            chi_odd_ISCO,
            chi1x_array,
            chi1y_array,
            chi1_array,  # z_comp
            chi2x_array,
            chi2y_array,
            chi2_array,  # z_comp
            chi_plus_array,
            chi_odd_array,
            chi1x_rot,
            chi1y_rot,
            chi1z_rot,
            chi2x_rot,
            chi2y_rot,
            chi2z_rot,
            chi_plus_rot,
            chi_minus_rot,
            chi_odd_rot,
            chi1x_rot_ISCO,
            chi1y_rot_ISCO,
            chi1z_rot_ISCO,
            chi2x_rot_ISCO,
            chi2y_rot_ISCO,
            chi2z_rot_ISCO,
            chi_plus_rot_ISCO,
            chi_minus_rot_ISCO,
            chi_odd_rot_ISCO,
            fixed_freq_array,
            fixed_tau_array,
            fixed_omegaRE_array,
            fixed_omegaIM_array,
            kick_vel_vector,
            kick_angle,
            chi1_r_rot_ISCO,
            chi1_lat_rot_ISCO,
            chi1_lon_rot_ISCO,
            chi2_r_rot_ISCO,
            chi2_lat_rot_ISCO,
            chi2_lon_rot_ISCO,
            chi1x_rot_L,
            chi1y_rot_L,
            chi1z_rot_L,
            chi2x_rot_L,
            chi2y_rot_L,
            chi2z_rot_L,
            chi_plus_rot_L,
            chi_minus_rot_L,
            chi_odd_rot_L,
            chi1x_rot_ISCO_L,
            chi1y_rot_ISCO_L,
            chi1z_rot_ISCO_L,
            chi2x_rot_ISCO_L,
            chi2y_rot_ISCO_L,
            chi2z_rot_ISCO_L,
            chi_plus_rot_ISCO_L,
            chi_minus_rot_ISCO_L,
            chi_odd_rot_ISCO_L,
            kick_vel,
            fixed_freq_array_ret,
            fixed_tau_array_ret,
            fixed_omegaRE_array_ret,
            fixed_omegaIM_array_ret,
        ] = ld.load_binary_params(names_array)

        X_7d_ISCO = np.vstack(
            (
                delta_array,
                chi1x_rot_ISCO,
                chi1y_rot_ISCO,
                chi1z_rot_ISCO,
                chi2x_rot_ISCO,
                chi2y_rot_ISCO,
                chi2z_rot_ISCO,
            )
        ).T

        X_6d_theta = np.vstack(
            (
                delta_array,
                chi_plus_rot_ISCO,
                chi_minus_rot_ISCO,
                theta2_isco_pro,
                kick_angle,
                kick_vel,
            )
        ).T

        if features == "X_7d_ISCO":
            X = X_7d_ISCO
        elif features == "X_6d_theta":
            X = X_6d_theta
        else:
            raise ValueError("Invalid 'features' argument. Choose one of the available")

        mode_list = sub.new_mode_list
        mode_n = mode_list.index(mode)

        fit_data_d = ld.load_fit_data(names_array, 20, "EMOP", "out_ROT2")
        new_amp_array = fit_data_d["new_amp_array"]
        new_error_zhu = fit_data_d["new_error_zhu"]

        emop_d = ld.load_emop_data(names_array, "out_ROT2")
        emop_abs_t_array = emop_d["emop_abs_t_array"]
        emop_array = emop_d["emop_array"]
        peak_d = ld.load_peak_data(names_array, "out_ROT2")
        peak_tot_array = peak_d["peak_tot_array"]
        delta_emop_t = emop_abs_t_array - peak_tot_array

        y = delta_emop_t

        err = new_error_zhu[:, mode_n]
        usable_err_max = np.max(err[err < 1.0])
        clipped_err = np.clip(
            err, a_min=None, a_max=usable_err_max
        )  # Clip because sometimes error > 1 when offset is present in waveform.
        dy = -1 / np.log10(clipped_err) + 1e-10  # const is default sklearn value

        if y_fit == "t_emop":
            dy = np.full_like(y, 1e-5)

        nparams = X.shape[1]
        npoints = X.shape[0]

        model_name = gprcl.model_name_fun(
            mode=mode,
            y_fit=y_fit,
            features=features,
            linear_fit=False,
            subsample=subsample,
            use_scaler=True,
            notes=notes,
        )

        print(f"Fitting model: {model_name}")

        gpr_pipe = pm.CustomGPR()

        gpr_pipe.fit(X=X, y=y, sample_weight=1 / dy, linear_fit=False, precessing=True)

        dict_to_save["t_emop"] = gpr_pipe
    else:

        for mode in tqdm(ld.mismatch_mode_list210):

            sub = ld.SourcesSubset(
                cut_condition="bad_fit",
                print_info=False,
                data_for_goodness_of_fit=[20, "EMOP", "out_ROT2"],
                modes_for_goodness_of_fit=[mode],
                status_subsample=subsample,
            )

            names_array = sub.names_array

            [
                mode_list,
                new_mode_list,
                status_array,
                angle_array,  # between initial L (computed as J - S1 - S2) and rem spin vec
                omega_angle_array,  # between initial L (computed) and J (initial_ADM_angular_momentum)
                j_rem_angle,  # between J (initial_ADM_angular_momentum) and rem spin vec
                beta_angle,  # between rem_spin_vector and z-component of rem_spin
                theta2_isco_pro,  # between L (computed as J - S1 - S2) at isco_prograde and rem spin vec
                theta2_peak22,  # between L (computed as J - S1 - S2) at peak22 and rem spin vec
                remnant_spin_x,
                remnant_spin_y,
                remnant_spin_z,
                remnant_spin_vector,
                remnant_spin_array,  # Norm of the vector
                remnant_mass_array,
                chip_ISCO_array,
                chip_array,
                q_array,
                eta_array,
                delta_array,
                mass1_array,
                mass2_array,
                Lx_array,
                Ly_array,
                Lz_array,
                L_ISCO_vector,
                J_adm_ini,
                J_hor_ini,
                J_ISCO_vector,
                ecc_array,
                separation_array,
                chi1x_ISCO,
                chi1y_ISCO,
                chi1z_ISCO,
                chi1_ISCO_mag,
                chi2x_ISCO,
                chi2y_ISCO,
                chi2z_ISCO,
                chi2_ISCO_mag,
                chi_plus_ISCO,
                chi_odd_ISCO,
                chi1x_array,
                chi1y_array,
                chi1_array,  # z_comp
                chi2x_array,
                chi2y_array,
                chi2_array,  # z_comp
                chi_plus_array,
                chi_odd_array,
                chi1x_rot,
                chi1y_rot,
                chi1z_rot,
                chi2x_rot,
                chi2y_rot,
                chi2z_rot,
                chi_plus_rot,
                chi_minus_rot,
                chi_odd_rot,
                chi1x_rot_ISCO,
                chi1y_rot_ISCO,
                chi1z_rot_ISCO,
                chi2x_rot_ISCO,
                chi2y_rot_ISCO,
                chi2z_rot_ISCO,
                chi_plus_rot_ISCO,
                chi_minus_rot_ISCO,
                chi_odd_rot_ISCO,
                fixed_freq_array,
                fixed_tau_array,
                fixed_omegaRE_array,
                fixed_omegaIM_array,
                kick_vel_vector,
                kick_angle,
                chi1_r_rot_ISCO,
                chi1_lat_rot_ISCO,
                chi1_lon_rot_ISCO,
                chi2_r_rot_ISCO,
                chi2_lat_rot_ISCO,
                chi2_lon_rot_ISCO,
                chi1x_rot_L,
                chi1y_rot_L,
                chi1z_rot_L,
                chi2x_rot_L,
                chi2y_rot_L,
                chi2z_rot_L,
                chi_plus_rot_L,
                chi_minus_rot_L,
                chi_odd_rot_L,
                chi1x_rot_ISCO_L,
                chi1y_rot_ISCO_L,
                chi1z_rot_ISCO_L,
                chi2x_rot_ISCO_L,
                chi2y_rot_ISCO_L,
                chi2z_rot_ISCO_L,
                chi_plus_rot_ISCO_L,
                chi_minus_rot_ISCO_L,
                chi_odd_rot_ISCO_L,
                kick_vel,
                fixed_freq_array_ret,
                fixed_tau_array_ret,
                fixed_omegaRE_array_ret,
                fixed_omegaIM_array_ret,
            ] = ld.load_binary_params(names_array)

            X_7d_ISCO = np.vstack(
                (
                    delta_array,
                    chi1x_rot_ISCO,
                    chi1y_rot_ISCO,
                    chi1z_rot_ISCO,
                    chi2x_rot_ISCO,
                    chi2y_rot_ISCO,
                    chi2z_rot_ISCO,
                )
            ).T

            X_6d_theta = np.vstack(
                (
                    delta_array,
                    chi_plus_rot_ISCO,
                    chi_minus_rot_ISCO,
                    theta2_isco_pro,
                    kick_angle,
                    kick_vel,
                )
            ).T

            if features == "X_7d_ISCO":
                X = X_7d_ISCO
            elif features == "X_6d_theta":
                X = X_6d_theta
            else:
                raise ValueError(
                    "Invalid 'features' argument. Choose one of the available"
                )

            mode_list = sub.new_mode_list
            mode_n = mode_list.index(mode)

            fit_data_d = ld.load_fit_data(names_array, 20, "EMOP", "out_ROT2")
            new_amp_array = fit_data_d["new_amp_array"]
            new_error_zhu = fit_data_d["new_error_zhu"]

            emop_d = ld.load_emop_data(names_array, "out_ROT2")
            emop_abs_t_array = emop_d["emop_abs_t_array"]
            emop_array = emop_d["emop_array"]
            peak_d = ld.load_peak_data(names_array, "out_ROT2")
            peak_tot_array = peak_d["peak_tot_array"]
            delta_emop_t = emop_abs_t_array - peak_tot_array

            if y_fit == "amps":
                y = new_amp_array[:, mode_n]
            elif y_fit == "t_emop":
                y = delta_emop_t
            elif y_fit == "abs_err":
                cvmod = gprcl.CrossValModel(
                    n_to_leave_out=1,
                    mode=mode,
                    features=features,
                    linear_fit=False,
                    subsample=0,
                    use_scaler=True,
                    y_fit="amp",
                    notes="LogWeights",
                )
                cvmod.load_cv_data()
                err_subsample_list = [
                    "all",
                    "non_spinning",
                    "spinning_aligned",
                    "aligned",
                    "precessing",
                    "original",
                ]
                err_subsample = err_subsample_list[subsample]
                cvmod.compute_cv_stats(
                    err_subsample=err_subsample, rel_err_rebinning=False
                )
                abs_err = np.abs(cvmod.abs_err)
                y = abs_err

            err = new_error_zhu[:, mode_n]
            usable_err_max = np.max(err[err < 1.0])
            clipped_err = np.clip(
                err, a_min=None, a_max=usable_err_max
            )  # Clip because sometimes error > 1 when offset is present in waveform.
            dy = -1 / np.log10(clipped_err) + 1e-10  # const is default sklearn value

            if y_fit == "t_emop":
                dy = np.full_like(y, 1e-5)

            nparams = X.shape[1]
            npoints = X.shape[0]

            model_name = gprcl.model_name_fun(
                mode=mode,
                y_fit=y_fit,
                features=features,
                linear_fit=False,
                subsample=subsample,
                use_scaler=True,
                notes=notes,
            )

            print(f"Fitting model: {model_name}")

            gpr_pipe = pm.CustomGPR()

            gpr_pipe.fit(
                X=X, y=y, sample_weight=1 / dy, linear_fit=False, precessing=True
            )

            dict_to_save[y_fit][mode] = gpr_pipe

        dict_to_save2 = dict_to_save.copy()
        for key, obj in dict_to_save2[y_fit].items():
            dict_to_save[y_fit][key] = {(key[0], key[1], 0): obj}

    with open(save_file, "wb") as f:
        joblib.dump(dict_to_save, f)
