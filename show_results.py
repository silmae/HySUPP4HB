import os
import logging
import subprocess
import datetime
import sys
import yaml
import json
import pandas as pd
import seaborn as sns

from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np

from src.model.blind import MiSiCNet

accepted_supervised_models = ["FCLS", "UnDIP"]
accepted_blind_models = ["MiSiCNet", "MSNet"]
accepted_extractors = ["SiVM", "SISAL", "VCA"]


def _init_logging():

    # path_dir_logs = PH.directory_log()
    # if not os.path.exists(path_dir_logs):
    #     os.makedirs(path_dir_logs)

    log_identifier = str(datetime.datetime.now())
    log_identifier = log_identifier.replace(" ", "_")
    log_identifier = log_identifier.replace(":", "")
    log_identifier = log_identifier.replace(".", "")

    # log_file_name = f"{log_identifier}.log"
    # log_path = PH.join(path_dir_logs, log_file_name)
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s: %(message)s",
        force=True,
        handlers=[
            # logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("Logging initialized")


def hysupp_runner(data_name: str, model_name: str, extractor_name: str=None, dry_run = True):
    """Run a single HySUPP experiment."""

    if model_name not in accepted_supervised_models and model_name not in accepted_blind_models:
        raise ValueError(f"Unsupported model {model_name}")

    if model_name == "FCLS" or model_name == "UnDIP":
        mode = "supervised"
    elif model_name == "MSNet" or model_name == "MiSiCNet":
        mode = "blind"
    else:
        raise ValueError(f"Unsupported model {model_name}. Check the list of accepted models.")

    if extractor_name is not None:

        if extractor_name not in accepted_extractors:
            raise ValueError(f"Unsupported extractor {extractor_name}")

        combination_name = f"{extractor_name}_{model_name}"
    else:
        combination_name = f"{model_name}"

    # path_result_dir_top = os.path.abspath(os.path.join(f"./Experiments/{mode}/{combination_name}/"))
    path_result_dir_top = os.path.abspath(os.path.join(f"./Experiments/{mode}/"))

    logging.error(f"The top level result directory is {path_result_dir_top}")


    folder_name = 'MyModel_hals2_ssim_v2'

    with open('./config/mlxp.yaml', 'r') as file:
        config_yaml = yaml.safe_load(file)

    # Modify a value
    config_yaml['logger']['parent_log_dir'] = f"{path_result_dir_top}/{combination_name}_{data_name}/"
    # config_yaml['logger']['parent_log_dir'] = f"{path_result_dir_top}"
    config_yaml['logger']['forced_log_id'] = -1

    # Save back to YAML file
    with open('./config/mlxp.yaml', 'w') as file:
        yaml.safe_dump(config_yaml, file)




    if extractor_name is not None:
        call_string = f"python unmixing.py mode={mode} data={data_name} model={model_name} extractor={extractor_name}"
    else:
        call_string = f"python unmixing.py mode={mode} data={data_name} model={model_name}"

    logging.info(f"running HySUPP with following argument list:\n\t'{call_string}'")

    if not dry_run:
        with open(os.devnull, "wb") as stream:

            exit_code = subprocess.run(call_string)

            if exit_code.returncode != 0:
                logging.fatal(
                    f"HySUPP run failed with exit code {exit_code.returncode}."
                )
                exit(exit_code.returncode)
    else:
        print("This was a dry run. Nothing is really executed.")


def experiment_looper(themes, resolutions, model_name: str, extractor_name=None):

    for resolution in resolutions:
        for theme in themes:
            data_name = f"{theme}_{resolution}"
            mat_name = f"{data_name}.mat"
            mat_path = os.path.abspath(os.path.join("./data/", mat_name))

            if os.path.exists(mat_path):
                print(f"Path to mat ok at '{mat_path}'")
                try:
                    hysupp_runner(data_name=data_name, model_name=model_name, extractor_name=extractor_name, dry_run=False)
                except ValueError as e:
                    logging.error(f"Failed to run HySUPP for {data_name}. Continuing to the next experiment.\n "
                                  f"Runner exception: {e}")
            else:
                print(f"Path not found '{mat_path}'")


def model_looper(themes, resolutions, models, extractors=None):
    for model in models:
        if model in accepted_supervised_models:
            if extractors is not None:
                for extractor in extractors:
                    experiment_looper(
                        themes=themes, resolutions=resolutions, extractor_name=extractor,
                        model_name=model
                        )
            else:
                raise ValueError(f"Running of supervised model '{model}' requested but no extractors were provided.")
        else:
            experiment_looper(themes=themes, resolutions=resolutions, model_name=model)



def by_key_val(dicts, keys: list, values: list):

    if len(keys) != len(values):
        raise ValueError(f"Provide as many keys {len(keys)} as values {len(values)}")

    # print(f"####### Results by {keys} == {values}  ################################")

    dict_list = []

    for json_dict in dicts:
        matches = True
        for i, key in enumerate(keys):
            if json_dict[key] != values[i]:
                matches = False

        if matches:
            dict_list.append(json_dict)
            # print(json_dict)

    return dict_list


def show_results(metric: str):
    """

    :param metric:
        One of: "SAD", "SRE", "aRMSE", "eRMSE".
    """

    path_dir_experiments = os.path.abspath(f"{os.getcwd()}/experiments/")
    path_dir_supervised_results = os.path.join(path_dir_experiments, "supervised/")
    path_dir_blind_results = os.path.join(path_dir_experiments, "blind/")

    path_loops = [path_dir_supervised_results, path_dir_blind_results]

    dicts = []

    key_extractor_name = "Extractor"
    key_unmixer_name = "Unmixer"
    key_ext_plus_umx = "ext_plus_umx"
    key_scene_name = "Scene name"
    key_scene_name_bare = "Scene"
    key_resolution = "Resolution"
    key_run_name = "Run"
    key_soil_type = "Soil type"
    key_overall = metric

    for path_loop in path_loops:
        for dir_name in os.listdir(path_loop):
            path_dir_result_top = os.path.abspath(os.path.join(path_loop, dir_name, "1/"))
            split_name = dir_name.split("_")
            if len(split_name) == 4:
                extractor_name = split_name[-4]
            else:
                extractor_name = "VCA*"
            unmixer_name = split_name[-3]
            scene_name = split_name[-2]
            resolution = split_name[-1]
            # print(f"Scene {scene_name}, resolution {resolution}, path {path_dir_result_top}")
            path_dir_metrics = os.path.join(path_dir_result_top, "metrics/")
            path_file_sad = os.path.join(path_dir_metrics, f"{metric}.json")

            if os.path.exists(path_file_sad):
                with open(path_file_sad) as file:
                    json_dict = json.load(file)
                    json_dict[key_extractor_name] = extractor_name
                    json_dict[key_unmixer_name] = unmixer_name
                    json_dict[key_ext_plus_umx] = extractor_name + "_" + unmixer_name
                    json_dict[key_scene_name] = scene_name
                    json_dict[key_scene_name_bare] = scene_name.replace("WP", "").replace("DS", "")
                    json_dict[key_resolution] = int(resolution)
                    json_dict[key_run_name] = dir_name
                    if "DS" in scene_name:
                        json_dict[key_soil_type] = "dry sand"
                    elif "WP" in scene_name:
                        json_dict[key_soil_type] = "wet peat"
                    else:
                        json_dict[key_soil_type] = None

                    # print(json_dict)
                    dicts.append(json_dict)

    pd_data = pd.DataFrame(dicts)
    pd_data = pd_data.rename(columns={"Overall": metric,})
    pd_data = pd_data[[key_run_name, key_resolution, key_overall, key_soil_type, key_extractor_name, key_unmixer_name, key_scene_name, key_ext_plus_umx, key_scene_name_bare,]]
    sns.catplot(data=pd_data, kind="bar", x=key_scene_name_bare, y=key_overall, hue=key_ext_plus_umx, row=key_soil_type, col=key_resolution)
    plt.show()

    return

    # runlist = [22]
    runlist = range(50,77)

    for run in runlist:
        # print(run)
        res = loadmat(f'{os.getcwd()}/logs/{run}/artifacts/Estimate/estimates.mat')
        num_of_ems = res['E'].shape[1]
        A = res['A']

        # Min and max values for scaling the ab maps
        vmin, vmax = 0, 1

        # Setting the subplots size
        cols = 3
        if num_of_ems < cols:
            rows = 1
        else:
            rows = int(np.ceil(num_of_ems / cols))
        # print(num_of_ems, cols, rows)

        fig, axes = plt.subplots(rows, cols, )
        axes = axes.flatten()

        i = 0
        while i < num_of_ems:
            # log_num = run.split('\\')[-1]

            axes[i].set_title('Endmember: ' + str(i + 1))
            im = axes[i].imshow(A[i, :, :], vmin=vmin, vmax=vmax)
            # axes[i].axis('off')

            i += 1

        # Turn off axis of all the plots
        for ax in axes:
            ax.axis('off')

        fig.tight_layout()
        cbar = fig.colorbar(im, ax=axes, orientation='horizontal')  # , fraction=0.03, pad=0.05)
        cbar.set_label("Abundance")
        plt.show()

        ####

        plt.close("all")

        for i, run in enumerate(runlist):
            try:
                E = res['E']
                print(f"Endmember max: {np.max(E)}")
                plt.plot(E)
                # plt.set_title(f'Endmembers: {E.shape[-1]}')
            except:
                print(f'This run ({run}) probably did not finish...')

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":

    """
    TODO: Gathering results (metrics, endmembers, abundances) How do we measure goodness?
          Log results in a meaningful way. Like:
                - model_XXX_extractor_YYY
                    - FDS1_4
                        - artifacts
                        - ...
                    - FDS1_16
                        - ...
                    - Overview
                        - run parameters
                        - running time
                        - what all this should contain?
    """

    _init_logging()
    data_name = "FDS1_16"
    model_name = "UnDIP"
    extractor_name = "SISAL"
    extractors = ["SISAL", "SiVM", "VCA"]
    supervised_models = ["FCLS", "UnDIP"]
    blind_models = ["MSNet"]
    # hysupp_runner(data_name=data_name, model_name=model_name, extractor_name=extractor_name, dry_run=False)
    themes = ["FWP1", "FDS1", "FWP2", "FDS2", "IWP1", "IDS1", "OWP1", "ODS1", "OWP2", "ODS2"]
    resolutions = [4, 16, 64, 256, 1024]
    # experiment_looper(themes=themes, resolutions=resolutions, extractor_name=extractor_name, model_name=model_name)
    # model_looper(themes=themes, resolutions=resolutions, extractors=["VCA"], models=["UnDIP"])
    # model_looper(themes=themes, resolutions=resolutions, models=accepted_blind_models)

    show_results(metric="aRMSE")



