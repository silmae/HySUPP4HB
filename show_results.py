import os
import logging
import subprocess
import datetime
import sys
import yaml
import json

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

    print(f"####### Results by {keys} == {values}  ################################")

    for json_dict in dicts:
        matches = True
        for i, key in enumerate(keys):
            if json_dict[key] != values[i]:
                matches = False

        if matches:
            print(json_dict)


def show_results():

    path_dir_experiments = os.path.abspath(f"{os.getcwd()}/experiments/")
    path_dir_supervised_results = os.path.join(path_dir_experiments, "supervised/")
    path_dir_blind_results = os.path.join(path_dir_experiments, "blind/")

    path_loop = path_dir_supervised_results
    # path_loop = path_dir_blind_results

    dicts = []

    for dir_name in os.listdir(path_loop):
        path_dir_result_top = os.path.abspath(os.path.join(path_loop, dir_name, "1/"))
        split_name = dir_name.split("_")
        extractor_name = split_name[-4]
        unmixer_name = split_name[-3]
        scene_name = split_name[-2]
        resolution = split_name[-1]
        # print(f"Scene {scene_name}, resolution {resolution}, path {path_dir_result_top}")
        path_dir_metrics = os.path.join(path_dir_result_top, "metrics/")
        path_file_sad = os.path.join(path_dir_metrics, "SAD.json")

        if os.path.exists(path_file_sad):
            with open(path_file_sad) as file:
                json_dict = json.load(file)
                json_dict["extractor_name"] = extractor_name
                json_dict["unmixer_name"] = unmixer_name
                json_dict["scene_name"] = scene_name
                json_dict["resolution"] = resolution
                json_dict["run_name"] = dir_name

                # print(json_dict)
                dicts.append(json_dict)

    # by_key_val(dicts, keys=["extractor_name"], values=["SISAL"])
    # by_key_val(dicts, keys=["extractor_name"], values=["SiVM"])
    # by_key_val(dicts, keys=["unmixer_name"], values=["UnDIP"])
    # by_key_val(dicts, keys=["unmixer_name"], values=["FCLS"])
    reso = "64"
    # by_key_val(dicts, keys=["scene_name", "resolution"], values=["FWP1", reso])
    # by_key_val(dicts, keys=["scene_name", "resolution"], values=["FDS1", reso])
    # by_key_val(dicts, keys=["scene_name", "resolution"], values=["FWP2", reso])
    # by_key_val(dicts, keys=["scene_name", "resolution"], values=["FDS2", reso])
    # by_key_val(dicts, keys=["scene_name", "resolution"], values=["IWP1", reso])
    # by_key_val(dicts, keys=["scene_name", "resolution"], values=["IDS1", reso])
    # by_key_val(dicts, keys=["scene_name", "resolution"], values=["OWP1", reso])
    # by_key_val(dicts, keys=["scene_name", "resolution"], values=["ODS1", reso])
    # by_key_val(dicts, keys=["scene_name", "resolution"], values=["OWP2", reso])
    # by_key_val(dicts, keys=["scene_name", "resolution"], values=["ODS2", reso])

    extract = "SISAL"
    # by_key_val(dicts, keys=["scene_name", "extractor_name"], values=["ODS2", extract])

    by_key_val(dicts, keys=["scene_name", "extractor_name"], values=["FWP1", extract])
    by_key_val(dicts, keys=["scene_name", "extractor_name"], values=["FDS1", extract])
    by_key_val(dicts, keys=["scene_name", "extractor_name"], values=["FWP2", extract])
    by_key_val(dicts, keys=["scene_name", "extractor_name"], values=["FDS2", extract])
    by_key_val(dicts, keys=["scene_name", "extractor_name"], values=["IWP1", extract])
    by_key_val(dicts, keys=["scene_name", "extractor_name"], values=["IDS1", extract])
    by_key_val(dicts, keys=["scene_name", "extractor_name"], values=["OWP1", extract])
    by_key_val(dicts, keys=["scene_name", "extractor_name"], values=["ODS1", extract])
    by_key_val(dicts, keys=["scene_name", "extractor_name"], values=["OWP2", extract])
    by_key_val(dicts, keys=["scene_name", "extractor_name"], values=["ODS2", extract])



    exit(0)

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
    # model_looper(themes=themes, resolutions=resolutions, extractors=accepted_extractors, models=accepted_supervised_models)
    # model_looper(themes=themes, resolutions=resolutions, models=accepted_blind_models)

    show_results()



