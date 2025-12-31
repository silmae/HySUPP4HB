import os
import logging
import subprocess
import datetime
import sys

from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np

accepted_models = ["FCLS", "UnDIP", "MiSiCNet", "MSNet"]
accepted_extractors = ["SiVM", "SISAL", "VCA"]

def show_results():

    runlist = [22]

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


def hysupp_runner(data_name: str, model_name: str, extractor_name: str, dry_run = True):
    """Run a single HySUPP experiment."""

    if extractor_name not in accepted_extractors:
        raise ValueError(f"Unsupported extractor {extractor_name}")
    if model_name not in accepted_models:
        raise ValueError(f"Unsupported model {model_name}")

    if model_name == "FCLS" or model_name == "UnDIP":
        mode = "supervised"
    elif model_name == "MSNet" or model_name == "MiSiCNet":
        mode = "blind"
    else:
        raise ValueError(f"Unsupported model {model_name}. Check the list of accepted models.")

    call_string = f"python unmixing.py mode={mode} data={data_name} model={model_name} extractor={extractor_name}"

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


def experiment_looper(themes, resolutions, extractor_name: str, model_name: str):

    for resolution in resolutions:
        for theme in themes:
            data_name = f"{theme}_{resolution}"
            mat_name = f"{data_name}.mat"
            mat_path = os.path.abspath(os.path.join("./data/", mat_name))

            if os.path.exists(mat_path):
                print(f"Path to mat ok at '{mat_path}'")
                try:
                    hysupp_runner(data_name=data_name, model_name=model_name, extractor_name=extractor_name, dry_run=True)
                except ValueError as e:
                    logging.error(f"Failed to run HySUPP for {data_name}. Continuing to the next experiment.\n "
                                  f"Runner exception: {e}")
            else:
                print(f"Path not found '{mat_path}'")


def model_looper(themes, resolutions, extractors, models):
    for model in models:
        for extractor in extractors:
            experiment_looper(
                themes=themes, resolutions=resolutions, extractor_name=extractor,
                model_name=model
                )


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
    extractors = ["SISAL", "SiVM", "FAKEextractor"]
    models = ["FCLS", "UnDIP", "FakeMODEL"]
    # hysupp_runner(data_name=data_name, model_name=model_name, extractor_name=extractor_name)
    themes = ["FWP1", "FDS1", "IWP1", "XYY6"]
    resolutions = [4, 16, 64, 256]
    # experiment_looper(themes=themes, resolutions=resolutions, extractor_name=extractor_name, model_name=model_name)
    model_looper(themes=themes, resolutions=resolutions, extractors=extractors, models=models)



