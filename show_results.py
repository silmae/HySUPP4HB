import os

from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np


if __name__ == "__main__":
    # res_dir = f"D:\Koodi\Python\HySUPP_INRIA\HySUPP\logs\1\artifacts\Estimate\estimates.mat"
    # print(os.getcwd())
    # res_dir = f"{os.getcwd()}/logs/1/artifacts/Estimate\estimates.mat"
    # res_dir = os.path.abspath(res_dir)
    #
    # print(f"Absolute res path: '{res_dir}'.")
    #
    # res = loadmat(res_dir)
    # em = res['E']
    # A = res['A']
    #
    # print("boi")

    # Plotting all the abundance maps.

    runlist = [5]

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

        fig, axes = plt.subplots(rows, cols,)
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
        cbar = fig.colorbar(im, ax=axes, orientation='horizontal')#, fraction=0.03, pad=0.05)
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
