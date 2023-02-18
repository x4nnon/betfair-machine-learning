import os

import pandas as pd
from onedrive import Onedrive
from flumine import clients
from flumine_simulator import piped_run
from matplotlib import pyplot as plt
from utils.config import app_principal, SITE_URL
import yaml

from utils.utils import plot_simulation_results


plt.rcParams["figure.figsize"] = (20, 3)

if __name__ == "__main__":
    test_folder_path = "horses_jul_wins"
    bsps_path = "july_22_bsps"
    strategy = "Strategy1"

    tracker = piped_run(
        strategy=strategy,
        onedrive=Onedrive(
            client_id=app_principal["client_id"],
            client_secret=app_principal["client_secret"],
            site_url=SITE_URL,
        ),
        client=clients.SimulatedClient(),
        test_folder_path=test_folder_path,
        bsps_path=bsps_path,
        model_name="BayesianRidge",
        test_run=True,
    )
    with open(f"{strategy}_results.yaml", "w") as f:
        yaml.dump(tracker, f)
    # tracker = dict()
    # with open("dummy_data/tracker.yaml", "r") as f:
    #     tracker = yaml.safe_load(f)

    # fig = plot_simulation_results(tracker, strategy)
    # plt.show()
