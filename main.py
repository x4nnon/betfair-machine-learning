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
        strategy="Strategy1",
        onedrive=Onedrive(
            client_id=app_principal["client_id"],
            client_secret=app_principal["client_secret"],
            site_url=SITE_URL,
        ),
        client=clients.SimulatedClient(),
        test_folder_path=test_folder_path,
        bsps_path=bsps_path,
    )

    # tracker = dict()
    # with open("dummy_data/trascker.yaml", "r") as f:
    #     tracker = yaml.safe_load(f)

    # # Remove the extra entry from the dictionary
    # tracker.pop(
    #     "To use this data in a Python script, you can use a YAML library to read the data from the file and load it into a dictionary. Here's an example using the PyYAML library",
    #     None,
    # )
    # print(tracker)
    # # for key, item in tracker.items():
    # #     print(f"{key}: {item}")
    # fig = plot_simulation_results(tracker, strategy)
    # plt.show()
