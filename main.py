from pprint import pprint

import yaml

from onedrive import Onedrive
from flumine import clients
from flumine_simulator import piped_run
from matplotlib import pyplot as plt
from utils.config import app_principal, SITE_URL
import argparse

from utils.utils import get_simulation_plot


plt.rcParams["figure.figsize"] = (20, 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--races", type=int, default=1, help="Number of races to run")
    args = parser.parse_args()

    test_folder_path = "horses_jul_wins"
    bsps_path = "july_22_bsps"
    strategy_name = "Strategy1"

    tracker = piped_run(
        strategy=strategy_name,
        onedrive=Onedrive(
            client_id=app_principal["client_id"],
            client_secret=app_principal["client_secret"],
            site_url=SITE_URL,
        ),
        client=clients.SimulatedClient(),
        test_folder_path=test_folder_path,
        bsps_path=bsps_path,
        model_name="BayesianRidge",
        races=args.races,
        save=True,
    )

    # tracker = dict()
    # with open("results/Strategy1_results.yaml", "r") as f:
    #     tracker = yaml.load(f, Loader=yaml.UnsafeLoader)
    # fig = get_simulation_plot(tracker, "Strategy1")
    # plt.show()
    # metrics = dict()
    # with open("dummy_data/metrics.yaml", "r") as f:
    #     metrics = yaml.safe_load(f)

    # pprint(tracker)
