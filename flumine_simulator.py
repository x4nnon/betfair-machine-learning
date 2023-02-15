import logging
import os
import time
from typing import List
from onedrive import Onedrive
from strategies.strategy1 import Strategy1
from flumine import FlumineSimulation, BaseStrategy
from pythonjsonlogger import jsonlogger
from flumine.clients import SimulatedClient

from utils.utils import train_model


def run(strategy: Strategy1, client: SimulatedClient):

    framework = FlumineSimulation(client=client)
    framework.add_strategy(strategy)
    market_files = strategy.market_filter["markets"]

    for index, market_file in enumerate(market_files):
        print(f"Race ({market_file}): {index}/{len(market_files)}")
        market_filter = {"markets": [market_file]}
        strategy.set_market_filter(market_filter=market_filter)
        framework.run()

        metrics = {
            key: []
            for key in strategy.metrics.keys()
            + ["actual_profit_plotter", "expected_profit_plotter", "green_plotter"]
        }

        profit = 0
        for market in framework.markets:
            print(
                "Profit: {0:.2f}".format(
                    sum([o.simulated.profit for o in market.blotter])
                )
            )
            profit += sum([o.simulated.profit for o in market.blotter])

        metrics["actual_profit_plotter"].append(profit)  # NOTE double check this
        metrics["expected_profit_plotter"].append(
            strategy.metrics["m_c_marg"] + strategy.metrics["m_i_marg"]
        )  # NOTE double check this

        for key, value in strategy.metrics.items():
            metrics[key].append[value]

    return metrics


def get_strategy(
    strategy: str, market_file: List[str] | str, onedrive: Onedrive, model_name: str
) -> Strategy1:

    market_file = market_file if isinstance(market_file, list) else [market_file]

    ticks_df = onedrive.get_folder_contents(
        target_folder="ticks", target_file="ticks.csv"
    )
    model, clm, scaler = train_model(
        ticks_df,
        onedrive,
        model=model_name,
    )
    test_analysis_df = onedrive.get_test_df(target_folder="Analysis_files")

    if strategy == "Strategy1":
        strategy_pick = Strategy1(
            model=model,
            ticks_df=ticks_df,
            clm=clm,
            scaler=scaler,
            test_analysis_df=test_analysis_df,
            market_filter={"markets": market_file},
            max_trade_count=100000,
            max_live_trade_count=100000,
            max_order_exposure=10000,
            max_selection_exposure=100000,
        )
        return strategy_pick


def piped_run(
    strategy: str,
    onedrive: Onedrive,
    client: SimulatedClient,
    test_folder_path: str,
    bsps_path: str,
    model_name: str,
):
    logger = logging.getLogger(__name__)
    custom_format = "%(asctime) %(levelname) %(message)"
    log_handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(custom_format)
    formatter.converter = time.gmtime
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    logger.setLevel(logging.CRITICAL)  # Set to logging.CRITICAL to speed up backtest

    tracker = {
        "total_profit": 0,
        "total_matched_correct": 0,
        "total_matched_incorrect": 0,
        "total_back_matched_correct": 0,
        "total_back_matched_incorrect": 0,
        "total_lay_matched_correct": 0,
        "total_lay_matched_incorrect": 0,
        "total_m_c_marg": 0,
        "total_m_i_marg": 0,
        "total_green_margin": 0,
        "total_amount_gambled": 0,
        "actual_profit_plotter": [],
        "expected_profit_plotter": [],
        "green_plotter": [],
        "race_counter": 0,
        "total_q_correct": 0,
        "total_q_incorrect": 0,
        "total_m_correct_margin": 0,
        "total_m_incorrect_margin": 0,
        "total_q_margin": 0,
    }

    bsp_df = onedrive.get_bsps(target_folder=bsps_path)

    test_folder_files = os.listdir(test_folder_path)
    number_files = len(test_folder_files)

    if number_files == 0:
        print("Starting test folder download...")
        onedrive.download_test_folder(target_folder="horses_jul_wins")
        print("Test folder download finished.")

    file_paths = [
        os.path.join(test_folder_path, f_name)
        for f_name in test_folder_files
        if float(f_name) in bsp_df["EVENT_ID"].values
    ]
    strategy_pick = get_strategy(strategy, file_paths, onedrive, model_name)
    results = run(strategy_pick, client)
    # process_run_results(results, tracker)

    print(tracker["total_matched_correct"])
    print(tracker["total_matched_incorrect"])
    print("total correct margin is : ", tracker["total_m_c_marg"])
    print("total incorrect margin is : ", tracker["total_m_i_marg"])

    return tracker
