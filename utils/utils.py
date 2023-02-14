import copy
import os
import joblib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, mean_absolute_error as mae
import seaborn as sns
from onedrive import Onedrive


def process_run_results(results: dict, metrics: dict):
    def print_add(x: int, y: int, msg: str):
        x += y
        print(x, msg)

    prev_key = ""
    for key, item in metrics.items():

        if isinstance(item, list):
            if "actual" in key:
                item.append(metrics["total_profit"])
            if "expected" in key:
                item.append(metrics["total_m_c_marg"] + metrics["total_m_i_marg"])

            if "green" in key:
                item.append(metrics["total_green_margin"])
        else:
            name_lst = key.split("_")

            result_key = "_".join(name_lst[1:])

            if result_key == "m_c_marg" or result_key == "m_i_marg":
                print(f"{key}: {item}")
                continue

            if result_key == "counter":
                print(f"Race: {metrics[key]}")
                continue

            result = results[result_key]

            if "total" in name_lst:
                print_add(item, result, f"{key}: ")

            print(
                f"{key}: {item}"
            ) if result_key == "m_c_marg" or result_key == "m_i_marg" or result_key == "counter" else print(
                f"{result_key}: ", result
            )

            if not result_key in prev_key:
                print("---")

        prev_key = key

    if metrics["race_counter"] % 10 == 0:
        # plt.plot(range(race_counter), actual_profit_plotter, label="backtest", color="b")
        plt.plot(
            range(metrics["race_counter"]),
            metrics["expected_profit_plotter"],
            label="expected",
            color="y",
        )
        plt.plot(
            range(metrics["race_counter"]),
            metrics["green_plotter"],
            label="greened_profit",
            color="g",
        )
        plt.axhline(y=0.5, color="r", linestyle="-")
        plt.xlabel("Number of Races")
        plt.ylabel("Profit")

        plt.legend()
        plt.draw()
        print("")


def rms(y, y_pred):
    rms = np.sqrt(np.mean((y - y_pred) ** 2))
    return rms


def plot_simulation_results(
    tracker: dict, strategy: str, style="darkgrid", palette="dark"
):
    sns.set_style(style)
    sns.set_palette(palette)

    profit_tracker, val_tracker = {}, {}
    for key, value in tracker.items():
        if key in ["actual_profit_plotter", "expected_profit_plotter", "green_plotter"]:
            profit_tracker[key] = value
        else:
            val_tracker[key] = value

    # Create a DataFrame for the val_tracker dictionary
    df = pd.DataFrame.from_dict(val_tracker, orient="index")
    print(df)

    fig = sns.lineplot(data=profit_tracker)
    fig.set(
        title=f"Simulation Results: {strategy}",
        xlabel="Number of Races",
        ylabel="Profit",
    )

    print("Total expected profit is ", tracker["expected_profit_plotter"][-1])

    return fig


def train_model(
    ticks_df: pd.DataFrame,
    onedrive: Onedrive,
    model: str,
    regression=True,
    x_train_path="utils/x_train_df.csv",
    y_train_path="utils/y_train_df.csv",
    model_path="models/",
):
    x_train_df = (
        pd.read_csv(x_train_path, index_col=False).drop("Unnamed: 0", axis=1)
        if os.path.exists(x_train_path)
        else None
    )
    y_train_df = (
        pd.read_csv(y_train_path, index_col=False).drop("Unnamed: 0", axis=1)
        if os.path.exists(y_train_path)
        else None
    )

    if x_train_df is None or y_train_df is None:
        print(
            "x_train_df and/or y_train_df not found, commencing fetch and normalization..."
        )
        train_df = onedrive.get_train_df()
        train_df = normalized_transform(train_df, ticks_df)
        print("Finished train data normalization...")

        mean120_actual_train = train_df["mean_120_temp"]
        if not regression:
            train_df = train_df.drop(["mean_120_temp"], axis=1)

        mean120_train_df = train_df["mean_120"]
        bsp_train_df = train_df["bsps"]
        train_df["bsps"] = ((mean120_train_df - bsp_train_df) > 0).astype(int)

        df_majority = train_df[(train_df["bsps"] == 0)]
        df_minority = train_df[(train_df["bsps"] == 1)]

        # downsample majority
        df_majority = df_majority.head(
            len(df_minority)
        )  # because I don't trust the resample

        # Combine majority class with upsampled minority class
        train_df = pd.concat([df_minority, df_majority])
        mean120_train_df = train_df["mean_120"]

        y_train_df = train_df["bsps"]
        if regression:
            y_train_df = train_df["bsps_temp"]

        y_train_df.to_csv(y_train_path, index=False)

        x_train_df = train_df.drop(["bsps"], axis=1)
        x_train_df = x_train_df.drop(["bsps_temp"], axis=1)

    clm = x_train_df.columns
    scaler = StandardScaler()
    x_train_df = pd.DataFrame(scaler.fit_transform(x_train_df), columns=clm)

    if not os.path.exists(x_train_path):
        x_train_df.to_csv(x_train_path, index=False)

    m = (
        joblib.load(f"{model_path}{model}.pkl")
        if os.path.exists(f"{model_path}{model}.pkl")
        else None
    )

    if m:
        return m, clm, scaler

    else:
        print(f"Commencing model training: {model}...")

        m = BayesianRidge() if model == "BayesianRidge" else BayesianRidge()
        m.fit(x_train_df, y_train_df)
        _ = joblib.dump(m, f"{model_path}{model}.pkl")

        return m, clm, scaler


def test_model():
    test_analysis_df = test_analysis_df.dropna()
    test_analysis_df = test_analysis_df[
        (test_analysis_df["mean_120"] <= 50) & (test_analysis_df["mean_120"] > 1.1)
    ]
    test_analysis_df = test_analysis_df[test_analysis_df["mean_14400"] > 0]
    # below is a slight hack ...
    test_analysis_df = test_analysis_df.drop(
        test_analysis_df[test_analysis_df["std_2700"] > 1].index
    )

    test_analysis_df_y = pd.DataFrame().assign(
        market_id=test_analysis_df["market_id"],
        selection_ids=test_analysis_df["selection_ids"],
        bsps=test_analysis_df["bsps"],
    )
    # Sort out our test the same as before.
    test_df = copy.copy(test_analysis_df)
    test_df = normalized_transform(test_df, ticks_df)

    mean120_actual_test = test_df["mean_120_temp"]
    if not regression:
        test_df = test_df.drop(["mean_120_temp"], axis=1)

    mean120_test_df = test_df["mean_120"]
    bsp_test_df = test_df["bsps"]
    test_df["bsps"] = ((mean120_test_df - bsp_test_df) > 0).astype(int)

    y_test_df = copy.copy(test_df["bsps"])
    if regression:
        y_test_df = test_df["bsps_temp"]
    x_test_df = test_df.drop(["bsps"], axis=1)

    bsp_actual_test = test_df["bsps_temp"]
    x_test_df = x_test_df.drop(["bsps_temp"], axis=1)

    print("TEST ------")
    print(x_test_df)
    x_test_df = pd.DataFrame(scaler.transform(x_test_df), columns=clm)

    # test_analysis_df = test_analysis_df.drop(["bsps"], axis=1)

    #################################################
    # ALL BELOW IS JUST TO CONFIRM IT IS FIT CORRECT:
    y_pred_train = model.predict(x_train_df)
    y_pred_test = model.predict(x_test_df)

    print(y_pred_test)
    print(y_test_df)

    results_train_mae = mae(y_train_df, y_pred_train)

    print("MAE train : ", results_train_mae)

    results_test_mae = mae(y_test_df, y_pred_test)

    print("MAE test : ", results_test_mae)

    if not regression:
        (tn, fp, fn, tp) = confusion_matrix(y_test_df, y_pred_test).ravel()
        print("- tn:", tn, "- fp:", fp, "- fn:", fn, "- tp:", tp)

        confidence = 0.6

        print("with a confidence probability of above ", confidence)

        y_pred_train_proba = model.predict_proba(x_train_df)
        y_pred_test_proba = model.predict_proba(x_test_df)

        neg_pred_after_conf_train = []
        neg_corresponding_label_train = []

        pos_pred_after_conf_train = []
        pos_corresponding_label_train = []

        for i in range(len(y_pred_train_proba)):
            if y_pred_train_proba[i][0] > confidence:
                neg_pred_after_conf_train.append(0)
                neg_corresponding_label_train.append(y_train_df.values[i])
            elif y_pred_train_proba[i][1] > confidence:
                pos_pred_after_conf_train.append(1)
                pos_corresponding_label_train.append(y_train_df.values[i])

        print()
        results_train_mae_neg = mae(
            neg_pred_after_conf_train, neg_corresponding_label_train
        )
        results_train_mae_pos = mae(
            pos_pred_after_conf_train, pos_corresponding_label_train
        )

        print("MAE conf train neg: ", results_train_mae_neg)
        print("MAE conf train pos : ", results_train_mae_pos)

        neg_pred_after_conf_test = []
        neg_corresponding_label_test = []

        pos_pred_after_conf_test = []
        pos_corresponding_label_test = []

        margin = 0
        print("mean_120_actual -------------")
        print(mean120_test_df)
        print("bsps_actual ------------------")
        print(bsp_actual_test)

        for i in range(len(y_pred_test_proba)):
            if y_pred_test_proba[i][0] > confidence:
                neg_pred_after_conf_test.append(0)
                neg_corresponding_label_test.append(y_test_df.values[i])
                margin_calc = (
                    (50 / mean120_actual_test.values[i])
                    * (bsp_actual_test.values[i] - mean120_actual_test.values[i])
                    / mean120_actual_test.values[i]
                )
                margin += margin_calc
            elif y_pred_test_proba[i][1] > confidence:
                pos_pred_after_conf_test.append(1)
                pos_corresponding_label_test.append(y_test_df.values[i])
                margin_calc = (
                    50
                    * (-bsp_actual_test.values[i] + mean120_actual_test.values[i])
                    / mean120_actual_test.values[i]
                )
                margin += margin_calc

        print()
        results_test_mae_neg = mae(
            neg_pred_after_conf_test, neg_corresponding_label_test
        )
        results_test_mae_pos = mae(
            pos_pred_after_conf_test, pos_corresponding_label_test
        )

        print("MAE conf test neg: ", results_test_mae_neg)
        print("MAE conf test pos : ", results_test_mae_pos)

        print("confusion for neg")
        (tn, fp, fn, tp) = confusion_matrix(
            neg_corresponding_label_test, neg_pred_after_conf_test
        ).ravel()
        print("- tn:", tn, "- fp:", fp, "- fn:", fn, "- tp:", tp)

        print("confusion for pos")
        (tn, fp, fn, tp) = confusion_matrix(
            pos_corresponding_label_test, pos_pred_after_conf_test
        ).ravel()
        print("- tn:", tn, "- fp:", fp, "- fn:", fn, "- tp:", tp)

        print("margin for final is :", margin, "stop")

        print("test_analysis_df")
        print(test_analysis_df["mean_120"])

    else:
        y_pred_train = model.predict(x_train_df)
        y_pred_test = model.predict(x_test_df)
        print("regression results are")

        print()
        results_train_mae = mae(y_pred_train, y_train_df)
        results_test_mae = mae(y_pred_test, y_test_df)

        print("MAE train: ", results_train_mae)
        print("MAE test : ", results_test_mae)

        results_train_at_mean_120 = mae(mean120_train_df, y_train_df)
        results_test_at_mean_120 = mae(mean120_test_df, y_test_df)

        print("if chosen at mean 120")
        print("MAE train - 120 : ", results_train_at_mean_120)
        print("MAE test - 120 : ", results_test_at_mean_120)

    return model, test_analysis_df, scaler, clm, test_analysis_df_y


def normalized_transform(train_df, ticks_df):
    """This takes the train_df and transform it to add ratios, WoM, and then turns everything
    into ticks and then normalizes everything"""
    # lets now try using ticks and total average? so mean_ticks / total_mean_ticks
    train_df = train_df.dropna()
    train_df = train_df[(train_df["mean_120"] > 1.1) & (train_df["mean_120"] <= 50)]
    train_df = train_df[train_df["mean_14400"] > 0]
    train_df = train_df.drop(train_df[train_df["std_2700"] > 1].index)  # slight hack
    # These above must match the test_analysis - kind of annoying I know ...

    # find wom columns
    lay_wom_list = []
    back_wom_list = []
    for column in train_df.columns:
        if "RWoML" in column:
            lay_wom_list.append(column)
        elif "RWoMB" in column:
            back_wom_list.append(column)

    # compute ratios and add them to train_df
    for i in range(len(lay_wom_list)):
        timie = lay_wom_list[i].split("_")[1]
        train_df["WoM_ratio_{}".format(timie)] = (
            train_df[lay_wom_list[i]] / train_df[back_wom_list[i]]
        )
        train_df = train_df.drop([lay_wom_list[i], back_wom_list[i]], axis=1)

    # find mean and volume columns
    mean_list = []
    volume_list = []
    for column in train_df.columns:
        if "mean" in column:
            mean_list.append(column)
        elif "volume" in column:
            volume_list.append(column)

    train_df["total_volume"] = 0
    train_df["sum_mean_volume"] = 0
    # compute ratios and add them to train_df
    for i in range(len(mean_list)):
        timie = lay_wom_list[i].split("_")[1]
        train_df["mean_and_volume_{}".format(timie)] = (
            train_df[mean_list[i]] * train_df[volume_list[i]]
        )
        train_df["total_volume"] += train_df[volume_list[i]]
        train_df["sum_mean_volume"] += train_df["mean_and_volume_{}".format(timie)]
        train_df = train_df.drop(["mean_and_volume_{}".format(timie)], axis=1)
    train_df["total_vwap"] = train_df["sum_mean_volume"] / train_df["total_volume"]
    train_df = train_df.drop(["sum_mean_volume"], axis=1)

    # OK now we have a total average ... we need to turn these into ticks

    total_vwap_ticks = []
    bsps_ticks = []
    mean_dict = {}

    for index, row in train_df.iterrows():
        total_vwap_ticks.append(
            ticks_df.iloc[ticks_df["tick"].sub(row["total_vwap"]).abs().idxmin()][
                "number"
            ]
        )
        bsps_ticks.append(
            ticks_df.iloc[ticks_df["tick"].sub(row["bsps"]).abs().idxmin()]["number"]
        )
    for i in range(len(mean_list)):
        timie = lay_wom_list[i].split("_")[1]
        mean_dict[timie] = []
        train_df["std_{}".format(timie)] = (
            train_df["std_{}".format(timie)] / train_df["mean_{}".format(timie)]
        )
        for index, row in train_df.iterrows():
            mean_dict[timie].append(
                ticks_df.iloc[ticks_df["tick"].sub(row[mean_list[i]]).abs().idxmin()][
                    "number"
                ]
            )

    train_df["total_vwap"] = total_vwap_ticks
    train_df["mean_120_temp"] = train_df["mean_120"]
    for key in mean_dict.keys():
        train_df["mean_{}".format(key)] = mean_dict[key]
        train_df["mean_{}".format(key)] = (
            train_df["mean_{}".format(key)] / train_df["total_vwap"]
        )
        train_df["volume_{}".format(key)] = (
            train_df["volume_{}".format(key)] / train_df["total_volume"]
        )

    try:
        train_df["bsps_temp"] = train_df[
            "bsps"
        ]  # drop this above but needed for margin
        # print(bsps_ticks)
        train_df["bsps"] = bsps_ticks
        train_df["bsps"] = train_df["bsps"] / train_df["total_vwap"]
    except:
        print("no bsps in this df")

    train_df = train_df.drop(["total_volume", "total_vwap"], axis=1)
    train_df = train_df.drop(["Unnamed: 0", "selection_ids", "market_id"], axis=1)

    return train_df


if __name__ == "__main__":
    df = pd.read_csv("utils/x_train_df.csv")
    print(df)
