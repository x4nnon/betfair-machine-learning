import os
import flumine
import betfairlightweight
import pandas as pd
import numpy as np
import os
import time
from betfairlightweight import StreamListener
import logging
import pickle
import copy

from flumine import Flumine, FlumineSimulation, clients, utils
from flumine import BaseStrategy
from flumine.order.trade import Trade
from flumine.order.order import LimitOrder, MarketOnCloseOrder, OrderStatus
from flumine.markets.market import Market
from betfairlightweight.filters import streaming_market_filter
from betfairlightweight.resources import MarketBook
from betfairlightweight import StreamListener

from sklearn.metrics import confusion_matrix

from typing import Optional, Tuple, Callable, Union

column_names = ["SecondsToStart",
                    "MarketId",
                    "SelectionId",
                    "MarketTotalMatched",
                    "SelectionTotalMatched",
                    "LastPriceTraded",
                    "volume_last_price",
                    "available_to_back_1_price",
                    "available_to_back_1_size",
                    "volume_traded_at_Bprice1",
                    "available_to_back_2_price",
                    "available_to_back_2_size",
                    "volume_traded_at_Bprice2",
                    "available_to_back_3_price",
                    "available_to_back_3_size",
                    "volume_traded_at_Bprice3",
                    "reasonable_back_WoM",
                    "available_to_lay_1_price",
                    "available_to_lay_1_size",
                    "volume_traded_at_Lprice1",
                    "available_to_lay_2_price",
                    "available_to_lay_2_size",
                    "volume_traded_at_Lprice2",
                    "available_to_lay_3_price",
                    "available_to_lay_3_size",
                    "volume_traded_at_Lprice3",
                    "reasonable_lay_WoM"
                    ]

TIME_BRACKETS = [120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345, 360, 390, 420, 450, 480, 510, 540, 570, 600, 700, 800, 900, 1200, 1500, 1800, 2700, 3600, 7200, 14400, 100000]

def extract_files(month_folder):
    "this takes the month folder directly from the download and extracts the .bz files"
    for day in os.listdir(month_folder):
        day_dir = os.path.join(month_folder, day)
        for event in os.listdir(day_dir):
            event_dir = os.path.join(day_dir, event)
            for file in os.listdir(event_dir):
                archive_path = os.path.join(event_dir, file)
                outfile_path = os.path.join(event_dir, file[:-4])
                with open(archive_path, 'rb') as source, open(outfile_path, 'wb') as dest:
                    dest.write(bz2.decompress(source.read()))
                os.remove(archive_path)


def collect_files(month_folder, results_folder_path):
    "this takes the month folder and puts all the files into a results folder path"
    for day in os.listdir(month_folder):
        day_dir = os.path.join(month_folder, day)
        for event in os.listdir(day_dir):
            event_dir = os.path.join(day_dir, event)
            for file in os.listdir(event_dir):
                file_path = os.path.join(event_dir, file)
                outfile_path = os.path.join(results_folder_path, file)
                os.replace(file_path, outfile_path)


def to_combined_csv(wins_folder, wins_combined_path):
    """This takes the folder of collected files from collect_files and turns them into a combined csv for that market
     note that the wins_folder is the results_folder_path, and the wins_combined_path is the combined_path"""

    logging.basicConfig(level=logging.INFO)
    # create trading instance (don't need username/password)
    trading = betfairlightweight.APIClient("username", "password", app_key="d5b6fltDUE03k4lQ")
    # create listener
    listener = StreamListener(max_latency=None)

    # below is an adapted version of marketFolder_to_csv from historic_data_processing_functions

    # define column names for pd
    column_names = ["SecondsToStart",
                    "MarketId",
                    "SelectionId",
                    "MarketTotalMatched",
                    "SelectionTotalMatched",
                    "LastPriceTraded",
                    "volume_last_price",
                    "available_to_back_1_price",
                    "available_to_back_1_size",
                    "volume_traded_at_Bprice1",
                    "available_to_back_2_price",
                    "available_to_back_2_size",
                    "volume_traded_at_Bprice2",
                    "available_to_back_3_price",
                    "available_to_back_3_size",
                    "volume_traded_at_Bprice3",
                    "reasonable_back_WoM",
                    "available_to_lay_1_price",
                    "available_to_lay_1_size",
                    "volume_traded_at_Lprice1",
                    "available_to_lay_2_price",
                    "available_to_lay_2_size",
                    "volume_traded_at_Lprice2",
                    "available_to_lay_3_price",
                    "available_to_lay_3_size",
                    "volume_traded_at_Lprice3",
                    "reasonable_lay_WoM"
                    ]
    j = 0
    for specific in os.listdir(wins_folder):
        spec_path = os.path.join(wins_combined_path, specific)
        print(j, "/", len(os.listdir(wins_folder)))
        j += 1
        file_path = os.path.join(wins_folder, specific)  # market
        if specific in os.listdir(wins_combined_path):
            if "{0}_combined.csv".format(specific) in os.listdir(spec_path):
                pass
        else:
            # create historical stream (update file_path to your file location)
            stream = trading.streaming.create_historical_generator_stream(
                file_path=file_path,
                listener=listener)

            # create generator
            gen = stream.get_generator()

            # set up master list of lists, we will populate this as per the columns
            master_list = []
            counter = 0
            master_counter = 0
            for market_books in gen():
                for market_book in market_books:  # this is only one.
                    for runner in market_book.runners:
                        temp_list = []
                        counter = counter + 1
                        master_counter = master_counter + 1
                        # how to get runner details from the market definition
                        market_def = market_book.market_definition
                        seconds_to_start = (
                                market_book.market_definition.market_time - market_book.publish_time
                        ).total_seconds()
                        # runners_dict = {
                        #     (runner.selection_id, runner.handicap): runner
                        #     for runner in market_def.runners
                        # }
                        # runner_def = runners_dict.get((runner.selection_id, runner.handicap,
                        #                                runner.total_matched, runner.last_price_traded,
                        #                               runner.sp, runner.ex))

                        temp_list = [seconds_to_start,
                                     market_book.market_id,
                                     runner.selection_id,
                                     market_book.total_matched,
                                     runner.total_matched,
                                     runner.last_price_traded or "",
                                     ]

                        # Set up the dictionaries
                        back_dict = {}
                        lay_dict = {}
                        volume_dict = {}

                        # this seems strange because we could just implement the back and lays directly
                        # however because we need to match the prices to volumes traded this is easier.
                        for i in range(len(runner.ex.available_to_back)):
                            back_dict[runner.ex.available_to_back[i]["price"]] = runner.ex.available_to_back[i]["size"]
                        for i in range(len(runner.ex.available_to_lay)):
                            lay_dict[runner.ex.available_to_lay[i]["price"]] = runner.ex.available_to_lay[i]["size"]
                        for i in range(len(runner.ex.traded_volume)):
                            volume_dict[runner.ex.traded_volume[i]["price"]] = runner.ex.traded_volume[i]["size"]

                        # Below will check traded volume at the last price traded
                        if temp_list[5] != "":
                            if temp_list[
                                5] in volume_dict.keys():  # need the extra for an edge case (doesn't seem like this could ever happen logically)
                                temp_list.append(volume_dict[temp_list[5]])
                            else:
                                temp_list.append(0)
                        else:
                            temp_list.append(0)

                        back_prices = list(back_dict.keys())
                        lay_prices = list(lay_dict.keys())
                        reasonable_back_WoM = 0
                        reasonable_lay_WoM = 0
                        # back
                        for i in range(3):
                            if i < len(back_prices) - 1:
                                temp_list.extend([back_prices[i],
                                                  back_dict[back_prices[i]],
                                                  ])
                                if back_prices[i] in volume_dict.keys():
                                    temp_list.append(volume_dict[back_prices[i]])
                                else:
                                    temp_list.append("")
                                reasonable_back_WoM += back_dict[back_prices[i]]
                            else:
                                temp_list.extend(["", "", ""])
                        temp_list.append(reasonable_back_WoM)
                        # lay
                        for i in range(3):
                            if i < len(lay_prices) - 1:
                                temp_list.extend([lay_prices[i],
                                                  lay_dict[lay_prices[i]],
                                                  ])
                                if lay_prices[i] in volume_dict.keys():
                                    temp_list.append(volume_dict[lay_prices[i]])
                                else:
                                    temp_list.append("")
                                reasonable_lay_WoM += lay_dict[lay_prices[i]]
                            else:
                                temp_list.extend(["", "", ""])
                        temp_list.append(reasonable_lay_WoM)

                        master_list.append(temp_list)
            new_dir = os.path.join(wins_combined_path, specific)
            if not os.path.isdir(new_dir):
                os.mkdir(new_dir)
            df_combined = pd.DataFrame(master_list, columns=column_names)

            df_combined.to_csv(new_dir + "/{0}_combined.csv".format(specific))


def create_individual_selection_csvs(combined_path):
    "takes the path of the combined csvs and makes individual selection id files"

    column_names = ["index",
                    "SecondsToStart", "MarketId", "SelectionId", "MarketTotalMatched",
                    "SelectionTotalMatched", "LastPriceTraded", "volume_last_price",
                    "available_to_back_1_price", "available_to_back_1_size",
                    "volume_traded_atBprice1", "available_to_back_2_price",
                    "available_to_back_2_size", "volume_traded_atBprice2",
                    "available_to_back_3_price",
                    "available_to_back_3_size", "volume_traded_atBprice3",
                    "reasonable_back_WoM",
                    "available_to_lay_1_price", "available_to_lay_1_size",
                    "volume_traded_atLprice1", "available_to_lay_2_price",
                    "available_to_lay_2_size", "volume_traded_atLprice2",
                    "available_to_lay_3_price",
                    "available_to_lay_3_size", "volume_traded_atLprice3",
                    "reasonable_lay_WoM", ]
    i = 0
    for folder in os.listdir(combined_path):
        print(i, "/", len(os.listdir(combined_path)))
        i += 1
        folder_path = os.path.join(combined_path, folder)
        file_path = os.path.join(folder_path, "{0}_combined.csv".format(folder))
        combined_df = pd.read_csv(file_path)
        selection_ids = combined_df["SelectionId"].unique()
        temp_dict = {}
        for selection_id in selection_ids:
            selection_df = combined_df[combined_df["SelectionId"] == selection_id]
            new_file_path = os.path.join(folder_path, str(selection_id) + ".csv")  # horrible hacky method sorry.

            selection_df.columns = column_names
            selection_df = selection_df.drop(["index", "available_to_back_3_price",
                                              "available_to_back_3_size",
                                              "volume_traded_atBprice3",
                                              "available_to_lay_3_price",
                                              "available_to_lay_3_size",
                                              "volume_traded_atLprice3"
                                              ], axis=1)

            selection_df.to_csv(new_file_path, index=False)



def combine_bsp_files(bsp_folder):
    "takes the bsp folder and creates a combined csv with all of the bsps in"
    bsp_list = []
    for f in os.listdir(bsp_folder):
        bsp_file = os.path.join(bsp_folder, f)
        df = pd.read_csv(bsp_file)
        bsp_list.append(df)

    df_cc = pd.concat(bsp_list)
    df_cc.to_csv(bsp_folder + "/combined_bsp.csv")


def check_bsp_and_remove_files(df_cc, combined_path):
    """This takes the bsp_df as df_cc and the combined_path, it compares to make sure the bsps for each selection
    exist - if it doesn't then it removes it from the files we have"""

    bsp_dict = {}
    bsp_dict["EVENT_ID"] = []
    bsp_dict["SELECTION_ID"] = []
    bsp_dict["BSP"] = []

    too_many = 0
    missing = 0
    for folder in os.listdir(combined_path):
        folder_path = os.path.join(combined_path, folder)
        market_compare = folder[2:]  # check this
        # print(market_compare)
        for file in os.listdir(folder_path):
            if (file[1] != ".") & (file[0] != "."):  # make sure it's not the combined or lock file
                selection_id = file[:-4]
                # print("-- ", selection_id)
                bsp_line = df_cc[(df_cc["EVENT_ID"] == int(market_compare)) & (df_cc["SELECTION_ID"] == int(selection_id))]
                if len(bsp_line["BSP"].values) == 0:
                    missing += 1
                    # Remove the missing bsps since there isn't much to do with them
                    # these might have even been non runners - who knows.
                    remove_file = os.path.join(folder_path, file)
                    os.remove(remove_file)
                elif len(bsp_line["BSP"].values) > 1:
                    too_many += 1
                else:
                    bsp = bsp_line["BSP"].values[0]

                    bsp_dict["EVENT_ID"].append(folder)
                    bsp_dict["SELECTION_ID"].append(selection_id)
                    bsp_dict["BSP"].append(float(bsp))

    bsp_df = pd.DataFrame.from_dict(bsp_dict)
    bsp_df.to_csv(bsp_folder + "/bsp_df.csv", index=False)

    print("too many", too_many)
    print("missing", missing)  # Note this will only show anything in the first run through


# because we remove the files if they aren't there.

def market_stream_to_analysis(market_id, market_stream_folder, min_selection_traded, folder_for_analysis):
    """
    Pass the market_id and this will grab the market_stream_file which has been recording
    returns True or False depending on it the analyses exists.

    """
    file_name_check = "{}.csv".format(market_id)
    temp_path = folder_for_analysis # change this for the aws
    #     if not os.path.isdir(temp_path):
    #         os.mkdir(temp_path)

    if market_id not in os.listdir(market_stream_folder):
        print("error: market_id ", market_id, " not in market stream folder")
        return False

    elif (file_name_check in os.listdir(temp_path)):
        print("analyses_dict already exists")
        return True

    else:
        trading = betfairlightweight.APIClient("username", "password", app_key="d5b6fltDUE03k4lQ")
        # create listener
        listener = StreamListener(max_latency=None)

        # below is an adapted version of marketFolder_to_csv from historic_data_processing_functions

        # define column names for pd
        column_names = ["SecondsToStart",
                        "MarketId",
                        "SelectionId",
                        "MarketTotalMatched",
                        "SelectionTotalMatched",
                        "LastPriceTraded",
                        "volume_last_price",
                        "available_to_back_1_price",
                        "available_to_back_1_size",
                        "volume_traded_at_Bprice1",
                        "available_to_back_2_price",
                        "available_to_back_2_size",
                        "volume_traded_at_Bprice2",
                        "available_to_back_3_price",
                        "available_to_back_3_size",
                        "volume_traded_at_Bprice3",
                        "reasonable_back_WoM",
                        "available_to_lay_1_price",
                        "available_to_lay_1_size",
                        "volume_traded_at_Lprice1",
                        "available_to_lay_2_price",
                        "available_to_lay_2_size",
                        "volume_traded_at_Lprice2",
                        "available_to_lay_3_price",
                        "available_to_lay_3_size",
                        "volume_traded_at_Lprice3",
                        "reasonable_lay_WoM"
                        ]
        file_path = os.path.join(market_stream_folder, market_id)
        # create historical stream (update file_path to your file location)
        stream = trading.streaming.create_historical_generator_stream(
            file_path=file_path,
            listener=listener)

        # create generator
        gen = stream.get_generator()
        master_list = []
        for market_books in gen():
            for market_book in market_books:  # this is only one.
                for runner in market_book.runners:
                    temp_list = []
                    # how to get runner details from the market definition
                    market_def = market_book.market_definition
                    seconds_to_start = (
                            market_book.market_definition.market_time - market_book.publish_time
                    ).total_seconds()
                    # runners_dict = {
                    #     (runner.selection_id, runner.handicap): runner
                    #     for runner in market_def.runners
                    # }
                    # runner_def = runners_dict.get((runner.selection_id, runner.handicap,
                    #                                runner.total_matched, runner.last_price_traded,
                    #                               runner.sp, runner.ex))

                    temp_list = [seconds_to_start,
                                 market_book.market_id,
                                 runner.selection_id,
                                 market_book.total_matched,
                                 runner.total_matched,
                                 runner.last_price_traded or "",
                                 ]

                    # Set up the dictionaries
                    back_dict = {}
                    lay_dict = {}
                    volume_dict = {}

                    # this seems strange because we could just implement the back and lays directly
                    # however because we need to match the prices to volumes traded this is easier.
                    for i in range(len(runner.ex.available_to_back)):
                        back_dict[runner.ex.available_to_back[i]["price"]] = runner.ex.available_to_back[i]["size"]
                    for i in range(len(runner.ex.available_to_lay)):
                        lay_dict[runner.ex.available_to_lay[i]["price"]] = runner.ex.available_to_lay[i]["size"]
                    for i in range(len(runner.ex.traded_volume)):
                        volume_dict[runner.ex.traded_volume[i]["price"]] = runner.ex.traded_volume[i]["size"]

                    # Below will check traded volume at the last price traded
                    if temp_list[5] != "":
                        if temp_list[
                            5] in volume_dict.keys():  # need the extra for an edge case (doesn't seem like this could ever happen logically)
                            temp_list.append(volume_dict[temp_list[5]])
                        else:
                            temp_list.append(0)
                    else:
                        temp_list.append(0)

                    back_prices = list(back_dict.keys())
                    lay_prices = list(lay_dict.keys())
                    reasonable_back_WoM = 0
                    reasonable_lay_WoM = 0
                    # back
                    for i in range(3):
                        if i < len(back_prices) - 1:
                            temp_list.extend([back_prices[i],
                                              back_dict[back_prices[i]],
                                              ])
                            if back_prices[i] in volume_dict.keys():
                                temp_list.append(volume_dict[back_prices[i]])
                            else:
                                temp_list.append("")
                            reasonable_back_WoM += back_dict[back_prices[i]]
                        else:
                            temp_list.extend(["", "", ""])
                    temp_list.append(reasonable_back_WoM)
                    # lay
                    for i in range(3):
                        if i < len(lay_prices) - 1:
                            temp_list.extend([lay_prices[i],
                                              lay_dict[lay_prices[i]],
                                              ])
                            if lay_prices[i] in volume_dict.keys():
                                temp_list.append(volume_dict[lay_prices[i]])
                            else:
                                temp_list.append("")
                            reasonable_lay_WoM += lay_dict[lay_prices[i]]
                        else:
                            temp_list.extend(["", "", ""])
                    temp_list.append(reasonable_lay_WoM)
                    master_list.append(temp_list)
        df_combined = pd.DataFrame(master_list, columns=column_names)

        # We have now got the combined file for the market - now we need the selection.

        selection_ids = df_combined["SelectionId"].unique()
        temp_dict = {}
        selection_df_dict = {}
        for selection_id in selection_ids:
            selection_df = df_combined[df_combined["SelectionId"] == selection_id]
            # print(selection_df.head())
            new_file_path = os.path.join(temp_path, str(selection_id) + ".csv")  # horrible hacky method sorry.
            # selection_df = selection_df.drop("Unnamed: 0", axis=1) #only if saving / loading
            #             for column in selection_df.columns:
            #                 selection_df = selection_df.rename(columns={column:column+"_{0}".format(selection_id)})

            selection_df_dict[selection_id] = selection_df
            # selection_df.to_csv(new_file_path)

        # Start the analyses part
        time_brackets = [300, 450, 600, 900, 1800, 2700, 3600, 7200, 14400, 100000]
        analysis_dict = {}
        analysis_dict["market_id"] = []
        analysis_dict["selection_ids"] = []

        for time in time_brackets[:-1]:
            analysis_dict["mean_{}".format(time)] = []
            analysis_dict["std_{}".format(time)] = []
            analysis_dict["volume_{}".format(time)] = []

        for selection_id in selection_ids:
            pricing_dict = {}
            analysis_dict["market_id"].append(market_id)
            analysis_dict["selection_ids"].append(selection_id)
            df = selection_df_dict[selection_id]
            df = df.loc[df["SelectionTotalMatched"] > min_selection_traded]
            for i in range(len(time_brackets) - 1):
                # print(all_prices_traded)
                # filter all before a certain time to off feel free to explore this value.
                upper_prev = None
                lower_prev = None

                time = time_brackets[i]
                time2 = time_brackets[i + 1]
                # filter to only certain periods
                df_new = df.loc[df["SecondsToStart"] > time]
                df_new2 = df_new.loc[df_new["SecondsToStart"] > time2]

                all_prices_traded = df_new["LastPriceTraded"].unique()
                first = True  # to avoid nan

                # init to calculate means
                u_v_x_volume = 0
                total_volume = 0
                median_list = []

                if len(all_prices_traded) < 3:
                    # print("less than 2 prices traded")
                    analysis_dict["mean_{}".format(time)].append(0)
                    analysis_dict["std_{}".format(time)].append(0)
                    analysis_dict["volume_{}".format(time)].append(0)

                else:
                    for u_v in all_prices_traded:
                        if not first:
                            # print(u_v)
                            grouped_df = df_new.groupby("LastPriceTraded")
                            volume1 = grouped_df.get_group(u_v).tail(1)["volume_last_price"].values[0]
                            grouped_df2 = df_new2.groupby("LastPriceTraded")
                            if u_v in df_new2["LastPriceTraded"].unique():
                                volume2 = grouped_df2.get_group(u_v).tail(1)["volume_last_price"].values[0]
                            else:
                                volume2 = 0

                            # if testing cont
                            # volume2 = 0

                            if volume1 - volume2 != 0:
                                median_list.append(u_v)

                            pricing_dict[u_v] = volume1 - volume2
                            u_v_x_volume += u_v * pricing_dict[u_v]
                            total_volume += pricing_dict[u_v]

                        first = False

                    # Calculate the weighted mean
                    mean = u_v_x_volume / total_volume

                    # calculate the std
                    sum_squared_diff = 0
                    for u_v in pricing_dict.keys():
                        if (pricing_dict[u_v] < 0) or (total_volume < 0):
                            # print(" ----------- a negative price traded ---------- ")
                            break
                        sq_diff = ((mean - u_v) ** 2) * (pricing_dict[u_v])  # times by amount traded
                        sum_squared_diff += sq_diff
                    var = sum_squared_diff / (total_volume - 1)
                    std = np.sqrt(var)

                    analysis_dict["mean_{}".format(time)].append(mean)
                    analysis_dict["std_{}".format(time)].append(std)
                    analysis_dict["volume_{}".format(time)].append(total_volume)



        # might need for our analyses.

        analyses_df = pd.DataFrame(analysis_dict)
        analyses_path = os.path.join(temp_path, "{}.csv".format(market_id))
        analyses_df.to_csv(analyses_path)

        return True


def combined_path_selection_to_analysis(combined_path, csv_name, bsp_df, min_selection_traded=1000, max_price=30,
                              time_brackets=TIME_BRACKETS):
    """ This takes the combined_path which has the selection csvs in it and creates the analysis files
    Therefore this is not direct"""
    number_files = len(os.listdir(combined_path))

    timed_dict_mean_cont = {}
    timed_dict_std_cont = {}
    means_from_bsp_dict = {}

    analysis_dict = {}
    analysis_dict["market_id"] = []
    analysis_dict["selection_ids"] = []
    analysis_dict["bsps"] = []
    for time in time_brackets[:-1]:
        analysis_dict["mean_{}".format(time)] = []
        analysis_dict["std_{}".format(time)] = []
        analysis_dict["volume_{}".format(time)] = []

    horse_counter = 0
    j = 1
    means_from_bsp_list = []
    for folder in os.listdir(combined_path):
        j += 1
        folder_dir = os.path.join(combined_path, folder)

        for file in os.listdir(folder_dir):
            if file[-5] == "d":
                pass
            else:
                pricing_dict = {}
                file_path = os.path.join(folder_dir, file)
                selection_id = file[:-4]
                analysis_dict["market_id"].append(folder)
                analysis_dict["selection_ids"].append(selection_id)

                df = pd.read_csv(file_path) # this is the selection_df
                df = df.loc[df["SelectionTotalMatched"] > min_selection_traded]
                for i in range(len(time_brackets) - 1):
                    # print(all_prices_traded)
                    # filter all before a certain time to off feel free to explore this value.
                    upper_prev = None
                    lower_prev = None

                    time = time_brackets[i]
                    time2 = time_brackets[i + 1]
                    # filter to only certain periods
                    df_new = df.loc[df["SecondsToStart"] > time]
                    df_new2 = df_new.loc[df_new["SecondsToStart"] > time2]
                    all_prices_traded = df_new["LastPriceTraded"].unique()
                    first = True  # to avoid nan

                    # init to calculate means
                    u_v_x_volume = 0
                    total_volume = 0
                    median_list = []

                    if len(all_prices_traded) < 3:
                        # print("less than 2 prices traded")
                        analysis_dict["mean_{}".format(time)].append(0)
                        analysis_dict["std_{}".format(time)].append(0)
                        analysis_dict["volume_{}".format(time)].append(0)
                    else:
                        for u_v in all_prices_traded:
                            if not first:
                                # print(u_v)
                                grouped_df = df_new.groupby("LastPriceTraded")
                                volume1 = grouped_df.get_group(u_v).tail(1)["volume_last_price"].values[0]
                                grouped_df2 = df_new2.groupby("LastPriceTraded")
                                if u_v in df_new2["LastPriceTraded"].unique():
                                    volume2 = grouped_df2.get_group(u_v).tail(1)["volume_last_price"].values[0]
                                else:
                                    volume2 = 0

                                # if testing cont
                                # volume2 = 0

                                if volume1 - volume2 != 0:
                                    median_list.append(u_v)

                                pricing_dict[u_v] = volume1 - volume2
                                u_v_x_volume += u_v * pricing_dict[u_v]
                                total_volume += pricing_dict[u_v]

                            first = False

                        # Calculate the weighted mean
                        mean = u_v_x_volume / total_volume

                        # calculate the std
                        sum_squared_diff = 0
                        for u_v in pricing_dict.keys():
                            if (pricing_dict[u_v] < 0) or (total_volume < 0):
                                # print(" ----------- a negative price traded ---------- ")
                                break
                            sq_diff = ((mean - u_v) ** 2) * (pricing_dict[u_v])  # times by amount traded
                            sum_squared_diff += sq_diff
                        var = sum_squared_diff / (total_volume - 1)
                        std = np.sqrt(var)

                        analysis_dict["mean_{}".format(time)].append(mean)
                        analysis_dict["std_{}".format(time)].append(std)
                        analysis_dict["volume_{}".format(time)].append(total_volume)

                    # BSP
                bsp_value = bsp_df.loc[(bsp_df["EVENT_ID"] == float(folder)) &
                                       (bsp_df["SELECTION_ID"] == int(file[:-4]))]["BSP"].values[0]

                analysis_dict["bsps"].append(bsp_value)

            horse_counter += 1
            if (horse_counter % 200) == 0:
                print("{}/{}".format(j, number_files))
                print("horses ", horse_counter)

    analysis_df = pd.DataFrame(analysis_dict)
    analysis_df.to_csv(csv_name)


def get_price(data: list, level: int) -> Optional[float]:
    try:
        return data[level]["price"]
    except KeyError:
        return
    except IndexError:
        return
    except TypeError:
        return


def to_analysis_direct(wins_folder, bsp_combined_file, csv_name, min_selection_traded=1000, max_price=30,
                              time_brackets=TIME_BRACKETS):
    """This takes the folder which has the collected wins and the combined_bsp file and then outputs the analysis
    file directly without having to save or load to csv which is faster."""

    combined_bsp = pd.read_csv(bsp_combined_file)

    logging.basicConfig(level=logging.INFO)
    # create trading instance (don't need username/password)
    trading = betfairlightweight.APIClient("username", "password", app_key="d5b6fltDUE03k4lQ")
    # create listener
    listener = StreamListener(max_latency=None)

    market_nonrunner_drop_list = []
    horse_counter = 0


    # below is an adapted version of marketFolder_to_csv from historic_data_processing_functions

    # define column names for pd
    column_names = ["SecondsToStart",
                    "MarketId",
                    "SelectionId",
                    "MarketTotalMatched",
                    "SelectionTotalMatched",
                    "LastPriceTraded",
                    "volume_last_price",
                    "available_to_back_1_price",
                    "available_to_back_1_size",
                    "volume_traded_at_Bprice1",
                    "available_to_back_2_price",
                    "available_to_back_2_size",
                    "volume_traded_at_Bprice2",
                    "available_to_back_3_price",
                    "available_to_back_3_size",
                    "volume_traded_at_Bprice3",
                    "reasonable_back_WoM",
                    "available_to_lay_1_price",
                    "available_to_lay_1_size",
                    "volume_traded_at_Lprice1",
                    "available_to_lay_2_price",
                    "available_to_lay_2_size",
                    "volume_traded_at_Lprice2",
                    "available_to_lay_3_price",
                    "available_to_lay_3_size",
                    "volume_traded_at_Lprice3",
                    "reasonable_lay_WoM"
                    ]

    analysis_dict = {}
    analysis_dict["market_id"] = []
    analysis_dict["selection_ids"] = []
    analysis_dict["bsps"] = []
    for time in time_brackets[:-1]:
        analysis_dict["mean_{}".format(time)] = []
        analysis_dict["std_{}".format(time)] = []
        analysis_dict["volume_{}".format(time)] = []
        analysis_dict["RWoML_{}".format(time)] = []
        analysis_dict["RWoMB_{}".format(time)] = []

    j = 0
    number_files = len(os.listdir(wins_folder))
    for specific in os.listdir(wins_folder):
        print(j, "/", len(os.listdir(wins_folder)))
        j += 1
        file_path = os.path.join(wins_folder, specific)  # market
        # create historical stream (update file_path to your file location)
        stream = trading.streaming.create_historical_generator_stream(
            file_path=file_path,
            listener=listener)

        # create generator
        gen = stream.get_generator()

        # set up master list of lists, we will populate this as per the columns
        master_list = []
        counter = 0
        master_counter = 0

        first_runners = True
        for market_books in gen():
            for market_book in market_books:  # this is only one.

                seconds_to_start = (
                        market_book.market_definition.market_time - market_book.publish_time
                ).total_seconds()

                if first_runners:
                    number_runners = market_book.number_of_active_runners
                    print(market_book.number_of_active_runners)
                    first_runners = False
                else:
                    if seconds_to_start > 0:
                        if number_runners != market_book.number_of_active_runners:
                            if market_book.market_id not in market_nonrunner_drop_list:
                                print("{} has a non_runner".format(market_book.market_id))
                                market_nonrunner_drop_list.append(market_book.market_id)
                for runner in market_book.runners:
                    temp_list = []
                    counter = counter + 1
                    master_counter = master_counter + 1
                    # how to get runner details from the market definition
                    market_def = market_book.market_definition
                    seconds_to_start = (
                            market_book.market_definition.market_time - market_book.publish_time
                    ).total_seconds()
                    # runners_dict = {
                    #     (runner.selection_id, runner.handicap): runner
                    #     for runner in market_def.runners
                    # }
                    # runner_def = runners_dict.get((runner.selection_id, runner.handicap,
                    #                                runner.total_matched, runner.last_price_traded,
                    #                               runner.sp, runner.ex))

                    temp_list = [seconds_to_start,
                                 market_book.market_id,
                                 runner.selection_id,
                                 market_book.total_matched,
                                 runner.total_matched,
                                 runner.last_price_traded or "",
                                 ]

                    # Set up the dictionaries
                    back_dict = {}
                    lay_dict = {}
                    volume_dict = {}

                    # this seems strange because we could just implement the back and lays directly
                    # however because we need to match the prices to volumes traded this is easier.
                    for i in range(len(runner.ex.available_to_back)):
                        back_dict[runner.ex.available_to_back[i]["price"]] = runner.ex.available_to_back[i]["size"]
                    for i in range(len(runner.ex.available_to_lay)):
                        lay_dict[runner.ex.available_to_lay[i]["price"]] = runner.ex.available_to_lay[i]["size"]
                    for i in range(len(runner.ex.traded_volume)):
                        volume_dict[runner.ex.traded_volume[i]["price"]] = runner.ex.traded_volume[i]["size"]

                    # Below will check traded volume at the last price traded
                    if temp_list[5] != "":
                        if temp_list[
                            5] in volume_dict.keys():  # need the extra for an edge case (doesn't seem like this could ever happen logically)
                            temp_list.append(volume_dict[temp_list[5]])
                        else:
                            temp_list.append(0)
                    else:
                        temp_list.append(0)

                    back_prices = list(back_dict.keys())
                    lay_prices = list(lay_dict.keys())
                    reasonable_back_WoM = 0
                    reasonable_lay_WoM = 0
                    # back
                    for i in range(3):
                        if i < len(back_prices) - 1:
                            temp_list.extend([back_prices[i],
                                              back_dict[back_prices[i]],
                                              ])
                            if back_prices[i] in volume_dict.keys():
                                temp_list.append(volume_dict[back_prices[i]])
                            else:
                                temp_list.append("")
                            reasonable_back_WoM += back_dict[back_prices[i]]
                        else:
                            temp_list.extend(["", "", ""])
                    temp_list.append(reasonable_back_WoM)
                    # lay
                    for i in range(3):
                        if i < len(lay_prices) - 1:
                            temp_list.extend([lay_prices[i],
                                              lay_dict[lay_prices[i]],
                                              ])
                            if lay_prices[i] in volume_dict.keys():
                                temp_list.append(volume_dict[lay_prices[i]])
                            else:
                                temp_list.append("")
                            reasonable_lay_WoM += lay_dict[lay_prices[i]]
                        else:
                            temp_list.extend(["", "", ""])
                    temp_list.append(reasonable_lay_WoM)

                    master_list.append(temp_list)

        df_combined = pd.DataFrame(master_list, columns=column_names)

        # So we not have the df_combined we take this directly and turn to selections

        selection_ids = df_combined["SelectionId"].unique()
        temp_dict = {}
        for selection_id in selection_ids:
            # check whether our market_id and selection_id are in the bsp_combined
            if int(market_book.market_id[2:]) in combined_bsp["EVENT_ID"].values:
                new_df = combined_bsp[combined_bsp["EVENT_ID"] == int(market_book.market_id[2:])]
                if int(selection_id) in new_df["SELECTION_ID"].values: # then continue :)
                    selection_df = df_combined[df_combined["SelectionId"] == selection_id]

                    selection_df.columns = column_names
                    selection_df = selection_df.drop(["available_to_back_3_price",
                                                      "available_to_back_3_size",
                                                      "volume_traded_at_Bprice3",
                                                      "available_to_lay_3_price",
                                                      "available_to_lay_3_size",
                                                      "volume_traded_at_Lprice3"
                                                      ], axis=1)

                    # So we now have the selection_dfs and we can

                    analysis_dict["market_id"].append(market_book.market_id)
                    analysis_dict["selection_ids"].append(selection_id)

                    df = selection_df.loc[selection_df["SelectionTotalMatched"] > min_selection_traded]
                    for i in range(len(time_brackets) - 1):
                        # print(all_prices_traded)
                        # filter all before a certain time to off feel free to explore this value.
                        upper_prev = None
                        lower_prev = None

                        pricing_dict = {}
                        time = time_brackets[i]
                        time2 = time_brackets[i + 1]
                        # filter to only certain periods
                        df_new = df.loc[df["SecondsToStart"] > time]
                        df_new2 = df_new.loc[df_new["SecondsToStart"] > time2]
                        df_WOM = df.loc[(df["SecondsToStart"] > time) & (df["SecondsToStart"] < time2)]
                        all_prices_traded = df_new["LastPriceTraded"].unique()
                        first = True  # to avoid nan

                        # init to calculate means
                        u_v_x_volume = 0
                        total_volume = 0
                        median_list = []

                        if len(all_prices_traded) < 3:
                            # print("less than 2 prices traded")
                            analysis_dict["mean_{}".format(time)].append(0)
                            analysis_dict["std_{}".format(time)].append(0)
                            analysis_dict["volume_{}".format(time)].append(0)
                            analysis_dict["RWoML_{}".format(time)].append(0)
                            analysis_dict["RWoMB_{}".format(time)].append(0)
                        else:
                            for u_v in all_prices_traded:
                                if not first:
                                    # print(u_v)
                                    grouped_df = df_new.groupby("LastPriceTraded")
                                    volume1 = grouped_df.get_group(u_v).tail(1)["volume_last_price"].values[0]
                                    grouped_df2 = df_new2.groupby("LastPriceTraded")
                                    if u_v in df_new2["LastPriceTraded"].unique():
                                        volume2 = grouped_df2.get_group(u_v).tail(1)["volume_last_price"].values[0]
                                    else:
                                        volume2 = 0

                                    # if testing cont
                                    # volume2 = 0

                                    if volume1 - volume2 != 0:
                                        median_list.append(u_v)

                                    pricing_dict[u_v] = volume1 - volume2
                                    u_v_x_volume += u_v * pricing_dict[u_v]
                                    total_volume += pricing_dict[u_v]

                                first = False

                            # Calculate the weighted mean
                            mean = u_v_x_volume / total_volume

                            # calculate the std
                            sum_squared_diff = 0
                            for u_v in pricing_dict.keys():
                                if (pricing_dict[u_v] < 0) or (total_volume < 0):
                                    # print(" ----------- a negative price traded ---------- ")
                                    break
                                sq_diff = ((mean - u_v) ** 2) * (pricing_dict[u_v])  # times by amount traded
                                sum_squared_diff += sq_diff
                            var = sum_squared_diff / (total_volume - 1)
                            std = np.sqrt(var)

                            analysis_dict["mean_{}".format(time)].append(mean)
                            analysis_dict["std_{}".format(time)].append(std)
                            analysis_dict["volume_{}".format(time)].append(total_volume)

                            mean_RWoML = df_WOM["reasonable_lay_WoM"].mean()
                            mean_RWoMB = df_WOM["reasonable_back_WoM"].mean()
                            
                            analysis_dict["RWoML_{}".format(time)].append(mean_RWoML)
                            analysis_dict["RWoMB_{}".format(time)].append(mean_RWoMB)

                        # BSP
                    bsp_value = combined_bsp.loc[(combined_bsp["EVENT_ID"] == int(market_book.market_id[2:])) &
                                           (combined_bsp["SELECTION_ID"] == int(selection_id))]["BSP"].values[0]
                    analysis_dict["bsps"].append(bsp_value)

                horse_counter += 1
                print("horse counter is ", horse_counter)
                if (horse_counter % 200) == 0:
                    print("{}/{}".format(j, number_files))
                    print("horses ", horse_counter)
                    print(analysis_dict["market_id"])

    analysis_df = pd.DataFrame(analysis_dict)

    analysis_df.to_csv("temp_{}".format(csv_name))
    print(market_nonrunner_drop_list)
    # remove those with non_runners.
    for drop_market in market_nonrunner_drop_list:
        analysis_df = analysis_df[analysis_df["market_id"] != drop_market]

    analysis_df.to_csv(csv_name)


def normalized_transform(train_df, ticks_df):
    """ This takes the train_df and transform it to add ratios, WoM, and then turns everything
    into ticks and then normalizes everything"""
    # lets now try using ticks and total average? so mean_ticks / total_mean_ticks
    train_df = train_df.dropna()
    train_df = train_df[(train_df["mean_120"] > 1.1) & (train_df["mean_120"] <= 50)]
    train_df = train_df[train_df["mean_14400"] > 0]
    train_df = train_df.drop(train_df[train_df["std_2700"] > 1].index) # slight hack
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
        train_df["WoM_ratio_{}".format(timie)] = train_df[lay_wom_list[i]] / train_df[back_wom_list[i]]
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
        train_df["mean_and_volume_{}".format(timie)] = train_df[mean_list[i]] * train_df[volume_list[i]]
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
        total_vwap_ticks.append(ticks_df.iloc[ticks_df['tick'].sub(row["total_vwap"]).abs().idxmin()]["number"])
        bsps_ticks.append(ticks_df.iloc[ticks_df['tick'].sub(row["bsps"]).abs().idxmin()]["number"])
    for i in range(len(mean_list)):
        timie = lay_wom_list[i].split("_")[1]
        mean_dict[timie] = []
        train_df["std_{}".format(timie)] = train_df["std_{}".format(timie)] / train_df["mean_{}".format(timie)]
        for index, row in train_df.iterrows():
            mean_dict[timie].append(ticks_df.iloc[ticks_df['tick'].sub(row[mean_list[i]]).abs().idxmin()]["number"])


    train_df["total_vwap"] = total_vwap_ticks
    train_df["mean_120_temp"] = train_df["mean_120"]
    for key in mean_dict.keys():
        train_df["mean_{}".format(key)] = mean_dict[key]
        train_df["mean_{}".format(key)] = train_df["mean_{}".format(key)] / train_df["total_vwap"]
        train_df["volume_{}".format(key)] = train_df["volume_{}".format(key)] / train_df["total_volume"]

    try:
        train_df["bsps_temp"] = train_df["bsps"]  # drop this above but needed for margin
        #print(bsps_ticks)
        train_df["bsps"] = bsps_ticks
        train_df["bsps"] = train_df["bsps"] / train_df["total_vwap"]
    except:
        print("no bsps in this df")

    train_df = train_df.drop(["total_volume", "total_vwap"], axis=1)
    train_df = train_df.drop(["Unnamed: 0", "selection_ids", "market_id"], axis=1)

    return train_df
