import flumine
import betfairlightweight
import pandas as pd
import numpy as np
import os
import pickle
import time
import bz2
from pythonjsonlogger import jsonlogger
import copy

from flumine import Flumine, FlumineSimulation, clients, utils
from flumine import BaseStrategy
from flumine.order.trade import Trade
from flumine.order.order import LimitOrder, MarketOnCloseOrder, OrderStatus
from flumine.markets.market import Market
from betfairlightweight.filters import streaming_market_filter
from betfairlightweight.resources import MarketBook
from betfairlightweight import StreamListener

import csv
import logging
from flumine.controls.loggingcontrols import LoggingControl
from flumine.order.ordertype import OrderTypes

from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (20, 3)

from sklearn.linear_model import LinearRegression, Ridge
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import RationalQuadratic

from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix

from toms_utils import market_stream_to_analysis, normalized_transform, get_price
from sklearn.neural_network import MLPClassifier, MLPRegressor

def fit_horse_model(train_df, test_analysis_df, model, regression=False):

    test_analysis_df = test_analysis_df.dropna()
    test_analysis_df = test_analysis_df[(test_analysis_df["mean_120"] <= 50) & (test_analysis_df["mean_120"] > 1.1)]
    test_analysis_df = test_analysis_df[test_analysis_df["mean_14400"] > 0]
    # below is a slight hack ...
    test_analysis_df = test_analysis_df.drop(test_analysis_df[test_analysis_df["std_2700"] > 1].index)

    test_analysis_df_y = pd.DataFrame().assign(market_id=test_analysis_df["market_id"],
                                               selection_ids=test_analysis_df["selection_ids"],
                                               bsps=test_analysis_df["bsps"])
    model = False
    scaler = None
    clm = None

    return model, test_analysis_df, scaler, clm, test_analysis_df_y

class Strategy1(BaseStrategy):
    # back and lay in here
    def start(self) -> None:
        print("starting strategy 'Strategy1'")

        self.column_names = ["SecondsToStart", "MarketId", "SelectionId", "MarketTotalMatched",
                             "SelectionTotalMatched"
            , "LastPriceTraded", "volume_last_price",
                             "available_to_back_1_price", "available_to_back_1_size", "volume_traded_at_Bprice1",
                             "available_to_back_2_price", "available_to_back_2_size", "volume_traded_at_Bprice2",
                             "available_to_back_3_price", "available_to_back_3_size", "volume_traded_at_Bprice3",
                             "reasonable_back_WoM",
                             "available_to_lay_1_price", "available_to_lay_1_size", "volume_traded_at_Lprice1",
                             "available_to_lay_2_price", "available_to_lay_2_size", "volume_traded_at_Lprice2",
                             "available_to_lay_3_price", "available_to_lay_3_size", "volume_traded_at_Lprice3",
                             "reasonable_lay_WoM"
                             ]

        self.back_bet_tracker = {}
        self.matched_back_bet_tracker = {}
        self.lay_bet_tracker = {}
        self.matched_lay_bet_tracker = {}
        self.order_dict_back = {}
        self.order_dict_lay = {}
        self.LP_traded = {}
        self.matched_correct = 0
        self.matched_incorrect = 0
        self.q_correct = 0
        self.q_incorrect = 0
        self.m_incorrect_margin = 0
        self.m_correct_margin = 0
        self.green_margin = 0
        self.seconds_to_start = None
        self.green_orders = []
        self.normal_orders = []
        self.amount_gambled = 0
        self.market_open = True
        self.stake = 30
        self.first_nonrunners = True
        self.runner_number = None
        self.lay_matched_correct = 0
        self.back_matched_correct = 0
        self.lay_matched_incorrect = 0
        self.back_matched_incorrect = 0
        self.q_margin = 0

    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
        # process_market_book only executed if this returns True
        self.process_fundamentals(market_book)

        if self.first_nonrunners:
            self.runner_number = market_book.number_of_active_runners
            self.first_nonrunners = False
        else:
            if market_book.number_of_active_runners != self.runner_number:
                return False  # this will stop any more action happening in this market.

        if (market_book.status == "CLOSED") or (self.seconds_to_start < TIME_BEFORE_START):
            return False
        else:
            return True

    def process_fundamentals(self, market_book):
        # We already have everything we need in an excel file.
        runner_count = 0
        seconds_to_start = (
                market_book.market_definition.market_time - market_book.publish_time
        ).total_seconds()
        self.seconds_to_start = seconds_to_start

        return True

    def process_runners(self):
        # This doesn't need to do anything.
        pass

    def process_market_book(self, market: Market, market_book: MarketBook) -> None:
        # process marketBook object
        # Take each incoming message and combine to a df
        cont = self.process_fundamentals(market_book)

        self.market_open = market_book.status
        # We want to limit our betting to before the start of the race.

        market_id = float(market_book.market_id)
        if self.check_market_book(market, market_book):
            if ((self.seconds_to_start < 28800) & (self.seconds_to_start > 28770)):
                for runner in market_book.runners:
                    # # backs
                    if market_id not in self.back_bet_tracker.keys():
                        self.back_bet_tracker[market_id] = {}
                        self.matched_back_bet_tracker[market_id] = {}
                    if runner.selection_id not in self.back_bet_tracker[market_id].keys():
                        self.back_bet_tracker[market_id][runner.selection_id] = {}
                        self.matched_back_bet_tracker[market_id][runner.selection_id] = {}
                        print("before checking selection_id")
                        if runner.status == "ACTIVE":
                            try:
                                predict_row = test_analysis_df.loc[
                                    (test_analysis_df["selection_ids"] == runner.selection_id)
                                    & (test_analysis_df["market_id"] == market_id)]

                                mean_14400 = predict_row["mean_14400"].values[0]

                            except:
                                print("can't find ", runner.selection_id, " and ", market_id, "LPT was ", runner.last_price_traded)
                                mean_14400 = False

                            if (mean_14400 != False):
                                back_number = ticks_df.iloc[ticks_df['tick'].sub(get_price(runner.ex.available_to_back, 0)).abs().idxmin()][
                                    "number"]
                                back_number_adjust = back_number + 2
                                back_price_adjusted = \
                                    ticks_df.iloc[ticks_df['number'].sub(back_number_adjust).abs().idxmin()][
                                        "tick"]
                                bsp_row = test_analysis_df_y.loc[
                                    (test_analysis_df_y["selection_ids"] == runner.selection_id) &
                                    (test_analysis_df_y["market_id"] == market_id)]
                                bsp_value = bsp_row["bsps"].values[0]

                                if (mean_120 <= 50) & (mean_120 > 1.1):
                                    trade = Trade(
                                        market_id=market_book.market_id,
                                        selection_id=runner.selection_id,
                                        handicap=runner.handicap,
                                        strategy=self,
                                    )

                                    if back_price_adjusted > bsp_value:
                                        self.q_correct += 1
                                    else:
                                        self.q_incorrect += 1

                                    order = trade.create_order(
                                        side="BACK", order_type=LimitOrder(price=back_price_adjusted,
                                                                           size=self.stake,
                                                                           persistence_type="LAPSE"
                                                                           )
                                    )

                                    print("MADE A BACK BET")
                                    self.back_bet_tracker[market_id][runner.selection_id] = [order, bsp_value,
                                                                                             market_book.market_id,
                                                                                             runner.selection_id,
                                                                                             runner.handicap,
                                                                                             runner.sp.actual_sp,
                                                                                             back_price_adjusted]
                                    self.matched_back_bet_tracker[market_id][runner.selection_id] = False

                                    market.place_order(order)

                                    self.q_margin += self.stake * (
                                                back_price_adjusted - bsp_value) / back_price_adjusted



                    if market_id not in self.lay_bet_tracker.keys():
                        self.lay_bet_tracker[market_id] = {}
                        self.matched_lay_bet_tracker[market_id] = {}
                    if runner.selection_id not in self.lay_bet_tracker[market_id].keys():
                        self.lay_bet_tracker[market_id][runner.selection_id] = {}
                        self.matched_lay_bet_tracker[market_id][runner.selection_id] = {}
                        if runner.status == "ACTIVE":
                            try:
                                predict_row = test_analysis_df.loc[
                                    (test_analysis_df["selection_ids"] == runner.selection_id)
                                    & (test_analysis_df["market_id"] == market_id)]
                                mean_14400 = predict_row["mean_14400"].values[0]

                            except:
                                mean_14400 = False

                            if (mean_14400 != False):
                                # in the lay_price / number put in where you want to base from: prevoisly runner.last_price_traded
                                # get_price(runner.ex.available_to_lay, 1)
                                lay_price = ticks_df.iloc[ticks_df['tick'].sub(get_price(runner.ex.available_to_lay, 0)).abs().idxmin()]["tick"]
                                lay_number = ticks_df.iloc[ticks_df['tick'].sub(get_price(runner.ex.available_to_lay, 0)).abs().idxmin()][
                                    "number"]
                                lay_number_adjust = lay_number - 2
                                lay_price_adjusted = \
                                    ticks_df.iloc[ticks_df['number'].sub(lay_number_adjust).abs().idxmin()][
                                        "tick"]
                                bsp_row = test_analysis_df_y.loc[
                                    (test_analysis_df_y["selection_ids"] == runner.selection_id) &
                                    (test_analysis_df_y["market_id"] == market_id)]
                                bsp_value = bsp_row["bsps"].values[0]

                                if (lay_price_adjusted <= self.stake) and (lay_price_adjusted > 1.1):
                                    trade = Trade(
                                        market_id=market_book.market_id,
                                        selection_id=runner.selection_id,
                                        handicap=runner.handicap,
                                        strategy=self,
                                    )

                                    if lay_price_adjusted < bsp_value:
                                        self.q_correct += 1
                                    else:
                                        self.q_incorrect += 1

                                    order = trade.create_order(
                                        side="LAY", order_type=LimitOrder(price=lay_price_adjusted,
                                                                          size=round(self.stake / (
                                                                                  lay_price_adjusted - 1),
                                                                                     2),
                                                                          persistence_type="LAPSE"
                                                                          )
                                    )

                                    self.lay_bet_tracker[market_id][runner.selection_id] = [order, bsp_value,
                                                                                            market_book.market_id,
                                                                                            runner.selection_id,
                                                                                            runner.handicap,
                                                                                            runner.sp.actual_sp,
                                                                                            lay_price_adjusted]
                                    self.matched_lay_bet_tracker[market_id][runner.selection_id] = False

                                    market.place_order(order)
                                    amount = round(self.stake / (lay_price_adjusted - 1), 2)

                                    self.q_margin += amount * (
                                                bsp_value - lay_price_adjusted) / lay_price_adjusted

                                    # print("MADE A LAY BET")


    def process_orders(self, market: Market, orders: list) -> None:
        # check our backs3
        if len(self.back_bet_tracker.keys()) > 0:
            del_list_back = []
            for market_id in self.back_bet_tracker.keys():
                for selection_id in self.back_bet_tracker[market_id].keys():
                    if len(self.back_bet_tracker[market_id][selection_id]) != 0:
                        order = self.back_bet_tracker[market_id][selection_id][0]
                        if (order.status == OrderStatus.EXECUTION_COMPLETE):
                            if self.matched_back_bet_tracker[market_id][selection_id] == False:
                                self.matched_back_bet_tracker[market_id][selection_id] = True
                                backed_price = self.back_bet_tracker[market_id][selection_id][-1]
                                # because 10 is the minimum liability for the bsp lay
                                # print("back MATCHED IS ", order.size_matched)
                                if order.size_matched >= 10.00:  # because lowest amount
                                    # print("here 1")
                                    bsp_value = self.back_bet_tracker[market_id][selection_id][1]

                                    if backed_price > bsp_value:
                                        self.matched_correct += 1
                                        self.back_matched_correct += 1
                                        self.m_correct_margin += order.size_matched * (backed_price - bsp_value) / backed_price

                                    else:
                                        self.matched_incorrect += 1
                                        self.back_matched_incorrect += 1
                                        self.m_incorrect_margin += order.size_matched * (backed_price - bsp_value) / backed_price

                                    # expected green margin
                                    green_margin_add = order.size_matched * (backed_price - bsp_value) / backed_price

                                    self.green_margin += green_margin_add

                                    # collect for lay at bsp
                                    market_id_ = self.back_bet_tracker[market_id][selection_id][2]
                                    selection_id_ = self.back_bet_tracker[market_id][selection_id][3]
                                    handicap_ = self.back_bet_tracker[market_id][selection_id][4]

                                    # lay at BSP
                                    trade2 = Trade(
                                        market_id=market_id_,
                                        selection_id=selection_id_,
                                        handicap=handicap_,
                                        strategy=self,
                                    )

                                    order2 = trade2.create_order(
                                        side="LAY", order_type=MarketOnCloseOrder(liability=order.size_matched)
                                    )

                                    market.place_order(order2)

                                    # Frees this up to be done again once we have greened.
                                    # !! unhash below and the horse del_list if you want to do more bets.
                                    # del_list_back.append(selection_id)
                                    # self.matched_back_bet_tracker[market_id][selection_id] = False

                                elif order.size_matched != 0:
                                    bsp_value = self.back_bet_tracker[market_id][selection_id][1]
                                    backed_price = self.back_bet_tracker[market_id][selection_id][-1]
                                    self.amount_gambled += order.size_matched
                                    self.matched_back_bet_tracker[market_id][selection_id] = True
                                    if backed_price > bsp_value:
                                        self.matched_correct += 1
                                        self.back_matched_correct += 1
                                        self.m_correct_margin += order.size_matched * (backed_price - bsp_value) / backed_price

                                    else:
                                        self.matched_incorrect += 1
                                        self.back_matched_incorrect += 1
                                        self.m_incorrect_margin += order.size_matched * (backed_price - bsp_value) / backed_price

                        elif (order.status == OrderStatus.EXECUTABLE) & (order.size_matched != 0):
                            bsp_value = self.back_bet_tracker[market_id][selection_id][1]
                            backed_price = self.back_bet_tracker[market_id][selection_id][-1]
                            if self.seconds_to_start < TIME_BEFORE_START:
                                self.amount_gambled += order.size_matched
                                if backed_price > bsp_value:
                                    self.matched_correct += 1
                                    self.back_matched_correct += 1
                                    self.m_correct_margin += order.size_matched * (backed_price - bsp_value) / backed_price

                                else:
                                    self.matched_incorrect += 1
                                    self.back_matched_incorrect += 1
                                    self.m_incorrect_margin += order.size_matched * (backed_price - bsp_value) / backed_price

                                market.cancel_order(order)
                                self.matched_back_bet_tracker[market_id][selection_id] = True

                    # for horse in del_list_back:
                    #     del self.back_bet_tracker[market_id][horse]

        if len(self.lay_bet_tracker.keys()) > 0:
            del_list_lay = []
            for market_id in self.lay_bet_tracker.keys():
                for selection_id in self.lay_bet_tracker[market_id].keys():
                    if len(self.lay_bet_tracker[market_id][selection_id]) != 0:
                        order = self.lay_bet_tracker[market_id][selection_id][0]
                        if (order.status == OrderStatus.EXECUTION_COMPLETE):
                            if self.matched_lay_bet_tracker[market_id][selection_id] == False:
                                self.matched_lay_bet_tracker[market_id][selection_id] = True
                                lay_price_adjusted = self.lay_bet_tracker[market_id][selection_id][-1]
                                layed_size = round(self.stake / (lay_price_adjusted - 1), 2)
                                # print("in the lay order.size_matched is ", order.size_matched)
                                # print("supposed layed size ", layed_size)
                                if order.size_matched > 1:  # because min liability on back is 1
                                    # print("here 2")
                                    bsp_value = self.lay_bet_tracker[market_id][selection_id][1]
                                    if lay_price_adjusted < bsp_value:
                                        self.matched_correct += 1
                                        self.lay_matched_correct += 1
                                        self.m_correct_margin += order.size_matched * (bsp_value - lay_price_adjusted) / lay_price_adjusted

                                    else:
                                        self.matched_incorrect += 1
                                        self.lay_matched_incorrect += 1
                                        self.m_incorrect_margin += order.size_matched * (bsp_value - lay_price_adjusted) / lay_price_adjusted

                                        # expected green margin
                                    green_margin_add = order.size_matched * (bsp_value - lay_price_adjusted) / lay_price_adjusted

                                    self.green_margin += green_margin_add

                                    # collect for back at bsp
                                    market_id_ = self.lay_bet_tracker[market_id][selection_id][2]
                                    selection_id_ = self.lay_bet_tracker[market_id][selection_id][3]
                                    handicap_ = self.lay_bet_tracker[market_id][selection_id][4]

                                    # back at BSP
                                    trade3 = Trade(
                                        market_id=market_id_,
                                        selection_id=selection_id_,
                                        handicap=handicap_,
                                        strategy=self,
                                    )

                                    order3 = trade3.create_order(
                                        side="BACK", order_type=MarketOnCloseOrder(liability=order.size_matched)
                                    )

                                    market.place_order(order3)


                                    # Frees this up to be done again

                                    # del_list_lay.append(selection_id)
                                    # self.matched_lay_bet_tracker[market_id][selection_id] = False

                                elif (order.size_matched != 0):  # not remove all of this eventually this is just a trial
                                    bsp_value = self.lay_bet_tracker[market_id][selection_id][1]
                                    lay_price = self.lay_bet_tracker[market_id][selection_id][-1]
                                    self.amount_gambled += order.size_matched * (lay_price - 1)
                                    self.matched_lay_bet_tracker[market_id][selection_id] = True
                                    if lay_price < bsp_value:
                                        self.matched_correct += 1
                                        self.lay_matched_correct += 1
                                        self.m_correct_margin += order.size_matched * (bsp_value - lay_price_adjusted) / lay_price_adjusted

                                    else:
                                        self.matched_incorrect += 1
                                        self.lay_matched_incorrect += 1
                                        self.m_incorrect_margin += order.size_matched * (bsp_value - lay_price_adjusted) / lay_price_adjusted


                        elif (order.status == OrderStatus.EXECUTABLE) & (order.size_matched != 0):
                            bsp_value = self.lay_bet_tracker[market_id][selection_id][1]
                            lay_price = self.lay_bet_tracker[market_id][selection_id][-1]
                            if self.seconds_to_start < TIME_BEFORE_START:
                                self.amount_gambled += order.size_matched * (lay_price - 1)
                                if lay_price < bsp_value:
                                    self.matched_correct += 1
                                    self.lay_matched_correct += 1
                                    self.m_correct_margin += order.size_matched * (bsp_value - lay_price_adjusted) / lay_price_adjusted

                                else:
                                    self.matched_incorrect += 1
                                    self.lay_matched_incorrect += 1
                                    self.m_incorrect_margin += order.size_matched * (bsp_value - lay_price_adjusted) / lay_price_adjusted

                                market.cancel_order(order)
                                self.matched_lay_bet_tracker[market_id][selection_id] = True

            # for horse in del_list_lay:
            #     del self.lay_bet_tracker[market_id][horse]

def run(market_file, framework):
    # NOTE change strategy in here to whatever you want above.
    strategy = Strategy1(market_filter={"markets": [market_file]}, max_trade_count=100000
                         , max_live_trade_count=100000, max_order_exposure=10000,
                         max_selection_exposure=100000)
    strategy.regression = False
    framework.add_strategy(strategy)
    framework.run()

    # COMMENT OUT IF NOT USING REGRESSION!


    profit_a = 0
    for market in framework.markets:
        print("Profit: {0:.2f}".format(sum([o.simulated.profit for o in market.blotter])))
        profit_a += sum([o.simulated.profit for o in market.blotter])
        for order in market.blotter:
            #             print("_-_-_-_-_-_")
            #             print(order.selection_id)
            #             print(order.order_type)
            #             print(order.size_matched)
            #             print(order.simulated.profit)
            #             write_list = [
            #                 order.selection_id,
            #                 order.responses.date_time_placed,
            #                 order.status,
            #                 order.order_type.price,  # these three don't make sense with bsp orders
            #                 order.average_price_matched,
            #                 order.size_matched,
            #                 order.simulated.profit,
            #             ]
            pass

    return profit_a, strategy.q_correct, strategy.q_incorrect, strategy.matched_correct, strategy.matched_incorrect, strategy.m_correct_margin, strategy.m_incorrect_margin, strategy.green_margin, strategy.amount_gambled, strategy.lay_matched_correct, strategy.lay_matched_incorrect, strategy.back_matched_correct, strategy.back_matched_incorrect, strategy.q_margin


def piped_run(test_folder, bsp_df):
    number_files = len(os.listdir(test_folder))
    j = 1

    total_profit = 0
    total_matched_correct = 0
    total_matched_incorrect = 0
    total_back_correct = 0
    total_back_incorrect = 0
    total_lay_correct = 0
    total_lay_incorrect = 0
    total_m_c_marg = 0
    total_m_i_marg = 0
    total_green_margin = 0
    total_amount_gambled = 0
    actual_profit_plotter = []
    expected_profit_plotter = []
    green_plotter = []
    race_counter = 0
    total_q_correct = 0
    total_q_incorrect = 0
    total_m_correct = 0
    total_m_incorrect = 0
    total_q_margin = 0

    for file in os.listdir(test_folder):
        print("{}/{}".format(j, number_files))
        j += 1
        print(file)
        if float(file) in bsp_df["EVENT_ID"].values:

            if file[-5] == "d":
                pass
            else:
                race_counter += 1
                file_path = os.path.join(test_folder, file)
                framework = FlumineSimulation(client=client)

                profit_1, q_correct, q_incorrect, matched_correct, matched_incorrect, m_c_marg, m_i_marg, \
                green_margin, amount_gambled, lay_matched_correct, lay_matched_incorrect, back_matched_correct, \
                back_matched_incorrect, q_margin = run(file_path, framework)

                total_profit += profit_1
                print("total_profit = ", total_profit)
                print("---")
                print("q correct = ", q_correct)
                total_q_correct += q_correct
                print("total_q_correct = ", total_q_correct)
                print("q incorrect = ", q_incorrect)
                total_q_incorrect += q_incorrect
                print("total_q_incorrect = ", total_q_incorrect)
                print("matched_correct = ", matched_correct)
                total_matched_correct += matched_correct
                print("total matched correct = ", total_matched_correct)
                print("matched_incorrect = ", matched_incorrect)
                total_matched_incorrect += matched_incorrect
                print("total matched incorrect = ", total_matched_incorrect)
                print("---")
                print("green margin = ", green_margin)
                print("q_margin = ", q_margin)
                total_back_correct += back_matched_correct
                total_back_incorrect += back_matched_incorrect
                total_lay_correct += lay_matched_correct
                total_lay_incorrect += lay_matched_incorrect
                print("---")
                print("back matched correct = ", back_matched_correct)
                print("back matched incorrect = ", back_matched_incorrect)
                print("lay matched correct = ", lay_matched_correct)
                print("lay matched incorrect = ", lay_matched_incorrect)
                print("---")
                total_m_c_marg += m_c_marg
                total_m_i_marg += m_i_marg
                total_green_margin += green_margin
                total_amount_gambled += amount_gambled
                total_q_margin += q_margin
                print("total c margin ", total_m_c_marg)
                print("total_i_margin ", total_m_i_marg)
                print("total_green_margin ", total_green_margin)
                print("total_q_margin ", total_q_margin)
                print("---")
                print("total amount gambled ", total_amount_gambled)
                actual_profit_plotter.append(total_profit)
                expected_profit_plotter.append(total_m_c_marg + total_m_i_marg)
                green_plotter.append(total_green_margin)
                print("")
                if race_counter % 10 == 0:
                    # plt.plot(range(race_counter), actual_profit_plotter, label="backtest", color="b")
                    plt.plot(range(race_counter), expected_profit_plotter, label="expected", color="y")
                    plt.plot(range(race_counter), green_plotter, label="greened_profit", color="g")
                    plt.axhline(y=0.5, color='r', linestyle='-')
                    plt.xlabel("Number of Races")
                    plt.ylabel("Profit")

                    plt.legend()
                    plt.draw()
                print("")
                # when the profit is printed it will be for each market only

    print(total_matched_correct)
    print(total_matched_incorrect)
    print("total correct margin is : ", total_m_c_marg)
    print("total incorrect margin is : ", total_m_i_marg)

    return race_counter, actual_profit_plotter, expected_profit_plotter, green_plotter



class BacktestLoggingControl(LoggingControl):
    NAME = "BACKTEST_LOGGING_CONTROL"

    def __init__(self, *args, **kwargs):
        super(BacktestLoggingControl, self).__init__(*args, **kwargs)
        self._setup()

    def _setup(self):
        with open("orders.txt", "w") as m:
            csv_writer = csv.DictWriter(m, delimiter=",", fieldnames=FIELDNAMES)
            csv_writer.writeheader()

    def _process_cleared_orders_meta(self, event):
        orders = event.event
        with open("orders.txt", "a") as m:
            for order in orders:
                if order.order_type.ORDER_TYPE == OrderTypes.LIMIT:
                    size = order.order_type.size
                else:
                    size = order.order_type.liability
                if order.order_type.ORDER_TYPE == OrderTypes.MARKET_ON_CLOSE:
                    price = None
                else:
                    price = order.order_type.price
                try:
                    order_data = {
                        "bet_id": order.bet_id,
                        "strategy_name": order.trade.strategy,
                        "market_id": order.market_id,
                        "selection_id": order.selection_id,
                        "trade_id": order.trade.id,
                        "date_time_placed": order.responses.date_time_placed,
                        "price": price,
                        "price_matched": order.average_price_matched,
                        "size": size,
                        "size_matched": order.size_matched,
                        "profit": order.simulated.profit,
                        "side": order.side,
                        "elapsed_seconds_executable": order.elapsed_seconds_executable,
                        "order_status": order.status.value,
                        "market_note": order.trade.market_notes,
                        "trade_notes": order.trade.notes_str,
                        "order_notes": order.notes_str,
                    }
                    csv_writer = csv.DictWriter(m, delimiter=",", fieldnames=FIELDNAMES)
                    csv_writer.writerow(order_data)
                except Exception as e:
                    logger.error(
                        "_process_cleared_orders_meta: %s" % e,
                        extra={"order": order, "error": e},
                    )

        logger.info("Orders updated", extra={"order_count": len(orders)})

    def _process_cleared_markets(self, event):
        cleared_markets = event.event
        for cleared_market in cleared_markets.orders:
            logger.info(
                "Cleared market",
                extra={
                    "market_id": cleared_market.market_id,
                    "bet_count": cleared_market.bet_count,
                    "profit": cleared_market.profit,
                    "commission": cleared_market.commission,
                },
            )


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    FIELDNAMES = [
        "bet_id",
        "strategy_name",
        "market_id",
        "selection_id",
        "trade_id",
        "date_time_placed",
        "price",
        "price_matched",
        "size",
        "size_matched",
        "profit",
        "side",
        "elapsed_seconds_executable",
        "order_status",
        "market_note",
        "trade_notes",
        "order_notes",
    ]

    TIME_BEFORE_START = 10  # Hyperparam for sorting out the end

    client = clients.SimulatedClient()
    framework = FlumineSimulation(client=client)

    custom_format = "%(asctime) %(levelname) %(message)"
    log_handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(custom_format)
    formatter.converter = time.gmtime
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    logger.setLevel(logging.CRITICAL)  # Set to logging.CRITICAL to speed up backtest

    ticks_path = "/home/x4nno/Documents/BetFair/Bristol_project/ticks.csv"
    ticks_df = pd.read_csv(ticks_path)

    #  pick classifier or regressor !!! MAKE SURE YOU CHANGE THE TAG IN "RUN"

    model = MLPClassifier(alpha=5, max_iter=1000)

    # JULY
    month_folder = "/media/x4nno/TOSHIBA EXT/Wanker Tom/Betfair_data/BetFair_data_new_aug_22/historic_data/Jan"
    test_folder = "/media/x4nno/TOSHIBA EXT/Wanker Tom/Betfair_data/BetFair_data_new_aug_22/horses_jan_wins"
    bsp_folder = "/media/x4nno/TOSHIBA EXT/Wanker Tom/Betfair_data/BetFair_data_new_aug_22/Jan_bsps"
    bsp_df = pd.read_csv(bsp_folder + "/bsp_df.csv")

    train_df1 = pd.read_csv("jan20_analysis_direct_nr0_100_50_many_wom.csv")
    train_df2 = pd.read_csv("feb20_analysis_direct_nr0_100_50_many_wom.csv")
    train_df3 = pd.read_csv("mar20_analysis_direct_nr0_100_50_many_wom.csv")
    train_df4 = pd.read_csv("may22_analysis_direct_nr0_100_50_many_wom.csv")
    train_df5 = pd.read_csv("jun22_analysis_direct_nr0_100_50_many_wom.csv")
    train_df6 = pd.read_csv("jul22_analysis_direct_nr0_100_50_many_wom.csv")

    frames = [
        train_df2,
        train_df3,
        train_df4,
        train_df5,
        train_df6]
    train_df = pd.concat(frames)

    test_analysis_df = train_df1

    # remove regression tag if using classifier
    model, test_analysis_df, scaler, clm, test_analysis_df_y = fit_horse_model(train_df, test_analysis_df, model, regression=False)

    race_counter, actual_profit_plotter, expected_profit_plotter, green_plotter = piped_run(test_folder, bsp_df)

    # race_counter = len(os.listdir(test_folder))
    # plt.plot(range(race_counter), actual_profit_plotter, label="backtest", color = "b")
    plt.plot(range(race_counter), expected_profit_plotter, label="expected", color="y")
    plt.plot(range(race_counter), green_plotter, label="greened_profit", color="g")
    plt.axhline(y=0.5, color='r', linestyle='-')
    plt.xlabel("Number of Races")
    plt.ylabel("Profit")

    plt.legend()
    plt.show()
    print("Total expected profit is ", expected_profit_plotter[-1])
