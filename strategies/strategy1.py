from typing import Union
import pandas as pd

from flumine import BaseStrategy
from flumine.order.trade import Trade
from flumine.order.order import LimitOrder, MarketOnCloseOrder, OrderStatus
from flumine.markets.market import Market
from betfairlightweight.resources import MarketBook

from onedrive import Onedrive
from utils.constants import TIME_BEFORE_START

from tom.toms_utils import normalized_transform
from utils.utils import train_model
from utils.config import app_principal, SITE_URL


class Strategy1(BaseStrategy):
    def __init__(self, onedrive: Onedrive, *args, **kwargs):
        self.onedrive = onedrive
        super_kwargs = kwargs.copy()
        super_kwargs.pop("onedrive", None)
        super().__init__(*args, **super_kwargs)

    # back and lay in here

    def start(self) -> None:
        self.regression = True
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
        self.stake = 50  # @WHAT IS THIS
        self.first_nonrunners = True
        self.runner_number = None
        self.lay_matched_correct = 0
        self.back_matched_correct = 0
        self.lay_matched_incorrect = 0
        self.back_matched_incorrect = 0
        self.q_margin = 0
        self.metrics = {
            "profit": 0,
            "q_correct": self.q_correct,
            "q_incorrect": self.q_incorrect,
            "matched_correct": self.matched_correct,
            "matched_incorrect": self.matched_incorrect,
            "m_correct_margin": self.m_correct_margin,
            "m_incorrect_margin": self.m_incorrect_margin,
            "green_margin": self.green_margin,
            "amount_gambled": self.amount_gambled,
            "lay_matched_correct": self.lay_matched_correct,
            "lay_matched_incorrect": self.lay_matched_incorrect,
            "back_matched_correct": self.back_matched_correct,
            "back_matched_incorrect": self.back_matched_incorrect,
            "q_margin": self.q_margin,
        }

    def set_market_filter(self, market_filter: Union[dict, list]) -> None:
        self.market_filter = market_filter

    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:

        # process_market_book only executed if this returns True
        self.process_fundamentals(market_book)
        if self.first_nonrunners:
            self.runner_number = market_book.number_of_active_runners
            self.first_nonrunners = False
        else:
            if market_book.number_of_active_runners != self.runner_number:
                return False  # this will stop any more action happening in this market.

        if (market_book.status == "CLOSED") or (
            self.seconds_to_start < TIME_BEFORE_START
        ):
            return False
        else:
            return True

    def process_fundamentals(self, market_book: MarketBook):
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
            if (self.seconds_to_start < 120) & (self.seconds_to_start > 100):
                for runner in market_book.runners:
                    # backs
                    if market_id not in self.back_bet_tracker.keys():
                        self.back_bet_tracker[market_id] = {}
                        self.matched_back_bet_tracker[market_id] = {}
                    if (
                        runner.selection_id
                        not in self.back_bet_tracker[market_id].keys()
                    ):
                        self.back_bet_tracker[market_id][runner.selection_id] = {}
                        self.matched_back_bet_tracker[market_id][
                            runner.selection_id
                        ] = {}
                        print("before checking selection_id")
                        if runner.status == "ACTIVE":
                            print("is active")
                            try:
                                onedrive = Onedrive(
                                    client_id=app_principal["client_id"],
                                    client_secret=app_principal["client_secret"],
                                    site_url=SITE_URL,
                                )

                                ticks_df = onedrive.get_folder_contents(
                                    target_folder="ticks", target_file="ticks.csv"
                                )
                                model, clm, scaler = train_model(
                                    ticks_df,
                                    onedrive,
                                    model="BayesianRidge",
                                )

                                print("Model successfully loaded.")

                                test_analysis_df = onedrive.get_test_df()
                                test_analysis_df = test_analysis_df.dropna()
                                test_analysis_df = test_analysis_df[
                                    (test_analysis_df["mean_120"] <= 50)
                                    & (test_analysis_df["mean_120"] > 1.1)
                                ]
                                test_analysis_df = test_analysis_df[
                                    test_analysis_df["mean_14400"] > 0
                                ]
                                # below is a slight hack ...
                                test_analysis_df = test_analysis_df.drop(
                                    test_analysis_df[
                                        test_analysis_df["std_2700"] > 1
                                    ].index
                                )

                                test_analysis_df_y = pd.DataFrame().assign(
                                    market_id=test_analysis_df["market_id"],
                                    selection_ids=test_analysis_df["selection_ids"],
                                    bsps=test_analysis_df["bsps"],
                                )

                                predict_row = test_analysis_df.loc[
                                    (
                                        test_analysis_df["selection_ids"]
                                        == runner.selection_id
                                    )
                                    & (test_analysis_df["market_id"] == market_id)
                                ]

                                mean_120 = predict_row["mean_120"].values[0]

                                predict_row = normalized_transform(
                                    predict_row, ticks_df
                                )
                                predict_row = predict_row.drop(
                                    ["bsps_temp", "bsps"], axis=1
                                )
                                if not self.regression:
                                    predict_row = predict_row.drop(
                                        ["mean_120_temp"], axis=1
                                    )
                                predict_row = pd.DataFrame(
                                    scaler.transform(predict_row), columns=clm
                                )

                                # this is for regression
                                if self.regression:
                                    runner_predicted_bsp = model.predict(predict_row)

                                # below is for classification
                                else:
                                    runner_predicted_bsp_pos = model.predict_proba(
                                        predict_row
                                    )[0][1]
                                    print("pos_prob", runner_predicted_bsp_pos)
                            except Exception as e:
                                print(f"Excpetion: {e}")

                                if self.regression:
                                    runner_predicted_bsp = False
                                else:
                                    runner_predicted_bsp_pos = False
                                mean_120 = False

                            if mean_120 != False:
                                # in the back_price / number put in where you want to base from: prevoisly runner.last_price_traded
                                # get_price(runner.ex.available_to_back, 1)
                                back_price = ticks_df.iloc[
                                    ticks_df["tick"].sub(mean_120).abs().idxmin()
                                ]["tick"]
                                back_number = ticks_df.iloc[
                                    ticks_df["tick"].sub(mean_120).abs().idxmin()
                                ]["number"]
                                back_number_adjust = back_number
                                confidence_number = back_number - 4
                                confidence_price = ticks_df.iloc[
                                    ticks_df["number"]
                                    .sub(confidence_number)
                                    .abs()
                                    .idxmin()
                                ]["tick"]
                                back_price_adjusted = ticks_df.iloc[
                                    ticks_df["number"]
                                    .sub(back_number_adjust)
                                    .abs()
                                    .idxmin()
                                ]["tick"]
                                bsp_row = test_analysis_df_y.loc[
                                    (
                                        test_analysis_df_y["selection_ids"]
                                        == runner.selection_id
                                    )
                                    & (test_analysis_df_y["market_id"] == market_id)
                                ]
                                bsp_value = bsp_row["bsps"].values[0]

                                if self.regression:
                                    if (
                                        (runner_predicted_bsp < confidence_price)
                                        & (mean_120 <= 50)
                                        & (mean_120 > 1.1)
                                    ):

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
                                            side="BACK",
                                            order_type=LimitOrder(
                                                price=back_price_adjusted,
                                                size=self.stake,
                                                persistence_type="LAPSE",
                                            ),
                                        )

                                        print(
                                            f"Back bet order created: \\ back price adjusted{back_price_adjusted}\\ order size: {self.stake}"
                                        )
                                        self.back_bet_tracker[market_id][
                                            runner.selection_id
                                        ] = [
                                            order,
                                            bsp_value,
                                            market_book.market_id,
                                            runner.selection_id,
                                            runner.handicap,
                                            runner.sp.actual_sp,
                                            back_price_adjusted,
                                        ]
                                        self.matched_back_bet_tracker[market_id][
                                            runner.selection_id
                                        ] = False

                                        market.place_order(order)

                                        self.q_margin += (
                                            self.stake
                                            * (back_price_adjusted - bsp_value)
                                            / back_price_adjusted
                                        )

                                else:
                                    if (
                                        (runner_predicted_bsp_pos > 0.6)
                                        & (mean_120 <= 50)
                                        & (mean_120 > 1.1)
                                    ):

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
                                            side="BACK",
                                            order_type=LimitOrder(
                                                price=back_price_adjusted,
                                                size=self.stake,
                                                persistence_type="LAPSE",
                                            ),
                                        )

                                        print(
                                            f"Back bet order created: \\ back price adjusted{back_price_adjusted}\\ order size: {self.stake}"
                                        )
                                        self.back_bet_tracker[market_id][
                                            runner.selection_id
                                        ] = [
                                            order,
                                            bsp_value,
                                            market_book.market_id,
                                            runner.selection_id,
                                            runner.handicap,
                                            runner.sp.actual_sp,
                                            back_price_adjusted,
                                        ]
                                        self.matched_back_bet_tracker[market_id][
                                            runner.selection_id
                                        ] = False

                                        market.place_order(order)

                                        self.q_margin += (
                                            self.stake
                                            * (back_price_adjusted - bsp_value)
                                            / back_price_adjusted
                                        )

                                # print("que a back on ", runner.selection_id, " at price ", u_v, "at bsp ", bsp_value)

                        # below here we will sort out the lays

                        ""

                    if market_id not in self.lay_bet_tracker.keys():
                        self.lay_bet_tracker[market_id] = {}
                        self.matched_lay_bet_tracker[market_id] = {}
                    if (
                        runner.selection_id
                        not in self.lay_bet_tracker[market_id].keys()
                    ):
                        self.lay_bet_tracker[market_id][runner.selection_id] = {}
                        self.matched_lay_bet_tracker[market_id][
                            runner.selection_id
                        ] = {}
                        if runner.status == "ACTIVE":
                            try:
                                predict_row = test_analysis_df.loc[
                                    (
                                        test_analysis_df["selection_ids"]
                                        == runner.selection_id
                                    )
                                    & (test_analysis_df["market_id"] == market_id)
                                ]
                                mean_120 = predict_row["mean_120"].values[0]

                                predict_row = normalized_transform(
                                    predict_row, ticks_df
                                )
                                # note when we do the live stream there will be no bsps
                                predict_row = predict_row.drop(
                                    ["bsps_temp", "bsps"], axis=1
                                )
                                if not self.regression:
                                    predict_row = predict_row.drop(
                                        ["mean_120_temp"], axis=1
                                    )

                                predict_row = pd.DataFrame(
                                    scaler.transform(predict_row), columns=clm
                                )

                                if self.regression:
                                    runner_predicted_bsp = model.predict(predict_row)
                                else:
                                    runner_predicted_bsp_neg = model.predict_proba(
                                        predict_row
                                    )[0][0]
                                    print("neg_prob", runner_predicted_bsp_neg)
                            except:
                                if self.regression:
                                    runner_predicted_bsp = False
                                else:
                                    runner_predicted_bsp_neg = False
                                mean_120 = False

                            if mean_120 != False:
                                # in the lay_price / number put in where you want to base from: prevoisly runner.last_price_traded
                                # get_price(runner.ex.available_to_lay, 1)
                                lay_price = ticks_df.iloc[
                                    ticks_df["tick"].sub(mean_120).abs().idxmin()
                                ]["tick"]
                                lay_number = ticks_df.iloc[
                                    ticks_df["tick"].sub(mean_120).abs().idxmin()
                                ]["number"]
                                lay_number_adjust = lay_number
                                confidence_number = lay_number + 4
                                confidence_price = ticks_df.iloc[
                                    ticks_df["number"]
                                    .sub(confidence_number)
                                    .abs()
                                    .idxmin()
                                ]["tick"]
                                lay_price_adjusted = ticks_df.iloc[
                                    ticks_df["number"]
                                    .sub(lay_number_adjust)
                                    .abs()
                                    .idxmin()
                                ]["tick"]
                                bsp_row = test_analysis_df_y.loc[
                                    (
                                        test_analysis_df_y["selection_ids"]
                                        == runner.selection_id
                                    )
                                    & (test_analysis_df_y["market_id"] == market_id)
                                ]
                                bsp_value = bsp_row["bsps"].values[0]

                                if self.regression:
                                    if (
                                        (runner_predicted_bsp > confidence_price)
                                        and (lay_price_adjusted <= self.stake)
                                        and (lay_price_adjusted > 1.1)
                                    ):
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

                                        amount = round(
                                            self.stake / (lay_price_adjusted - 1), 2
                                        )
                                        order = trade.create_order(
                                            side="LAY",
                                            order_type=LimitOrder(
                                                price=lay_price_adjusted,
                                                size=amount,
                                                persistence_type="LAPSE",
                                            ),
                                        )

                                        self.lay_bet_tracker[market_id][
                                            runner.selection_id
                                        ] = [
                                            order,
                                            bsp_value,
                                            market_book.market_id,
                                            runner.selection_id,
                                            runner.handicap,
                                            runner.sp.actual_sp,
                                            lay_price_adjusted,
                                        ]
                                        self.matched_lay_bet_tracker[market_id][
                                            runner.selection_id
                                        ] = False

                                        market.place_order(order)

                                        self.q_margin += (
                                            amount
                                            * (bsp_value - lay_price_adjusted)
                                            / lay_price_adjusted
                                        )

                                        print(
                                            f"Lay bet order created: \\ lay_price_adjusted{lay_price_adjusted}\\ order size: {amount}"
                                        )
                                # not regression (classifier)
                                else:
                                    if (
                                        (runner_predicted_bsp_neg > 0.6)
                                        and (lay_price_adjusted <= self.stake)
                                        and (lay_price_adjusted > 1.1)
                                    ):
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

                                        amount = round(
                                            self.stake / (lay_price_adjusted - 1), 2
                                        )
                                        order = trade.create_order(
                                            side="LAY",
                                            order_type=LimitOrder(
                                                price=lay_price_adjusted,
                                                size=amount,
                                                persistence_type="LAPSE",
                                            ),
                                        )

                                        self.lay_bet_tracker[market_id][
                                            runner.selection_id
                                        ] = [
                                            order,
                                            bsp_value,
                                            market_book.market_id,
                                            runner.selection_id,
                                            runner.handicap,
                                            runner.sp.actual_sp,
                                            lay_price_adjusted,
                                        ]
                                        self.matched_lay_bet_tracker[market_id][
                                            runner.selection_id
                                        ] = False

                                        market.place_order(order)

                                        self.q_margin += (
                                            amount
                                            * (bsp_value - lay_price_adjusted)
                                            / lay_price_adjusted
                                        )

                                        print(
                                            f"Lay bet order created: \\ lay_price_adjusted{lay_price_adjusted}\\ order size: {amount}"
                                        )

    def process_orders(self, market: Market, orders: list) -> None:
        # check our backs3
        if len(self.back_bet_tracker.keys()) > 0:
            del_list_back = []
            for market_id in self.back_bet_tracker.keys():
                for selection_id in self.back_bet_tracker[market_id].keys():
                    if len(self.back_bet_tracker[market_id][selection_id]) != 0:
                        order = self.back_bet_tracker[market_id][selection_id][0]
                        if order.status == OrderStatus.EXECUTION_COMPLETE:
                            if (
                                self.matched_back_bet_tracker[market_id][selection_id]
                                == False
                            ):
                                self.matched_back_bet_tracker[market_id][
                                    selection_id
                                ] = True
                                backed_price = self.back_bet_tracker[market_id][
                                    selection_id
                                ][-1]
                                # because 10 is the minimum liability for the bsp lay
                                print("back MATCHED IS ", order.size_matched)
                                if order.size_matched >= 10.00:  # because lowest amount
                                    print("here 1")
                                    bsp_value = self.back_bet_tracker[market_id][
                                        selection_id
                                    ][1]

                                    if backed_price > bsp_value:
                                        self.matched_correct += 1
                                        self.back_matched_correct += 1
                                        self.m_correct_margin += (
                                            order.size_matched
                                            * (backed_price - bsp_value)
                                            / backed_price
                                        )

                                    else:
                                        self.matched_incorrect += 1
                                        self.back_matched_incorrect += 1
                                        self.m_incorrect_margin += (
                                            order.size_matched
                                            * (backed_price - bsp_value)
                                            / backed_price
                                        )

                                    # expected green margin
                                    green_margin_add = (
                                        order.size_matched
                                        * (backed_price - bsp_value)
                                        / backed_price
                                    )

                                    self.green_margin += green_margin_add

                                    # collect for lay at bsp
                                    market_id_ = self.back_bet_tracker[market_id][
                                        selection_id
                                    ][2]
                                    selection_id_ = self.back_bet_tracker[market_id][
                                        selection_id
                                    ][3]
                                    handicap_ = self.back_bet_tracker[market_id][
                                        selection_id
                                    ][4]

                                    # lay at BSP
                                    trade2 = Trade(
                                        market_id=market_id_,
                                        selection_id=selection_id_,
                                        handicap=handicap_,
                                        strategy=self,
                                    )

                                    order2 = trade2.create_order(
                                        side="LAY",
                                        order_type=MarketOnCloseOrder(
                                            liability=order.size_matched
                                        ),
                                    )

                                    market.place_order(order2)

                                # Frees this up to be done again once we have greened.
                                # !! unhash below and the horse del_list if you want to do more bets.
                                #  del_list_back.append(selection_id)
                                #  self.matched_back_bet_tracker[market_id][selection_id] = False

                                elif order.size_matched != 0:
                                    bsp_value = self.back_bet_tracker[market_id][
                                        selection_id
                                    ][1]
                                    backed_price = self.back_bet_tracker[market_id][
                                        selection_id
                                    ][-1]
                                    self.amount_gambled += order.size_matched
                                    self.matched_back_bet_tracker[market_id][
                                        selection_id
                                    ] = True
                                    if backed_price > bsp_value:
                                        self.matched_correct += 1
                                        self.back_matched_correct += 1
                                        self.m_correct_margin += (
                                            order.size_matched
                                            * (backed_price - bsp_value)
                                            / backed_price
                                        )

                                    else:
                                        self.matched_incorrect += 1
                                        self.back_matched_incorrect += 1
                                        self.m_incorrect_margin += (
                                            order.size_matched
                                            * (backed_price - bsp_value)
                                            / backed_price
                                        )

                        elif (order.status == OrderStatus.EXECUTABLE) & (
                            order.size_matched != 0
                        ):
                            bsp_value = self.back_bet_tracker[market_id][selection_id][
                                1
                            ]
                            backed_price = self.back_bet_tracker[market_id][
                                selection_id
                            ][-1]
                            if self.seconds_to_start < TIME_BEFORE_START:
                                self.amount_gambled += order.size_matched
                                if backed_price > bsp_value:
                                    self.matched_correct += 1
                                    self.back_matched_correct += 1
                                    self.m_correct_margin += (
                                        order.size_matched
                                        * (backed_price - bsp_value)
                                        / backed_price
                                    )

                                else:
                                    self.matched_incorrect += 1
                                    self.back_matched_incorrect += 1
                                    self.m_incorrect_margin += (
                                        order.size_matched
                                        * (backed_price - bsp_value)
                                        / backed_price
                                    )

                                market.cancel_order(order)
                                self.matched_back_bet_tracker[market_id][
                                    selection_id
                                ] = True

        # for horse in del_list_back:
        # del self.back_bet_tracker[market_id][horse]

        if len(self.lay_bet_tracker.keys()) > 0:
            del_list_lay = []
            for market_id in self.lay_bet_tracker.keys():
                for selection_id in self.lay_bet_tracker[market_id].keys():
                    if len(self.lay_bet_tracker[market_id][selection_id]) != 0:
                        order = self.lay_bet_tracker[market_id][selection_id][0]
                        if order.status == OrderStatus.EXECUTION_COMPLETE:
                            if (
                                self.matched_lay_bet_tracker[market_id][selection_id]
                                == False
                            ):
                                self.matched_lay_bet_tracker[market_id][
                                    selection_id
                                ] = True
                                lay_price_adjusted = self.lay_bet_tracker[market_id][
                                    selection_id
                                ][-1]
                                layed_size = round(
                                    self.stake / (lay_price_adjusted - 1), 2
                                )
                                print(
                                    "in the lay order.size_matched is ",
                                    order.size_matched,
                                )
                                print("supposed layed size ", layed_size)
                                if (
                                    order.size_matched > 1
                                ):  # because min liability on back is 1
                                    print("here 2")
                                    bsp_value = self.lay_bet_tracker[market_id][
                                        selection_id
                                    ][1]
                                    if lay_price_adjusted < bsp_value:
                                        self.matched_correct += 1
                                        self.lay_matched_correct += 1
                                        self.m_correct_margin += (
                                            order.size_matched
                                            * (bsp_value - lay_price_adjusted)
                                            / lay_price_adjusted
                                        )

                                    else:
                                        self.matched_incorrect += 1
                                        self.lay_matched_incorrect += 1
                                        self.m_incorrect_margin += (
                                            order.size_matched
                                            * (bsp_value - lay_price_adjusted)
                                            / lay_price_adjusted
                                        )

                                        # expected green margin
                                    green_margin_add = (
                                        order.size_matched
                                        * (bsp_value - lay_price_adjusted)
                                        / lay_price_adjusted
                                    )

                                    self.green_margin += green_margin_add

                                    # collect for back at bsp
                                    market_id_ = self.lay_bet_tracker[market_id][
                                        selection_id
                                    ][2]
                                    selection_id_ = self.lay_bet_tracker[market_id][
                                        selection_id
                                    ][3]
                                    handicap_ = self.lay_bet_tracker[market_id][
                                        selection_id
                                    ][4]

                                    # back at BSP
                                    trade3 = Trade(
                                        market_id=market_id_,
                                        selection_id=selection_id_,
                                        handicap=handicap_,
                                        strategy=self,
                                    )

                                    order3 = trade3.create_order(
                                        side="BACK",
                                        order_type=MarketOnCloseOrder(
                                            liability=order.size_matched
                                        ),
                                    )

                                    market.place_order(order3)

                                # Frees this up to be done again

                                # del_list_lay.append(selection_id)
                                # self.matched_lay_bet_tracker[market_id][selection_id] = False

                                elif (
                                    order.size_matched != 0
                                ):  # not remove all of this eventually this is just a trial
                                    bsp_value = self.lay_bet_tracker[market_id][
                                        selection_id
                                    ][1]
                                    lay_price = self.lay_bet_tracker[market_id][
                                        selection_id
                                    ][-1]
                                    self.amount_gambled += order.size_matched * (
                                        lay_price - 1
                                    )
                                    self.matched_lay_bet_tracker[market_id][
                                        selection_id
                                    ] = True
                                    if lay_price < bsp_value:
                                        self.matched_correct += 1
                                        self.lay_matched_correct += 1
                                        self.m_correct_margin += (
                                            order.size_matched
                                            * (bsp_value - lay_price_adjusted)
                                            / lay_price_adjusted
                                        )

                                    else:
                                        self.matched_incorrect += 1
                                        self.lay_matched_incorrect += 1
                                        self.m_incorrect_margin += (
                                            order.size_matched
                                            * (bsp_value - lay_price_adjusted)
                                            / lay_price_adjusted
                                        )

                        elif (order.status == OrderStatus.EXECUTABLE) & (
                            order.size_matched != 0
                        ):
                            bsp_value = self.lay_bet_tracker[market_id][selection_id][1]
                            lay_price = self.lay_bet_tracker[market_id][selection_id][
                                -1
                            ]
                            if self.seconds_to_start < TIME_BEFORE_START:
                                self.amount_gambled += order.size_matched * (
                                    lay_price - 1
                                )
                                if lay_price < bsp_value:
                                    self.matched_correct += 1
                                    self.lay_matched_correct += 1
                                    self.m_correct_margin += (
                                        order.size_matched
                                        * (bsp_value - lay_price_adjusted)
                                        / lay_price_adjusted
                                    )

                                else:
                                    self.matched_incorrect += 1
                                    self.lay_matched_incorrect += 1
                                    self.m_incorrect_margin += (
                                        order.size_matched
                                        * (bsp_value - lay_price_adjusted)
                                        / lay_price_adjusted
                                    )

                                market.cancel_order(order)
                                self.matched_lay_bet_tracker[market_id][
                                    selection_id
                                ] = True
