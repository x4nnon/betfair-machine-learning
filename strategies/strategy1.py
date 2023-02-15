from typing import Tuple, Union
import numpy as np
import pandas as pd

from flumine import BaseStrategy
from flumine.order.trade import Trade
from flumine.order.order import LimitOrder, MarketOnCloseOrder, OrderStatus
from flumine.markets.market import Market
from betfairlightweight.resources import MarketBook, RunnerBook
from sklearn.preprocessing import StandardScaler

from utils.constants import TIME_BEFORE_START

from tom.toms_utils import normalized_transform


class Strategy1(BaseStrategy):
    def __init__(
        self,
        scaler: StandardScaler,
        ticks_df: pd.DataFrame,
        test_analysis_df: pd.DataFrame,
        model,
        clm,
        *args,
        **kwargs,
    ):
        self.scaler = scaler
        self.ticks_df = ticks_df
        self.model = model
        self.clm = clm
        self.test_analysis_df = test_analysis_df
        super_kwargs = kwargs.copy()
        super_kwargs.pop("scaler", None)
        super_kwargs.pop("ticks_df", None)
        super_kwargs.pop("test_analysis_df", None)
        super_kwargs.pop("model", None)
        super_kwargs.pop("clm", None)
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
        _ = self.process_fundamentals(market_book)
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
        # runner_count = 0
        seconds_to_start = (
            market_book.market_definition.market_time - market_book.publish_time
        ).total_seconds()
        self.seconds_to_start = seconds_to_start

        return True

    def process_runners(self):
        # This doesn't need to do anything.
        pass

    def __send_order(
        self,
        market_id: float,
        runner: RunnerBook,
        price_adjusted: np.float64,
        bsp_value: np.float64,
        market: Market,
        side: str,
    ):
        trade = Trade(
            market_id=str(market_id),
            selection_id=runner.selection_id,
            handicap=runner.handicap,
            strategy=self,
        )

        if price_adjusted > bsp_value:
            self.q_correct += 1
        else:
            self.q_incorrect += 1

        size = (
            self.stake
            if side == "BACK"
            else round(self.stake / (price_adjusted - 1), 2)
        )
        order = trade.create_order(
            side=side,
            order_type=LimitOrder(
                price=price_adjusted,
                size=size,
                persistence_type="LAPSE",
            ),
        )

        print(
            f"{side} bet order created with: \n{side} price adjusted {price_adjusted}\norder size {size}"
        )
        tracker = self.back_bet_tracker if side == "BACK" else self.lay_bet_tracker
        tracker[market_id][runner.selection_id] = [
            order,
            bsp_value,
            str(market_id),
            runner.selection_id,
            runner.handicap,
            runner.sp.actual_sp,
            price_adjusted,
        ]

        matched_tracker = (
            self.matched_back_bet_tracker
            if side == "BACK"
            else self.matched_lay_bet_tracker
        )

        matched_tracker[market_id][runner.selection_id] = False

        market.place_order(order)

        self.q_margin += size * (price_adjusted - bsp_value) / price_adjusted

    def __preprocess_test_analysis(self):
        test_analysis_df = self.test_analysis_df.dropna()
        test_analysis_df = test_analysis_df[
            (test_analysis_df["mean_120"] <= 50) & (test_analysis_df["mean_120"] > 1.1)
        ]
        test_analysis_df = test_analysis_df[test_analysis_df["mean_14400"] > 0]
        test_analysis_df = test_analysis_df.drop(
            test_analysis_df[test_analysis_df["std_2700"] > 1].index
        )

        test_analysis_df_y = pd.DataFrame().assign(
            market_id=test_analysis_df["market_id"],
            selection_ids=test_analysis_df["selection_ids"],
            bsps=test_analysis_df["bsps"],
        )

        return test_analysis_df, test_analysis_df_y

    def __get_model_prediction_and_mean_120(
        self, test_analysis_df: pd.DataFrame, runner: RunnerBook, market_id: float
    ) -> Tuple[np.float64, np.float64]:
        predict_row = test_analysis_df.loc[
            (test_analysis_df["selection_ids"] == runner.selection_id)
            & (test_analysis_df["market_id"] == market_id)
        ]
        mean_120 = predict_row["mean_120"].values[0]
        predict_row = normalized_transform(predict_row, self.ticks_df)
        predict_row = predict_row.drop(["bsps_temp", "bsps"], axis=1)
        predict_row = pd.DataFrame(self.scaler.transform(predict_row), columns=self.clm)
        runner_predicted_bsp = self.model.predict(predict_row)

        return runner_predicted_bsp, mean_120

    def __get_back_lay(
        self,
        test_analysis_df_y: pd.DataFrame,
        mean_120: np.float64,
        runner: RunnerBook,
        market_id: float,
    ) -> Tuple[np.float64, np.float64, np.float64]:
        # price = self.ticks_df.iloc[self.ticks_df["tick"].sub(mean_120).abs().idxmin()][
        #     "tick"
        # ]
        number = self.ticks_df.iloc[self.ticks_df["tick"].sub(mean_120).abs().idxmin()][
            "number"
        ]
        number_adjust = number
        confidence_number = number + 4
        confidence_price = self.ticks_df.iloc[
            self.ticks_df["number"].sub(confidence_number).abs().idxmin()
        ]["tick"]
        price_adjusted = self.ticks_df.iloc[
            self.ticks_df["number"].sub(number_adjust).abs().idxmin()
        ]["tick"]
        bsp_row = test_analysis_df_y.loc[
            (test_analysis_df_y["selection_ids"] == runner.selection_id)
            & (test_analysis_df_y["market_id"] == market_id)
        ]
        bsp_value = bsp_row["bsps"].values[0]

        return price_adjusted, confidence_price, bsp_value

    def process_market_book(self, market: Market, market_book: MarketBook) -> None:
        # process marketBook object
        # Take each incoming message and combine to a df
        cont = self.process_fundamentals(market_book)

        self.market_open = market_book.status
        # We want to limit our betting to before the start of the race.

        market_id = float(market_book.market_id)
        if (
            self.seconds_to_start < 120
            and self.seconds_to_start > 100
            and self.check_market_book(market, market_book)
        ):
            for runner in market_book.runners:

                self.back_bet_tracker.setdefault(market_id, {})
                self.matched_back_bet_tracker.setdefault(market_id, {})
                self.lay_bet_tracker.setdefault(market_id, {})
                self.matched_lay_bet_tracker.setdefault(market_id, {})

                runner_in_back_tracker = (
                    runner.selection_id in self.back_bet_tracker[market_id].keys()
                )
                runner_in_lay_tracker = (
                    runner.selection_id in self.lay_bet_tracker[market_id].keys()
                )
                if not runner_in_back_tracker or not runner_in_lay_tracker:
                    self.back_bet_tracker[market_id][runner.selection_id] = {}
                    self.matched_back_bet_tracker[market_id][runner.selection_id] = {}

                    if runner.status == "ACTIVE":
                        print("RUNNER is active.")
                        try:
                            (
                                test_analysis_df,
                                test_analysis_df_y,
                            ) = self.__preprocess_test_analysis()
                            # if bsps not available skip
                            (
                                runner_predicted_bsp,
                                mean_120,
                            ) = self.__get_model_prediction_and_mean_120(
                                test_analysis_df, runner, market_id
                            )

                        except Exception as e:
                            error_message = f"An error occurred during preprocessing test_analysis_df and/or getting model prediction: {e.__class__.__name__}"
                            print(f"{error_message} - {e}")
                            runner_predicted_bsp = mean_120 = False

                        if mean_120:
                            # in the back_price / number put in where you want to base from: prevoisly runner.last_price_traded
                            # get_price(runner.ex.available_to_back, 1)
                            try:
                                (
                                    price_adjusted,
                                    confidence_price,
                                    bsp_value,
                                ) = self.__get_back_lay(
                                    test_analysis_df_y, mean_120, runner, market_id
                                )

                                # SEND BACK BET ORDER
                                if (
                                    (runner_predicted_bsp < confidence_price)
                                    and (mean_120 <= 50)
                                    and (mean_120 > 1.1)
                                    and not runner_in_back_tracker
                                ):
                                    self.back_bet_tracker[market_id].setdefault(
                                        runner.selection_id, {}
                                    )
                                    self.matched_back_bet_tracker[market_id].setdefault(
                                        runner.selection_id, {}
                                    )
                                    self.__send_order(
                                        market_id,
                                        runner,
                                        price_adjusted,
                                        bsp_value,
                                        market,
                                        side="BACK",
                                    )

                                # SEND LAY BET ORDER
                                if (
                                    (runner_predicted_bsp > confidence_price)
                                    and (price_adjusted <= self.stake)
                                    and (price_adjusted > 1.1)
                                    and not runner_in_lay_tracker
                                ):
                                    self.lay_bet_tracker[market_id].setdefault(
                                        runner.selection_id, {}
                                    )
                                    self.matched_lay_bet_tracker[market_id].setdefault(
                                        runner.selection_id, {}
                                    )
                                    self.__send_order(
                                        market_id,
                                        runner,
                                        price_adjusted,
                                        bsp_value,
                                        market,
                                        side="LAY",
                                    )
                            except Exception as e:
                                error_message = f"An error occurred during order process: {e.__class__.__name__} - {e}"
                                print(error_message)

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

                                # difference to look at
                                layed_size = round(
                                    self.stake / (lay_price_adjusted - 1), 2
                                )
                                print(
                                    "in the lay order.size_matched is ",
                                    order.size_matched,
                                )
                                print("supposed layed size ", layed_size)
                                if (
                                    order.size_matched > 1  # difference to look at
                                ):  # because min liability on back is 1
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
                                    )  # diff
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
