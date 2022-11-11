# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 15:29:55 2021

@author: x4nno
"""

"""functions which will be used for processing historic and later - live data."""
    

import logging
import betfairlightweight
from betfairlightweight import StreamListener
import os
import shutil
import pandas as pd
import numpy as np


def market_collector(bz2_folder_path, results_folder_path, trading, listener):
    """Takes a folder with .bz2 historic betfair data in and collects all the
    similar markets (for instance, MatchOdds) into seperate folders in the results_folder_path
    
    **note if you have a month or day folder nested then you need to loop this 
    outside of this function
    
    This function doesn't return anything"""
    
    for event in os.listdir(bz2_folder_path):
        event_path = bz2_folder_path +"/{0}".format(event) #event
        for market in os.listdir(event_path)[:-1]: #need this in to take out the final "all" file
            file_path = event_path +"/{0}".format(market) #market
            #print ("file path is ", file_path)
            
            # create historical stream (update file_path to your file location)
            stream = trading.streaming.create_historical_generator_stream(
                file_path=file_path,
                listener=listener)
            
            # create generator
            print(file_path)
            gen = stream.get_generator()
                        
            # check the name of the market
            i=0
            for market_books in gen():
                # we don't need to loop over because they are all the same market
                # just use the first one [0]
                if i > 0:
                    break
    
                market_name = market_books[0].market_definition.name
                market_name = market_name.replace(" ", "") # dir don't like spaces ... 
                market_name = market_name.replace("?", "Questionmark")
                market_name = market_name.replace("+", "plus")
                market_name = market_name.replace("-", "neg")
                market_name = market_name.replace(".","point")
                market_name = market_name.replace("/", "slash")
                
                market_name_list = os.listdir(results_folder_path)
                
                if market_name not in market_name_list:
                    new_dir = os.path.join(results_folder_path,market_name)
                    os.mkdir(new_dir)
                    new_dst = os.path.join(new_dir,market)
                    shutil.copyfile(file_path,new_dst)
                    market_name_list.append(market_name)
                else:
                    new_dir = os.path.join(results_folder_path,market_name)
                    new_dst = os.path.join(new_dir,market)
                    shutil.copyfile(file_path,new_dst)
                    
                # print(month)
                # print(day)
                # print(event)
                # print(market)
                # print(market_name)
                
                i += 1

def marketFolder_to_csv(market_collated_folder_path, trading, listener):
    """This function takes a folder which will have inside it the types of
    markets of interest from the collation of the market_collector function. 
    it will then process each market within and create a folder named as that 
    marketID. Inside the folder will be the complete csv files for 
    back, lay, volume, fundamental (basics) and the original .bz2 file"""
    
    column_names = ["MarketName","MarketId","SelectionId","Time","seconds_to_start","Status","Inplay",
                "MarketTotalMatched","SelectionTotalMatched",
                "LastPriceTraded","SP_near","SP_far",
                "SP_actual","BackPrice1","BackSize1","LayPrice1", "LaySize1"
               ]
    market_name_list = os.listdir(market_collated_folder_path)
    
    for market in market_name_list:
        market_path = market_collated_folder_path + "/{0}".format(market)
        for specific in os.listdir(market_path):
            if specific[-3:] != "bz2": #allows us to run this when we have other folders and files in the dir
                pass
            else:
                file_path = market_path +"/{0}".format(specific) #market
                #print ("file path is ", file_path)
                
                # create historical stream (update file_path to your file location)
                stream = trading.streaming.create_historical_generator_stream(
                    file_path=file_path,
                    listener=listener)
                
                # create generator
                gen = stream.get_generator()
                            
                # set up the dict lists 
        
                master_fundamental_list = []
                master_back_dict_list = []
                master_lay_dict_list = []
                master_volume_dict_list = []
                
                counter = 0
                master_counter = 0
                for market_books in gen():
                    for market_book in market_books:
                        for runner in market_book.runners:
                            counter = counter +1
                            master_counter = master_counter + 1
                            # how to get runner details from the market definition
                            market_def = market_book.market_definition
                            seconds_to_start = (
                            market_book.market_definition.market_time - market_book.publish_time
                            ).total_seconds()
                            runners_dict = {
                                (runner.selection_id, runner.handicap): runner
                                for runner in market_def.runners
                            }
                            runner_def = runners_dict.get((runner.selection_id, runner.handicap,
                                                           runner.total_matched, runner.last_price_traded,
                                                          runner.sp, runner.ex))
                            
                            back_dict = {0.0001: market_book.market_definition.name, 0.0002: runner.selection_id, "seconds_to_start": seconds_to_start}
                            lay_dict = {0.0001: market_book.market_definition.name, 0.0002: runner.selection_id, "seconds_to_start": seconds_to_start}
                            volume_dict = {0.0001: market_book.market_definition.name, 0.0002: runner.selection_id, "seconds_to_start": seconds_to_start}
                                        
                            for i in range(len(runner.ex.available_to_back)):
                                    back_dict[runner.ex.available_to_back[i].price] = runner.ex.available_to_back[i].size
                            for i in range(len(runner.ex.available_to_lay)):        
                                    lay_dict[runner.ex.available_to_lay[i].price] = runner.ex.available_to_lay[i].size
                            for i in range(len(runner.ex.traded_volume)):      
                                    volume_dict[runner.ex.traded_volume[i].price] = runner.ex.traded_volume[i].size
                                    
                            master_back_dict_list.append(back_dict)
                            master_lay_dict_list.append(lay_dict)
                            master_volume_dict_list.append(volume_dict)
                            
                            # below forms the fundamental lists
                            
                            temp_list = [market_book.market_definition.name,
                                         market_book.market_id,
                                         runner.selection_id,
                                         market_book.publish_time, 
                                         seconds_to_start,
                                         market_book.status,
                                         market_book.inplay,
                                         market_book.total_matched,
                                         runner.total_matched,
                                         runner.last_price_traded or "",
                                         runner.sp.near_price,
                                         runner.sp.far_price,
                                         runner.sp.actual_sp,]
                            
                            if len(runner.ex.available_to_back) > 0:
                                back_list_temp = [runner.ex.available_to_back[0].price, runner.ex.available_to_back[0].size]
                            else:
                                back_list_temp = ["None","None"]
                            
                            if len(runner.ex.available_to_lay) > 0:
                                lay_list_temp = [runner.ex.available_to_lay[0].price, runner.ex.available_to_lay[0].size]
                            else:
                                lay_list_temp = ["None","None"]
                            for i in range(2):
                                temp_list.append(back_list_temp[i])
                            for i in range(2):
                                temp_list.append(lay_list_temp[i])
                            
                            master_fundamental_list.append(temp_list)
                            
                            if counter >= 100000:
                                market_id_folder_list = os.listdir(market_path)
                                if market_book.market_id not in market_id_folder_list:
                                    new_dir = os.path.join(market_path,market_book.market_id)
                                    os.mkdir(new_dir)
                                else:
                                    new_dir = os.path.join(market_path,market_book.market_id)
                                
                                if len(master_back_dict_list) != 0 and len(master_lay_dict_list) != 0 and len(master_volume_dict_list) != 0:
                                    df_back = pd.DataFrame(master_back_dict_list)
                                    df_back = df_back.set_index("seconds_to_start")
                                    df_back = df_back.sort_index(axis=1)
                                    df_back.to_csv("{0}/{1}_back_{2}.csv".format(new_dir, market_book.market_id, master_counter))
                            
                                    df_lay = pd.DataFrame(master_lay_dict_list)
                                    df_lay = df_lay.set_index("seconds_to_start")
                                    df_lay = df_lay.sort_index(axis=1)
                                    df_lay.to_csv("{0}/{1}_lay_{2}.csv".format(new_dir, market_book.market_id, master_counter))
                                    
                                    df_volume = pd.DataFrame(master_volume_dict_list)
                                    df_volume = df_volume.set_index("seconds_to_start")
                                    df_volume = df_volume.sort_index(axis=1)
                                    df_volume.to_csv("{0}/{1}_volume_{2}.csv".format(new_dir, market_book.market_id, master_counter))
                                    
                                    df_fundamental = pd.DataFrame(master_fundamental_list, columns = column_names)
                                    df_fundamental.to_csv("{0}/{1}_fundamental_{2}.csv".format(new_dir, market_book.market_id, master_counter))   
                                
                                #reset everything
                                counter = 0
                                master_fundamental_list = []
                                master_back_dict_list = []
                                master_lay_dict_list = []
                                master_volume_dict_list = []
                                    
                #catch anything at the end where the counter is below 100000 but we have run out.                
                market_id_folder_list = os.listdir(market_path)
                if market_book.market_id not in market_id_folder_list:
                    new_dir = os.path.join(market_path,market_book.market_id)
                    os.mkdir(new_dir)
                else:
                    new_dir = os.path.join(market_path,market_book.market_id)
                
                if len(master_back_dict_list) != 0 and len(master_lay_dict_list) != 0 and len(master_volume_dict_list) != 0:
                    df_back = pd.DataFrame(master_back_dict_list)
                    df_back = df_back.set_index("seconds_to_start")
                    df_back = df_back.sort_index(axis=1)
                    df_back.to_csv("{0}/{1}_back_{2}.csv".format(new_dir, market_book.market_id, master_counter))
            
                    df_lay = pd.DataFrame(master_lay_dict_list)
                    df_lay = df_lay.set_index("seconds_to_start")
                    df_lay = df_lay.sort_index(axis=1)
                    df_lay.to_csv("{0}/{1}_lay_{2}.csv".format(new_dir, market_book.market_id, master_counter))
                    
                    df_volume = pd.DataFrame(master_volume_dict_list)
                    df_volume = df_volume.set_index("seconds_to_start")
                    df_volume = df_volume.sort_index(axis=1)
                    df_volume.to_csv("{0}/{1}_volume_{2}.csv".format(new_dir, market_book.market_id, master_counter))
                    
                    df_fundamental = pd.DataFrame(master_fundamental_list, columns = column_names)
                    df_fundamental.to_csv("{0}/{1}_fundamental_{2}.csv".format(new_dir, market_book.market_id, master_counter))
                        
                shutil.move(file_path,new_dir)      

def to_combined_csv(market_type_folder_path, records_path):
    
    ## ** THIS SHOULDNT BE USED ANYMORE IT IS WAY TO SLOW **
    
    """ takes the folder for the market type (for instance MatchOdds) and the path for records of markets 
    proccessed. creates a file with combined infomation on. doesn't return anything. A note - this runs 
    completely on the pandas framework and is very slow."""
    
    column_names =      ["SecondsToStart", "SelectionId", "UniqueValue","MarketTotalMatched","SelectionTotalMatched", "LastPriceTraded", "volume_last_price",
                         "available_to_back_1_price", "available_to_back_1_size", "volume_traded_at_Bprice1",
                         "available_to_back_2_price", "available_to_back_2_size", "volume_traded_at_Bprice2",
                         "available_to_back_3_price", "available_to_back_3_size", "volume_traded_at_Bprice3",
                         "reasonable_back_WoM",
                         "available_to_lay_1_price", "available_to_lay_1_size", "volume_traded_at_Lprice1",
                         "available_to_lay_2_price", "available_to_lay_2_size", "volume_traded_at_Lprice2",
                         "available_to_lay_3_price", "available_to_lay_3_size", "volume_traded_at_Lprice3",
                         "reasonable_lay_WoM"
                         ]
    
    
    numeric_columns = ["MarketTotalMatched","SelectionTotalMatched","LastPriceTraded",
                       "available_to_back_1_price", "available_to_back_1_size", "volume_traded_at_Bprice1",
                       "available_to_back_2_price", "available_to_back_2_size", "volume_traded_at_Bprice2",
                       "available_to_back_3_price", "available_to_back_3_size", "volume_traded_at_Bprice3",
                        "reasonable_back_WoM",
                        "available_to_lay_1_price", "available_to_lay_1_size", "volume_traded_at_Lprice1",
                        "available_to_lay_2_price", "available_to_lay_2_size", "volume_traded_at_Lprice2",
                        "available_to_lay_3_price", "available_to_lay_3_size", "volume_traded_at_Lprice3",
                        "reasonable_lay_WoM"
                        ]
    
    folder_list = os.listdir(market_type_folder_path)
    combined_path = os.path.join(market_type_folder_path, "combined")
    
    str_check = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
    for folder in folder_list:
        print("folder is :", folder)
        folder_path = os.path.join(market_type_folder_path, folder)
        f = open(records_path)
        # make sure we are only dealing with the correct folders
        if folder not in f.read() and len(os.listdir(folder_path)) == 5:
            f.close() #save memory
            
            data_path = os.path.join(market_type_folder_path,folder)
            dir_list = os.listdir(data_path)
            
            #open each of the data_frames we have previously created
            back_df = pd.read_csv(data_path+"/"+dir_list[1])
            back_df.rename(columns={"0.0002":"SelectionId"}, inplace=True)
            lay_df = pd.read_csv(data_path+"/"+dir_list[3])
            lay_df.rename(columns={"0.0002":"SelectionId"}, inplace=True)
            fundamental_df = pd.read_csv(data_path+"/"+dir_list[2], index_col=[0])
            volume_df = pd.read_csv(data_path+"/"+dir_list[4])
            volume_df.rename(columns={"0.0002":"SelectionId"}, inplace=True)

            #create a unique identifier for each line
            df_list = [fundamental_df, back_df, lay_df, volume_df]
            for df in df_list:
                df["seconds_to_start"] = df["seconds_to_start"].round(2) #stops strange rounding
                df["unique_row_id"] = df["seconds_to_start"].apply(str)+"_"+df["SelectionId"].apply(str)
            
            # iterate over these unique_row_id and get the relevant prices and volume  
            # either side of the last price traded.
            unique_values = fundamental_df["unique_row_id"].unique()
            master_list = []
            counter = 0
            for unique_value in unique_values:
                print(counter)
                counter += 1
                row_list = []
                back_row = back_df.loc[back_df.unique_row_id == unique_value]
                back_row = back_row.dropna(axis=1)
                lay_row = lay_df.loc[lay_df.unique_row_id == unique_value]
                lay_row = lay_row.dropna(axis=1)
                volume_row = volume_df.loc[volume_df.unique_row_id == unique_value]
                volume_row = volume_row.dropna(axis=1)
                fundamental_row = fundamental_df.loc[fundamental_df.unique_row_id == unique_value]
                
                # Fundamental_row_infomation
                seconds_to_start = fundamental_row["seconds_to_start"].iat[0]
                selection_id = fundamental_row["SelectionId"].iat[0]
                market_total_matched = fundamental_row["MarketTotalMatched"].iat[0]
                selection_total_matched = fundamental_row["SelectionTotalMatched"].iat[0]
                last_price_traded = fundamental_row["LastPriceTraded"].iat[0]
                
                row_list.extend([seconds_to_start, selection_id, unique_value,
                                 market_total_matched, selection_total_matched, last_price_traded])
                
                if last_price_traded != np.nan:                 # can add in any other filters here
                    # this will find the volume traded at the last price traded
                    if last_price_traded in volume_row.columns:
                        volume_last_price = volume_row[last_price_traded].iat[0]
                    else:
                        volume_last_price = 0
                else:
                    volume_last_price = 0
                
                row_list.append(volume_last_price)
                # this will get all the back price, back size and volume traded at these numbers
                reasonable_back_WoM = 0
                for i in range(3):
                    j = i + 3
                    if j < len(back_row.columns)-1:
                        back_price = back_row.columns[j]
                        if back_price not in str_check:
                            back_size = back_row.iat[0,j]
                            if back_price in volume_row.columns:
                                volume_size_back = volume_row[back_price].iat[0]
                            else:
                                volume_size_back = 0
                        else:
                            back_price = 0
                            back_size = 0
                            volume_size_back = 0
                    else:
                        back_price = 0
                        back_size = 0
                        volume_size_back = 0
                    
                    reasonable_back_WoM += float(back_size)
                    row_list.extend([back_price, back_size, volume_size_back])
                
                row_list.append(reasonable_back_WoM)
                
                # this will get all the lay prices, back size and volume traded at these numbers
                reasonable_lay_WoM = 0
                for i in range(3):
                    j = i + 3
                    if j < len(lay_row.columns)-1:
                        lay_price = lay_row.columns[j]
                        if lay_price != "unique_row_id":
                            lay_size = lay_row.iat[0,j]
                            if lay_price in volume_row.columns:
                                volume_size_lay = volume_row[lay_price].iat[0]
                            else:
                                volume_size_lay = 0
                        else:
                            lay_price = 0
                            lay_size = 0
                            volume_size_lay = 0
                    else:
                        lay_price = 0
                        lay_size = 0
                        volume_size_lay = 0
                        
                    reasonable_lay_WoM += float(back_size)
                    row_list.extend([lay_price, lay_size, volume_size_lay])
                            
                row_list.append(reasonable_lay_WoM)
                            
                master_list.append(row_list)
                
            df_major = pd.DataFrame(master_list, columns = column_names)
            
            df_major.to_csv("{}/{}.csv".format(combined_path, folder))
            
            with open(records_path, 'a') as f:
                f.write('{0}\n'.format(folder))
                print("folder complete : ", folder)
        else:
            f.close()
            
def marketFolder_to_combined_csv(market_collated_folder_path, trading, listener):
    """This function takes a folder which will have inside it the types of
    markets of interest from the collation of the market_collector function. 
    it will then process each market within and create a folder named as that 
    marketID. Inside the folder will be the complete csv files for 
    a combined back,lay, fundamental and volume and the original .bz2 file
    
    a note this will run faster than pandas functions of collection."""
    
    
    column_names =      ["SecondsToStart", "MarketId", "SelectionId", "MarketTotalMatched","SelectionTotalMatched", "LastPriceTraded", "volume_last_price",
                         "available_to_back_1_price", "available_to_back_1_size", "volume_traded_at_Bprice1",
                         "available_to_back_2_price", "available_to_back_2_size", "volume_traded_at_Bprice2",
                         "available_to_back_3_price", "available_to_back_3_size", "volume_traded_at_Bprice3",
                         "reasonable_back_WoM",
                         "available_to_lay_1_price", "available_to_lay_1_size", "volume_traded_at_Lprice1",
                         "available_to_lay_2_price", "available_to_lay_2_size", "volume_traded_at_Lprice2",
                         "available_to_lay_3_price", "available_to_lay_3_size", "volume_traded_at_Lprice3",
                         "reasonable_lay_WoM", "bsp"
                         ]
    
    market_name_list = os.listdir(market_collated_folder_path)
    
    for market in market_name_list:
        market_path = market_collated_folder_path + "/{0}".format(market)
        for specific in os.listdir(market_path):
            if specific[-3:] != "bz2": #allows us to run this when we have other folders and files in the dir
                pass
            else:
                file_path = market_path +"/{0}".format(specific) #market
                print ("file path is ", file_path)
                
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
                            counter = counter +1
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
                            
                            #Set up the dictionaries 
                            back_dict = {}
                            lay_dict = {}
                            volume_dict = {}
                            
                            # this seems strange because we could just implement the back and lays directly
                            # however because we need to match the prices to volumes traded this is easier.
                            for i in range(len(runner.ex.available_to_back)):
                                    back_dict[runner.ex.available_to_back[i].price] = runner.ex.available_to_back[i].size
                            for i in range(len(runner.ex.available_to_lay)):        
                                    lay_dict[runner.ex.available_to_lay[i].price] = runner.ex.available_to_lay[i].size
                            for i in range(len(runner.ex.traded_volume)):      
                                    volume_dict[runner.ex.traded_volume[i].price] = runner.ex.traded_volume[i].size
                            
                            # Below will check traded volume at the last price traded 
                            if temp_list[5] != "":
                                if temp_list[5] in volume_dict.keys():#need the extra for an edge case (doesn't seem like this could ever happen logically)
                                    temp_list.append(volume_dict[temp_list[5]])
                                else:
                                    temp_list.append(0)
                            else:
                                temp_list.append(0)
                            
                            back_prices = list(back_dict.keys())
                            lay_prices= list(lay_dict.keys())
                            reasonable_back_WoM = 0
                            reasonable_lay_WoM = 0
                            #back
                            for i in range(3):
                                if i < len(back_prices)-1:
                                    temp_list.extend([back_prices[i],
                                                    back_dict[back_prices[i]],
                                                    ])
                                    if back_prices[i] in volume_dict.keys():
                                        temp_list.append(volume_dict[back_prices[i]])
                                    else:
                                        temp_list.append("")
                                    reasonable_back_WoM += back_dict[back_prices[i]]
                                else:
                                    temp_list.extend(["","",""])
                            temp_list.append(reasonable_back_WoM)
                            #lay
                            for i in range(3):
                                if i < len(lay_prices)-1:
                                    temp_list.extend([lay_prices[i],
                                                    lay_dict[lay_prices[i]],
                                                    ])   
                                    if lay_prices[i] in volume_dict.keys():
                                        temp_list.append(volume_dict[lay_prices[i]])
                                    else:
                                        temp_list.append("")
                                    reasonable_lay_WoM += lay_dict[lay_prices[i]]
                                else:
                                    temp_list.extend(["","",""])
                            temp_list.append(reasonable_lay_WoM)

                            temp_list.append(runner.ex.bsp or "")
                            
                            master_list.append(temp_list)
                new_dir = market_path + "/{0}".format(market_book.market_id)
                os.mkdir(new_dir)
                df_combined = pd.DataFrame(master_list, columns = column_names)
                
                df_combined.to_csv(new_dir+"/{0}_combined.csv".format(market_book.market_id))

                shutil.move(file_path,new_dir)
                
def combined_to_selection_CSV(market_collated_folder_path):
    """This takes the directory which has the folders produced from marketFolder_to_combined_csv, 
    and seperates them into selection_id only dataframes, leaves them in the same folder.
    
    """
    for marketname in os.listdir(market_collated_folder_path):
        market_name_folder_path = os.path.join(market_collated_folder_path, marketname)
        for folder in os.listdir(market_name_folder_path):
            folder_path = os.path.join(market_name_folder_path, folder)
            file_path = os.path.join(folder_path,"{0}_combined.csv".format(folder))
            combined_df = pd.read_csv(file_path)
            selection_ids = combined_df["SelectionId"].unique()
            temp_dict = {}
            identifier = "first"
            for selection_id in selection_ids:
                selection_df = combined_df[combined_df["SelectionId"] == selection_id]
                if selection_id == 58805:
                    new_file_path = os.path.join(folder_path,"draw_{1}.csv".format(identifier,selection_id))
                    selection_df = selection_df.drop("Unnamed: 0", axis=1)
                    for column in selection_df.columns:
                        selection_df = selection_df.rename(columns={column:column+"_draw"})
                else:
                    new_file_path = os.path.join(folder_path,"{0}_{1}.csv".format(identifier,selection_id))
                    selection_df = selection_df.drop("Unnamed: 0", axis=1)
                    for column in selection_df.columns:
                        selection_df = selection_df.rename(columns={column:column+"_{0}".format(identifier)})
                    identifier = "second"    
                
                selection_df.to_csv(new_file_path)
                
def combined_to_selection_CSV_horses(market_collated_folder_path):
    """This takes the directory which has the folders produced from marketFolder_to_combined_csv, 
    and seperates them into selection_id only dataframes, leaves them in the same folder - this is different to the won above 
    by that it will save each selection id as it's own file instead of first, draw and second
    
    """
    for marketname in os.listdir(market_collated_folder_path):
        market_name_folder_path = os.path.join(market_collated_folder_path, marketname)
        for folder in os.listdir(market_name_folder_path):
            folder_path = os.path.join(market_name_folder_path, folder)
            file_path = os.path.join(folder_path,"{0}_combined.csv".format(folder))
            combined_df = pd.read_csv(file_path)
            selection_ids = combined_df["SelectionId"].unique()
            temp_dict = {}
            for selection_id in selection_ids:
                selection_df = combined_df[combined_df["SelectionId"] == selection_id]
                new_file_path = os.path.join(folder_path, str(selection_id)+".csv") #horrible hacky method sorry.
                selection_df = selection_df.drop("Unnamed: 0", axis=1)
                for column in selection_df.columns:
                    selection_df = selection_df.rename(columns={column:column+"_{0}".format(selection_id)})
                    
                selection_df.to_csv(new_file_path)


def selections_to_combined_long(market_collated_folder_path):
    """this function will take the individual selection csv's and produce a combined database
    with all of the them labelled as 'first', 'draw' and 'second' "numbers in aswell.
    It will save it into the same folder as a combined_long file."""
    
    id_list = ["draw","first","second"]
    for marketname in os.listdir(market_collated_folder_path):
        market_name_folder = os.path.join(market_collated_folder_path, marketname)
        for folder in os.listdir(market_name_folder):
            df_dict_temp = {}
            folder_path = os.path.join(market_name_folder, folder)
            file_list = os.listdir(folder_path)
            for file in file_list:
                if file.split("_")[0] in id_list:      #this checks if it is either draw, first or second
                    specific_file = os.path.join(folder_path,file)
                    df_temp = pd.read_csv(specific_file)
                    df_dict_temp[file.split("_")[0]] = df_temp
            keys_list = list(df_dict_temp.keys())
            for key in keys_list:
                df_dict_temp[key] = df_dict_temp[key].set_index("SecondsToStart_{0}".format(key))
            if keys_list == ["draw","first","second"]:
                df_combined_long = pd.concat([df_dict_temp["draw"],df_dict_temp["first"],df_dict_temp["second"]], axis=1)
                combined_long_path = os.path.join(folder_path,"{}_combined_long.csv".format(folder))
                df_combined_long.to_csv(combined_long_path)
                
def selections_to_combined_long_horses(market_collated_folder_path):
    """this function needs re-looking at, this should take all of the individual dataframes and concatenate onto one line for insights
    at the moment this is not as simple as changing some lines from the football one so am leaving for now """
    pass
            
def combined_long_clean(market_collated_folder_path, STS_restrictions = None, MTM_restrictions = None):
    """ this function takes the market collated folder path and checks in each folder for a combined_long file
    the function will then clean the columns (default) and will restrict the data to seconds to start (list lower, upper) and 
    totalmarketmatched (lower)
    
    Note that this does not clean out the selectionIds which will need to be removed before loaded into a model"""
    
    id_list = ["draw","first","second"]
    for marketname in os.listdir(market_collated_folder_path):
        market_name_folder = os.path.join(market_collated_folder_path, marketname)
        for folder in os.listdir(market_name_folder):
            folder_path = os.path.join(market_name_folder, folder)
            file_list = os.listdir(folder_path)
            for file in file_list:
                if file.split("_")[-1] == "long.csv":
                    long_dir = os.path.join(folder_path,file)
                    long_df = pd.read_csv(long_dir)
                    
                    for id_ in id_list:
                        long_df = long_df.drop(["available_to_back_3_price_{}".format(id_),"available_to_back_3_size_{}".format(id_),
                                               "volume_traded_at_Bprice3_{}".format(id_),"available_to_lay_3_price_{}".format(id_),
                                               "available_to_lay_3_size_{}".format(id_),"volume_traded_at_Lprice3_{}".format(id_)], 
                                               axis = 1)
                    long_df = long_df.drop(["Unnamed: 0.1", "Unnamed: 0.2", "Unnamed: 0.3", "MarketId_first", "MarketId_second",
                                            "MarketTotalMatched_first", "MarketTotalMatched_second"], axis = 1)
                    
                    # column cleaning complete
                    ## restrictions being applied below
                    if STS_restrictions != None:
                        long_df = long_df[(long_df["Unnamed: 0"] < STS_restrictions[0]) & (long_df["Unnamed: 0"] > STS_restrictions[1])]
                    if MTM_restrictions != None:
                        long_df = long_df[(long_df["MarketTotalMatched_draw"] > MTM_restrictions)]
                                        
                    long_df = long_df.set_index("Unnamed: 0")
                    
                    long_df.to_csv("{0}/{1}_combined_long_clean.csv".format(folder_path, folder))
                
def pipeline(month_folder, results_folder_path, trading, listener, STS_restrictions = None, MTM_restrictions = None):
    """This function will take all of the above functions and run them from the monthly folder for instance 'JAN' and 
    apply these functions correctly in order to produce the results which are ready for processing, they will be saved 
    in a folder inside the collate folder
    
    note the end of this will purge everything except the combined_long_clean file."""
    
    #collate the folders
    for day in os.listdir(month_folder):
        day_path = os.path.join(month_folder, day)
        market_collector(day_path, results_folder_path, trading, listener)
        print ("{0} / {1}".format(day, len(os.listdir(month_folder)))) 
    
    # to csv in a combined format.
    marketFolder_to_combined_csv(results_folder_path, trading, listener)
    # takes the results of the csv's and combines them into seperate selectionIds
    combined_to_selection_CSV(results_folder_path)
    # takes each selection id DF and concats on the horizontal
    selections_to_combined_long(results_folder_path)
    # cleans the data by deleting columns not needed and puts filters in.
    combined_long_clean(results_folder_path, STS_restrictions, MTM_restrictions)
    
def pipeline_without_collation(of_interest_path, trading, listener, STS_restrictions = None, MTM_restrictions = None):
    """This passes the 'of_interest' path into the method and produces a combined_long_clean file which will be ready for
    training
    
    these files will be saved in the same folder as the relevant event type in the 'of_interest' folder."""
    # to csv in a combined format.
    marketFolder_to_combined_csv(of_interest_path, trading, listener)
    # takes the results of the csv's and combines them into seperate selectionIds
    combined_to_selection_CSV(of_interest_path)
    # takes each selection id DF and concats on the horizontal
    selections_to_combined_long(of_interest_path)
    # cleans the data by deleting columns not needed and puts filters in.
    combined_long_clean(of_interest_path, STS_restrictions, MTM_restrictions)
    
    # for marketname in os.listdir(of_interest_path):
    #     marketname_path = os.path.join(of_interest_path, marketname)
    #     for market in os.listdir(marketname_path):
    #         market_path = os.path.join(marketname_path,market)
    #         for file in os.listdir(market_path):
    #             if file.split("_")[-1] != "clean.csv":
    #                 file_path = os.path.join(market_path, file)
    #                 os.remove(file_path)



if __name__ == "__main__":
    # setup logging
    logging.basicConfig(level=logging.INFO)
    # create trading instance (don't need username/password)
    trading = betfairlightweight.APIClient("username", "password", app_key="d5b6fltDUE03k4lQ")
    # create listener
    listener = StreamListener(max_latency=None)
    
    STS_restrictions=[3600, 0]
    MTM_restrictions=1000
    
    #pipeline_without_collation("C:/Users/x4nno/Desktop/results", trading, listener, STS_restrictions = [36000, 0])

    of_interest_path = "/media/x4nno/TOSHIBA EXT/Wanker Tom/Betfair_data/collated_horses_Feb_wins"

    marketFolder_to_combined_csv(of_interest_path, trading, listener)
    combined_to_selection_CSV(of_interest_path)
    
    #marketFolder_to_combined_csv(of_interest_path, trading, listener)
    #combined_to_selection_CSV_horses(of_interest_path)
    
    #pipeline_without_collation("F:/Wanker Tom/Betfair_data/collated_horses_Jan", trading, listener, STS_restrictions=[3600, 0], MTM_restrictions=1000)
    
    #marketFolder_to_combined_csv("F:/Wanker Tom/Betfair_data/collated_horses", trading, listener)
    
    # of_interest_path = "F:/Wanker Tom/Betfair_data/collated_horses"
    # STS_restrictions=[3600, 0]
    # MTM_restrictions=1000
    # combined_to_selection_CSV_horses(of_interest_path)


    
