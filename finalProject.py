# import time and set a starting time to be able to measure computation later on
import time
t0 = time.time()

# standard packages and utilities
import pandas as pd
import json
import os
import copy
import numpy as np
import pickle

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# scientific and more advanced packages
import scipy.signal as signal
import ruptures as rpt

# set path and id:s for Liverpool games
path = '/mnt/d/GitHub/opendata/data' #set the path to the data-folder in the repo
lfc_matches = ['4039', '2440'] # manually identified the match_id for the Liverpool games
lfc_id = 2 # manually identified the id for Liverpool

# create a list of all match_ids
def create_match_list(path):
    with open(path + '/matches.json') as f:
        data = json.load(f)

    match_ids = []

    for match in data:
        match_ids.append(str(match['id']))

    return match_ids

all_matches = create_match_list(path)

# create player_dict for Liverpool players
def create_player_dicts(path, match_ids, team_id):
    lfc_id = team_id

    lfc_dict = {}
    lfc_list = []
    name_dict = {}
    lfc_dict = {}

    for match in match_ids:

        with open(path + '/matches/' + match + '/match_data.json') as f:
            data = json.load(f)

        for player in data['players']:
            obj_id = player['trackable_object']
            first_name = player['first_name']
            last_name = player['last_name']
            name_dict[obj_id] = first_name + ' ' + last_name

            if player['team_id'] == lfc_id:
                obj_id = player['trackable_object']
                first_name = player['first_name']
                last_name = player['last_name']
                name = first_name + ' ' + last_name
                # pos = player['player_role']['acronym']
                '''
                position excluded due to conflicting positions between the two games
                '''
                lfc_dict[obj_id] = [name] #, pos]
                lfc_list.append(obj_id)

    return lfc_dict, lfc_list , name_dict

lfc_dict, lfc_list, name_dict = create_player_dicts(path, match_ids = all_matches, team_id = lfc_id)

# load the tracking data
def load_tracking_data(path, match_ids, drop_limit=6000):

    match_dfs_list = []

    # tot_df = pd.DataFrame()

    for match in match_ids:

        with open(path + '/matches/' + match + '/structured_data.json') as f:
            data = json.load(f)

        frame_list =  []

        for frame in data:
            frame_nr = frame['frame']
            curr_frame = {}
            for obj in frame['data']:
                x = obj['x']
                y = obj['y']
                try:
                    obj_id = obj['trackable_object']
                    curr_frame[str(obj_id)+'_'+str(match)+'_x'] = x
                    curr_frame[str(obj_id)+'_'+str(match)+'_y'] = y
                except:
                    None
            row = pd.Series(curr_frame, name = frame_nr, dtype=np.float64)
            frame_list.append(curr_frame)

        match_df = pd.DataFrame(frame_list)

        if drop_limit != 0:
            drop_cols = []
            for col in match_df:
                col_df = match_df[col]
                c = len([i for i in col_df if not np.isnan(i)]) # frames covered

                if c <= drop_limit: # default: 6000 frames = 600 sec = 10 min
                    drop_cols.append(col)

            match_df = match_df.drop(columns = drop_cols)

        match_dfs_list.append(match_df)

    # print(match_dfs_list)
    # tot_df = tot_df.merge(match_df, how='full', )

    tot_df = pd.concat(match_dfs_list, axis=1)

    return tot_df

try:
    tot_df_all = pd.read_pickle('tot_df_all.pkl')
except:
    tot_df_all = load_tracking_data(path, all_matches, drop_limit = 6000)
    tot_df_all.to_pickle('tot_df_all.pkl')

dt = 0.1 # manually checked the time interval between frames

# computing distance between frames in x and y dimension
def compute_velocities_accelerations(tot_df):

    # compute difference between points
    diff_df = tot_df.diff(periods=1, axis=0)

    # loop to get a list of all players_match combinations
    player_match_list = []
    for col in diff_df.columns:
        if col[-1] == 'x':
            player_match_list.append(col[:-2])

    # loop to compute distance, raw velocity/acceleration and smoothed velocity/acceleration
    for i in player_match_list:
        diff_df[i+'_dist'] = np.sqrt(diff_df[i+'_x']**2 + diff_df[i+'_y']**2)
        diff_df[i+'_speed'] = diff_df[i+'_dist']/0.1
        diff_df[str(i)+'_speed'] = diff_df[str(i)+'_speed'].apply(lambda x: np.nan if x > 12 else x) # Usain Bolt filter
        diff_df[str(i)+'_acc'] = diff_df[str(i)+'_speed'].diff()/0.1

        diff_df[str(i)+'_SG3_speed'] = signal.savgol_filter(diff_df[i+'_speed'], 3, 1, mode='nearest')
        diff_df[str(i)+'_SG5_speed'] = signal.savgol_filter(diff_df[i+'_speed'], 5, 1, mode='nearest')

        diff_df[str(i)+'_SG5_acc'] = diff_df[i+'_SG5_speed'].diff()/0.1
        diff_df[str(i)+'_SG3_SG5_acc'] = signal.savgol_filter(diff_df[i+'_SG5_acc'], 3, 1, mode='nearest')

    # drop the columns with x and y coordinates
    drop_cols = []
    for col in diff_df.columns:
        if (col[-1] == 'x') or (col[-1] == 'y'):
            drop_cols.append(col)

    vel_acc_df = diff_df.drop(columns = drop_cols)

    return vel_acc_df

try:
    vel_acc_df_all = pd.read_pickle('vel_acc_df_all.pkl')
except:
    vel_acc_df_all = compute_velocities_accelerations(tot_df_all)
    vel_acc_df_all.to_pickle('vel_acc_df_all.pkl')

def acc_dec_ratios(vel_acc_df, acc_threshold=2, dec_threshold=-2):

    acc_cols = []
    for col in vel_acc_df.columns:
        if 'SG3_SG5_acc' in col:
            acc_cols.append(col)

    acc_df = vel_acc_df[acc_cols]

    adr_list = []

    for col in acc_df.columns:
        col_df = acc_df[col]
        accelerations = col_df[col_df >= 2]
        decelerations = col_df[col_df <= -2]

        acc_count = len(accelerations)
        dec_count = len(decelerations)

        ratio = acc_count / dec_count
        label = col[:-12]

        labels_split = label.split('_')

        player = labels_split[0]
        match = labels_split[1]

        adr_list.append([player, match, ratio])


    return adr_list

adr = acc_dec_ratios(vel_acc_df_all, acc_threshold=2, dec_threshold=-2)
print(len(adr)/9)

print(f'Time to run the script = {time.time()- t0}')
