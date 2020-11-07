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
from sklearn.cluster import KMeans

# set path and id:s for Liverpool games
# path = '/mnt/d/GitHub/opendata/data' #set the path to the data-folder in the repo
path = 'D:\\GitHub\\opendata\\data'
linux_path = False
lfc_matches = ['4039', '2440'] # manually identified the match_id for the Liverpool games
lfc_id = 2 # manually identified the id for Liverpool

# create a list of all match_ids
def create_match_list(path, linux_path = True):
    if linux_path:
        with open(path + '/matches.json') as f:
            data = json.load(f)
    else:
        with open(path + '\\matches.json') as f:
            data = json.load(f)

    match_ids = []

    for match in data:
        match_ids.append(str(match['id']))

    return match_ids

all_matches = create_match_list(path, linux_path)

# create player_dict for Liverpool players
def create_player_dicts(path, match_ids, team_id, linux_path=True):
    lfc_id = team_id

    lfc_dict = {}
    lfc_list = []
    name_dict = {}
    all_players_list = []

    for match in match_ids:
        
        if linux_path:
            with open(path + '/matches/' + match + '/match_data.json') as f:
                data = json.load(f)
        else:
            with open(path + '\\matches\\' + match + '\\match_data.json') as f:
                data = json.load(f)
    

        for player in data['players']:
            obj_id = player['trackable_object']
            first_name = player['first_name']
            last_name = player['last_name']
            name_dict[obj_id] = first_name + ' ' + last_name
            if obj_id not in all_players_list:
                all_players_list.append(obj_id)

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
                if obj_id not in lfc_list:
                    lfc_list.append(obj_id)

        assert len(lfc_dict) == len(lfc_list)
        assert len(name_dict) == len(all_players_list)

    return lfc_dict, lfc_list, name_dict, all_players_list

lfc_dict, lfc_list, name_dict, all_players_list = create_player_dicts(path, match_ids = all_matches, team_id = lfc_id, linux_path=linux_path)

# load the tracking data
def load_tracking_data(path, match_ids, drop_limit=6000, linux_path=True):

    match_dfs_list = []

    # tot_df = pd.DataFrame()

    for match in match_ids:
        
        if linux_path:
            with open(path + '/matches/' + match + '/structured_data.json') as f:
                data = json.load(f)
        else:
            with open(path + '\\matches\\' + match + '\\structured_data.json') as f:
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

def create_player_match_list(tot_df):

    player_match_list = []

    # loop to get a list of all players_match combinations
    for col in tot_df.columns:
        if col[-1] == 'x':
            player_match_list.append(col[:-2])

    return player_match_list

pm_list = create_player_match_list(tot_df_all)

# computing distance between frames in x and y dimension
def compute_velocities_accelerations(tot_df, player_match_list):

    # compute difference between points
    diff_df = tot_df.diff(periods=1, axis=0)

    # loop to compute distances, raw velocity/acceleration and smoothed velocity/acceleration
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
    vel_acc_df_all = compute_velocities_accelerations(tot_df_all, pm_list)
    vel_acc_df_all.to_pickle('vel_acc_df_all.pkl')

def summary_stats(vel_acc_df, player_match_list, acc_threshold=2, dec_threshold=-2):

    add_cols = []

    for col in vel_acc_df.columns:
        if 'SG3_SG5_acc' in col:
            add_cols.append(col)
        elif 'SG5_speed' in col:
            add_cols.append(col)

    add_df = vel_acc_df[add_cols]
    row_list = []

    for pm in player_match_list:
        dist = add_df[pm+'_SG5_speed'].sum()*0.1
        pm_acc = add_df[pm+'_SG3_SG5_acc']
        acceleration_time = len(pm_acc[pm_acc >= 2])*0.1
        deceleration_time = len(pm_acc[pm_acc <= -2])*0.1
        row_dict = {'distance_covered': dist,
                    'acceleration_time': acceleration_time,
                    'deceleration_time': deceleration_time}
        row = pd.Series(data = row_dict, name = pm)
        row_list.append(row)

    stats_df = pd.DataFrame(row_list).sort_index()

    return stats_df

try:
    stats = pd.read_pickle('summary_stats.pkl')
except:
    stats = summary_stats(vel_acc_df_all, pm_list, acc_threshold=2, dec_threshold=-2)
    stats.to_pickle('summary_stats.pkl')


def summary_stats_aggregated(summary_stats, players_list):
    indices = list(summary_stats.index)
    indices_split= []
    for i in indices:
        indices_split.append(i.split('_'))

    player_stats = []

    for i in players_list:
        idx_list = []
        for j in indices_split:
            # print(type(i), type(j[0]))
            if j[0]==str(i):
                idx_list.append(j)

        dist = 0
        acc = 0
        dec = 0

        row_dict = {}

        for idx in idx_list:
            dist = dist + summary_stats['distance_covered'][idx[0]+'_'+idx[1]]
            acc = acc + summary_stats['acceleration_time'][idx[0]+'_'+idx[1]]
            dec = dec + summary_stats['deceleration_time'][idx[0]+'_'+idx[1]]

        if (dist > 0) or (acc > 0) or (dec > 0): # only append rows with non-zero values
            row_dict = {'distance_covered': dist,
                        'acceleration_time': acc,
                        'deceleration_time': dec}

            row = pd.Series(data = row_dict, name = i)

            player_stats.append(row)

    stats_aggregated_df = pd.DataFrame(player_stats).sort_index()

    stats_aggregated_df['acc/dec-ratio'] = stats_aggregated_df['acceleration_time'] / stats_aggregated_df['deceleration_time']

    return stats_aggregated_df

try:
    stats_aggregated = pd.read_pickle('stats_aggregated.pkl')
except:
    stats_aggregated = summary_stats_aggregated(stats, all_players_list)
    stats_aggregated.to_pickle('stats_aggregated.pkl')

def filtered_df(df, player_list):
    row_list = []
    
    for player in player_list:
        if player in df.index:
            row = df.loc[player]
            row_list.append(row)
            
    lfc_df = pd.DataFrame(row_list)
    return lfc_df
    
lfc_df = filtered_df(stats_aggregated, lfc_list)

def transform_to_points(df, col1, col2):
    point_list = list(zip(df[col1], df[col2]))
    x_list = list(df[col1])
    y_list = list(df[col2])

    return point_list, x_list, y_list

point_list, x_list, y_list = transform_to_points(stats_aggregated, 'distance_covered', 'acc/dec-ratio')

def kmeans_profiling(points, k):
    kmeans = KMeans(n_clusters = k, random_state=0).fit(points)
    cluster_labels = kmeans.labels_

    return cluster_labels

cluster_labels = kmeans_profiling(point_list, 3)

# plot the cluster
fig = plt.figure()
plt.scatter(x_list, y_list, c = cluster_labels)
fig.savefig('clusterplot.png', dpi=100)

# histogram acc/dec ratio
fig, axs = plt.subplots(nrows=2, ncols=1)
axs[0].hist(stats_aggregated['acc/dec-ratio'])
axs[0].axis([0.8, 1.2, 0, 40])
axs[1].hist(lfc_df['acc/dec-ratio'], color='r')
axs[1].axis([0.8, 1.2, 0, 5])
fig.savefig('adr_hist.png', dpi=100)

print(f'Time to run the script = {time.time()- t0}')
