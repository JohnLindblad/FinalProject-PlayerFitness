# import time and set a starting time to be able to measure computation later on
import time
t0 = time.time()

# imports
import pandas as pd
import json
import os
import copy
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

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

    for match in match_ids:

        with open(path + '/matches/' + match + '/match_data.json') as f:
            data = json.load(f)

        name_dict = {}
        lfc_dict = {}

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

    tot_df = pd.DataFrame()

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
                c = len([i for i in col_df if i!= None]) # frames covered

                if c <= drop_limit: # default: 6000 frames = 600 sec = 10 min
                    drop_cols.append(col)

            match_df = match_df.drop(columns = drop_cols)

    tot_df = pd.concat([tot_df, match_df], axis=1)

    return tot_df

tot_df = load_tracking_data(path, all_matches)

dt = 0.1 # manually checked the time interval between frames

# computing distance between frames in x and y dimension
diff_df = tot_df.diff(periods=1, axis=0)

print(f'Time to run the script = {time.time()- t0}')
