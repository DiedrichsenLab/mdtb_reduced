from pathlib import Path

import os
import re
import pandas as pd
import numpy as np
import random
from math import ceil
import cv2
import glob
import shutil

import experiment_code.constants as consts
from experiment_code.targetfile_utils import Utils
# import experiment_code.targetfile_utils  as utils 

# create instances of directories

class VisualSearch(Utils):
    """
    This class makes target files for Visual Search using parameters set in __init__
        Args:
            task_name (str): 'visual_search'
            orientations (int): orientations of target/distractor stims
            balance_blocks (dict): keys are 'condition_name', 'trial_type'
            block_dur_secs (int): length of task_name (sec)
            num_blocks (int): number of blocks to make
            tile_block (int): determines number of repeats for task_name
            trial_dur (int): length of trial (sec)
            iti_dur (iti): length of iti (sec)
            instruct_dur (int): length of instruct for task_names (sec)
            hand (str): response hand
            replace (bool): sample stim with or without replacement
            display_trial_feedback (bool): display trial-by-trial feedback
    """
    
    def __init__(self):
        super().__init__()
        self.task_name = 'visual_search'
        self.orientations = list([90, 180, 270, 360]) # DO NOT CHANGE ORDER
        self.balance_blocks = {'condition_name': {'easy': '4', 'hard': '8'}, 'trial_type': [True, False]}
        self.block_dur_secs = 15
        self.num_blocks = 5
        self.tile_block = 1
        self.trial_dur = 2 
        self.iti_dur = .5
        self.instruct_dur = 5
        self.hand = 'right'
        self.key_hand_dict = 'key_hand_dict_binary_right'
        self.replace = False
        self.display_trial_feedback = True
    
    def _get_block_info(self, **kwargs):
        # length (in secs) of the block
        if kwargs.get('block_dur_secs'):
            self.block_dur_secs = kwargs['block_dur_secs']
        
        # repeat the target files
        if kwargs.get('tile_block'):
            self.tile_block = kwargs['tile_block']

        # num of blocks (i.e. target files) to make
        if kwargs.get('num_blocks'):
            self.num_blocks = kwargs['num_blocks'] * self.tile_block

        # get overall number of trials
        self.num_trials = int(self.block_dur_secs / (self.trial_dur + self.iti_dur))  

        # get `num_stims` - lowest denominator across `balance_blocks`
        denominator = np.prod([len(stim) for stim in [*self.balance_blocks.values()]])
        self.num_stims = ceil(self.num_trials / denominator) # round up to nearest int
    
    def _create_columns(self):

        def _get_condition(x):
            for key in self.balance_blocks['condition_name'].keys():
                cond = self.balance_blocks['condition_name'][key]
                if x==cond:
                    value = key
            return value

        dataframe = pd.DataFrame()
        # make `condition_name` column
        conds = [self.balance_blocks['condition_name'][key] for key in self.balance_blocks['condition_name'].keys()]
        # conds = [self.balance_blocks['condition_name']['easy'], self.balance_blocks['condition_name']['hard']]
        dataframe['stim'] = self.num_trials*conds
        dataframe['condition_name'] = dataframe['stim'].apply(lambda x: _get_condition(x))
        dataframe['stim'] = dataframe['stim'].astype(int)

        # make `trial_type` column
        dataframe['trial_type'] = self.num_trials*self.balance_blocks['trial_type']
        dataframe['trial_type'] = dataframe['trial_type'].sort_values().reset_index(drop=True)

        dataframe['display_trial_feedback'] = self.display_trial_feedback

        return dataframe

    def _balance_design(self, dataframe):
        dataframe =  dataframe.groupby([*self.balance_blocks], as_index=False).apply(lambda x: x.sample(n=self.num_stims, random_state=self.random_state, replace=True)).reset_index(drop=True)

        # ensure that only `num_trials` are sampled
        return dataframe.sample(n=self.num_trials, random_state=self.random_state, replace=False).reset_index(drop=True)

    def _save_visual_display(self, dataframe):
        # add visual display cols
        display_pos, orientations_correct = zip(*[self._make_search_display(cond, self.orientations, trial_type) for (cond, trial_type) in zip(dataframe["stim"], dataframe["trial_type"])])

        data_dicts = []
        for trial_idx, trial_conditions in enumerate(display_pos):
            for condition, point in trial_conditions.items():
                data_dicts.append({'trial': trial_idx, 'stim': condition, 'xpos': point[0], 'ypos': point[1], 'orientation': orientations_correct[trial_idx][condition]})  
        
        # save out to dataframe
        df_display = pd.DataFrame.from_records(data_dicts)

        # save out visual display
        visual_display_name = self._get_visual_display_name()
        df_display.to_csv(os.path.join(self.target_dir, visual_display_name))

    def _get_visual_display_name(self):
        tf_name = f"{self.task_name}_{self.block_dur_secs}sec"
        tf_name = self._get_target_file_name(tf_name)

        str_part = tf_name.partition(self.task_name)
        visual_display_name = 'display_pos' + str_part[2] 

        return visual_display_name
    
    def _make_search_display(self, display_size, orientations, trial_type):
        # make location and orientations lists (for target and distractor items)

        # STIM POSITIONS
        grid_h_dva = 8.4
        grid_v_dva = 11.7

        n_h_items = 6
        n_v_items = 8

        item_h_pos = np.linspace(-grid_h_dva / 2.0, +grid_h_dva/ 2.0, n_h_items)
        item_v_pos = np.linspace(-grid_v_dva / 2.0, +grid_v_dva / 2.0, n_v_items)

        grid_pos = []
        for curr_h_pos in item_h_pos:
            for curr_v_pos in item_v_pos:
                grid_pos.append([curr_h_pos, curr_v_pos])

        locations = random.sample(grid_pos, display_size)

        ## STIM ORIENTATIONS
        orientations_list = orientations*int(display_size/4)
        
        # if trial type is false - randomly replace target stim (90)
        # with a distractor
        if not trial_type:
            orientations_list = [random.sample(orientations[1:],1)[0] if x==90 else x for x in orientations_list]
        
        # if trial is true and larger than 4, leave one target stim (90) in list 
        # and randomly replace the others with distractor stims
        if display_size >4 and trial_type:
            indices = [i for i, x in enumerate(orientations_list) if x == 90]
            indices.pop(0)
            new_num = random.sample(orientations[1:],2) # always assumes that orientations_list is as follows: [90,180,270,360]
            for i, n in zip(*(indices, new_num)): 
                orientations_list[i] = n

        return dict(enumerate(locations)), dict(enumerate(orientations_list))
        
    def make_targetfile(self, **kwargs):
        """
        makes target file(s) for visual search given parameters in __init__
        """
        # get info about block
        self._get_block_info(**kwargs)

        seeds = np.arange(self.num_blocks)+1

        for self.block in np.arange(self.num_blocks):

            # randomly sample so that conditions (2Back- and 2Back+) are equally represented
            self.random_state = seeds[self.block]

            # create the dataframe
            df_target = self._create_columns()

            # balance the dataframe
            df_target = self._balance_design(dataframe = df_target)

            self.target_dir = os.path.join(consts.target_dir, self.task_name)

            # save visual display dataframe
            self._save_visual_display(dataframe = df_target)

            # save target file
            self._save_target_files(df_target)

class NBack(Utils):
    """
    This class makes target files for N Back using parameters set in __init__
        Args:
            task_name (str): 'n_back'
            n_back (int): default is 2
            balance_blocks (dict): keys are 'condition_name'
            block_dur_secs (int): length of task_name (sec)
            num_blocks (int): number of blocks to make
            tile_block (int): determines number of repeats for task_name
            trial_dur (int): length of trial (sec)
            iti_dur (iti): length of iti (sec)
            instruct_dur (int): length of instruct for task_names (sec)
            hand (str): response hand
            replace (bool): sample stim with or without replacement
            display_trial_feedback (bool): display trial-by-trial feedback
    """
    
    def __init__(self):
        super().__init__()
        self.task_name = 'n_back'
        self.n_back = 2
        self.balance_blocks = {'condition_name': {'easy': '2_back-', 'hard': '2_back+'}}
        self.block_dur_secs = 15
        self.num_blocks = 5
        self.tile_block = 1
        self.trial_dur = 1.5 
        self.iti_dur = .5
        self.instruct_dur = 5
        self.hand = 'left'
        self.key_hand_dict = 'key_hand_dict_binary_left'
        self.replace = False
        self.display_trial_feedback = True
    
    def _get_block_info(self, **kwargs):
        # length (in secs) of the block
        if kwargs.get('block_dur_secs'):
            self.block_dur_secs = kwargs['block_dur_secs']
        
        # repeat the target files
        if kwargs.get('tile_block'):
            self.tile_block = kwargs['tile_block']

        # num of blocks (i.e. target files) to make
        if kwargs.get('num_blocks'):
            self.num_blocks = kwargs['num_blocks'] * self.tile_block

        # get overall number of trials
        self.num_trials = int(self.block_dur_secs / (self.trial_dur + self.iti_dur))  

        # get `num_stims` - lowest denominator across `balance_blocks`
        denominator = np.prod([len(stim) for stim in [*self.balance_blocks.values()]])
        self.num_stims = ceil(self.num_trials / denominator) # round up to nearest int
       
    def _create_columns(self):

        def _get_condition(x):
            for key in self.balance_blocks['condition_name'].keys():
                cond = self.balance_blocks['condition_name'][key]
                if x==cond:
                    value = key
            return value

        # make trial_type column
        dataframe = pd.DataFrame()
        dataframe['trial_type'] = self.num_stims*(True, False)
        dataframe = dataframe.sample(n=self.num_trials, random_state=self.random_state, replace=False).reset_index(drop=True) 
        dataframe['trial_type'][:self.n_back] = False # first n+cond_type trials (depending on cond_type) have to be False

        # make `n_back` and `condition_name` cols
        conds = [self.balance_blocks['condition_name'][key] for key in self.balance_blocks['condition_name'].keys()]
        dataframe['n_back'] = np.where(dataframe["trial_type"]==False, conds[0], conds[1])
        dataframe['condition_name'] = dataframe['n_back'].apply(lambda x: _get_condition(x))

        dataframe['display_trial_feedback'] = self.display_trial_feedback

        return dataframe
    
    def _balance_design(self, dataframe):
        # load in stimuli
        stim_files = [f for f in os.listdir(str(consts.stim_dir / self.task_name)) if f.endswith('g')]

        # first two images are always random (and false)
        # all other images are either match or not a match
        random.seed(self.random_state)
        stim_list = random.sample(stim_files, k=self.n_back)
        for t in dataframe['trial_type'][self.n_back:]: # loop over n+self.n_back
            match_img = stim_list[-self.n_back]
            no_match_imgs = [stim for stim in stim_files if stim != match_img] # was match_img[0]
            if t == False: # not a match
                random.seed(self.random_state)
                stim_list.append(random.sample(no_match_imgs, k=self.n_back-1))
            else:      # match
                stim_list.append(match_img)

        dataframe["stim"] = [''.join(x) for x in stim_list]

        return dataframe

    def make_targetfile(self, **kwargs):
        """
        makes target file(s) for n back given parameters in __init__

        """
        # get info about block
        self._get_block_info(**kwargs)

        seeds = np.arange(self.num_blocks)+1

        for self.block in np.arange(self.num_blocks):

            # randomly sample so that conditions (2Back- and 2Back+) are equally represented
            self.random_state = seeds[self.block]

            # create the dataframe
            df_target = self._create_columns()

            # balance the dataframe
            df_target = self._balance_design(dataframe = df_target)

            self.target_dir = os.path.join(consts.target_dir, self.task_name)
            self._save_target_files(df_target)

class SocialPrediction(Utils):
    """
    This class makes target files for Social Prediction using parameters set in __init__
        Args:
            task_name (str): 'social_prediction'
            dataset_name (str): 'homevideos' is the default
            logging_file (str): csv file containing info about stimuli
            video_name (list of str): name of video(s) to include
            resized (bool): resize frames of video
            balance_blocks (dict): keys are 'actors', 'condition_name', 'label'
            block_dur_secs (int): length of task_name (sec)
            num_blocks (int): number of blocks to make
            tile_block (int): determines number of repeats for task_name
            trial_dur (int): length of trial (sec)
            iti_dur (iti): length of iti (sec)
            instruct_dur (int): length of instruct for task_names (sec)
            hand (str): response hand
            replace (bool): sample stim with or without replacement
            display_trial_feedback (bool): display trial-by-trial feedback
    """
    
    def __init__(self):
        super().__init__()
        self.task_name = 'social_prediction'
        self.dataset_name = 'homevideos'
        self.logging_file = 'homevideos_annotations_logging.csv'
        self.video_name = ['dynamic_0ms', 'dynamic_100ms']
        self.resized = True
        self.balance_blocks = {'actors': ['SB', 'MK'], 
                'condition_name': {'dynamic_0ms': 'easy', 'dynamic_100ms': 'hard'},
                'label': ['hug', 'handShake']}
        self.block_dur_secs = 15
        self.num_blocks = 5
        self.tile_block = 1
        self.trial_dur = 2.5 
        self.iti_dur = .5
        self.instruct_dur = 5
        self.hand = 'right'
        self.key_hand_dict = 'key_hand_dict_binary_right'
        self.replace = False 
        self.display_trial_feedback = True

    def _filter_dataframe(self, dataframe):
        # remove all filenames where any of the videos have not been extracted
        stims_to_remove = dataframe.query('extracted==False')["video_name"].to_list()
        df_filtered = dataframe[~dataframe["video_name"].isin(stims_to_remove)]

        # query rows with relevant videos and relevant labels
        label = self.balance_blocks['label']
        actors = self.balance_blocks['actors']
        df_filtered = df_filtered.query(f'condition_name=={self.video_name} and label=={label} and actors=={actors}')

        return df_filtered

    def _create_new_columns(self, dataframe):
        # make new `stim`
        if self.resized:
            dataframe['stim'] = dataframe['video_name'] + '_' + dataframe['condition_name'] + '_resized' + '.mp4'
        else:
            dataframe['stim'] = dataframe['video_name'] + '_' + dataframe['condition_name'] + '.mp4'

        # set `condition name`
        dataframe['condition_name'] = dataframe['condition_name'].apply(lambda x: self.balance_blocks['condition_name'][x])

        # assign dataset name
        dataframe['dataset'] = self.dataset_name

        # assign trial type (only works currently for two labels)
        labels = self.balance_blocks['label']
        if len(labels)==2:
            dataframe['trial_type'] = dataframe['label'].apply(lambda x: True if x==labels[0] else False)
        else:
            print(f'there are an incorrect number of labels, there should be two')

        dataframe['display_trial_feedback'] = self.display_trial_feedback

        return dataframe
    
    def _balance_design(self, dataframe):
        # group the dataframe according to `balance_blocks`
        dataframe = dataframe.groupby([*self.balance_blocks], as_index=False).apply(lambda x: x.sample(n=self.num_stims, random_state=self.random_state, replace=self.replace)).reset_index(drop=True)

        # ensure that only `num_trials` are sampled
        dataframe = dataframe.sample(n=self.num_trials, random_state=self.random_state, replace=False).reset_index(drop=True)

        return dataframe
     
    def _get_block_info(self, **kwargs):
        # length (in secs) of the block
        if kwargs.get('block_dur_secs'):
            self.block_dur_secs = kwargs['block_dur_secs']
        
        # repeat the target files
        if kwargs.get('tile_block'):
            self.tile_block = kwargs['tile_block']

        # num of blocks (i.e. target files) to make
        if kwargs.get('num_blocks'):
            self.num_blocks = kwargs['num_blocks'] * self.tile_block

        # get overall number of trials
        self.num_trials = int(self.block_dur_secs / (self.trial_dur + self.iti_dur))  

        # get `num_stims` - lowest denominator across `balance_blocks`
        denominator = np.prod([len(stim) for stim in [*self.balance_blocks.values()]])
        self.num_stims = ceil(self.num_trials / denominator) # round up to nearest int
    
    def make_targetfile(self, **kwargs):
        """
        makes target file(s) for social prediction given parameters in __init__

        """
        # get info about block
        self._get_block_info(**kwargs)

        # return logging file
        fpath = os.path.join(consts.stim_dir, self.task_name, self.logging_file)
        
        # read in stimulus dataframe
        df = pd.read_csv(fpath)

        # filter dataframe
        df_filtered = self._filter_dataframe(dataframe = df)

        # create new columns (`trial_type` etc)
        df_filtered = self._create_new_columns(dataframe = df_filtered)

        seeds = np.arange(self.num_blocks)+1

        # for self.block, self.key in enumerate(self.block_design):
        for self.block in np.arange(self.num_blocks):
            # randomly sample so that conditions (easy and hard) are equally represented
            self.random_state = seeds[self.block]

            # balance the dataframe by `condition_name` and `player_num`
            df_target = self._balance_design(dataframe = df_filtered)

            # remove `df_target` rows from the main dataframe so that we're always sampling from unique rows
            if self.replace==False:
                df_filtered = df_filtered.merge(df_target, how='left', indicator=True)
                df_filtered = df_filtered[df_filtered['_merge'] == 'left_only'].drop('_merge', axis=1)
            
            self.target_dir = os.path.join(consts.target_dir, self.task_name)
            self._save_target_files(df_target)

class SemanticPrediction(Utils):
    """
    This class makes target files for Semantic Prediction using parameters set in __init__
        Args:
            task_name (str): 'semantic_prediction'
            logging_file (str): csv file containing info about stimuli
            stem_word_dur (int): length of stem word (sec)
            last_word_dur (int): length of last word (sec)
            frac (int): proportion of meaningless trials. default is .3.
            balance_blocks (dict): keys are 'CoRT_descript', 'condition_name'
            block_dur_secs (int): length of task_name (sec)
            num_blocks (int): number of blocks to make
            tile_block (int): determines number of repeats for task_name
            trial_dur (int): length of trial (sec)
            iti_dur (iti): length of iti (sec)
            instruct_dur (int): length of instruct for task_names (sec)
            hand (str): response hand
            replace (bool): sample stim with or without replacement
            display_trial_feedback (bool): display trial-by-trial feedback
    """
    
    def __init__(self):
        super().__init__()
        self.task_name = 'semantic_prediction'
        self.logging_file = 'sentence_validation.csv'
        self.stem_word_dur = 0.5
        self.last_word_dur = 1.5
        self.frac = .3
        self.balance_blocks = {'CoRT_descript': ['strong non-CoRT', 'strong CoRT'],
                    'condition_name': {'high cloze': 'easy', 'low cloze': 'hard'}}
        self.block_dur_secs = 15
        self.num_blocks = 5
        self.tile_block = 1
        self.trial_dur = 7
        self.iti_dur = .5
        self.instruct_dur = 5
        self.hand = 'right'
        self.key_hand_dict = 'key_hand_dict_binary_right'
        self.replace = False
        self.display_trial_feedback = True

    def _filter_dataframe(self, dataframe):
        # conds = [self.balance_blocks['condition_name'][key] for key in self.balance_blocks['condition_name'].keys()]  
        conds = list(self.balance_blocks['condition_name'].keys()) 
        dataframe = dataframe.query(f'CoRT_descript=={self.balance_blocks["CoRT_descript"]} and cloze_descript=={conds}')

        # strip erroneous characters from sentences
        dataframe['stim'] = dataframe['full_sentence'].str.replace('|', ' ')
        
        return dataframe

    def _create_new_columns(self, dataframe):
        # add condition column
        dataframe['condition_name'] = dataframe['cloze_descript'].apply(lambda x: self.balance_blocks['condition_name'][x])
        dataframe['stem_word_dur'] = self.stem_word_dur
        dataframe['last_word_dur'] = self.last_word_dur
        dataframe['trial_dur_correct'] = (dataframe['word_count'] * dataframe['stem_word_dur']) + self.iti_dur + dataframe['last_word_dur']
        dataframe['display_trial_feedback'] = self.display_trial_feedback
        dataframe.drop({'full_sentence'}, inplace=True, axis=1)

        return dataframe

    def _add_random_word(self, dataframe, columns):
        """ sample `frac_random` and add to `full_sentence`
            Args: 
                dataframe (pandas dataframe): dataframe
            Returns: 
                dataframe with modified `full_sentence` col
        """
        idx = dataframe.groupby(columns).apply(lambda x: x.sample(frac=self.frac, replace=False, random_state=self.random_state)).index

        sampidx = idx.get_level_values(len(columns)) # get third level
        dataframe["trial_type"] = ~dataframe.index.isin(sampidx)
        dataframe["last_word"] = dataframe.apply(lambda x: x["random_word"] if not x["trial_type"] else x["target_word"], axis=1)

        return dataframe

    def _balance_design(self, dataframe):
        # group the dataframe according to `balance_blocks`
        dataframe = dataframe.groupby([*self.balance_blocks], as_index=False).apply(lambda x: x.sample(n=self.num_stims, random_state=self.random_state, replace=self.replace)).reset_index(drop=True)

        # ensure that only `num_trials` are sampled
        dataframe = dataframe.sample(n=self.num_trials, random_state=self.random_state, replace=False).reset_index(drop=True)

        return dataframe
    
    def _get_block_info(self, **kwargs):
        # length (in secs) of the block
        if kwargs.get('block_dur_secs'):
            self.block_dur_secs = kwargs['block_dur_secs']
        
        # repeat the target files
        if kwargs.get('tile_block'):
            self.tile_block = kwargs['tile_block']

        # num of blocks (i.e. target files) to make
        if kwargs.get('num_blocks'):
            self.num_blocks = kwargs['num_blocks'] * self.tile_block

        # get overall number of trials
        self.num_trials = int(self.block_dur_secs / (self.trial_dur + self.iti_dur))  

        # get `num_stims` - lowest denominator across `balance_blocks`
        denominator = np.prod([len(stim) for stim in [*self.balance_blocks.values()]])
        self.num_stims = ceil(self.num_trials / denominator) # round up to nearest int
    
    def make_targetfile(self, **kwargs):
        """
        makes target file(s) for semantic prediction given parameters in __init__
        """
        # get info about block
        self._get_block_info(**kwargs)

        # return logging file
        fpath = os.path.join(consts.stim_dir, self.task_name, self.logging_file)
        
        # read in stimulus dataframe
        df = pd.read_csv(fpath)

        # filter dataframe
        df_filtered = self._filter_dataframe(dataframe = df)

        # create new columns (`condition_name` etc)
        df_filtered = self._create_new_columns(dataframe = df_filtered)

        seeds = np.arange(self.num_blocks)+1

        # for self.block, self.key in enumerate(self.block_design):
        for self.block in np.arange(self.num_blocks):
            # randomly sample so that conditions (easy and hard) are equally represented
            self.random_state = seeds[self.block]

            # balance the dataframe by `condition_name` and `player_num`
            df_target = self._balance_design(dataframe = df_filtered)

            # remove `df_target` rows from the main dataframe so that we're always sampling from unique rows
            if self.replace==False:
                df_filtered = df_filtered.merge(df_target, how='left', indicator=True)
                df_filtered = df_filtered[df_filtered['_merge'] == 'left_only'].drop('_merge', axis=1)

            # add random word based on `self.frac`
            df_target = self._add_random_word(dataframe=df_target, columns=['condition_name']) # 'CoRT_descript'
            
            # save out target files
            self.target_dir = os.path.join(consts.target_dir, self.task_name)
            self._save_target_files(df_target)

class ActionObservation(Utils):
    """
    This class makes target files for Action Observation using parameters set in __init__
        Args:
            task_name (str): 'rest'
            logging_file (str): csv file containing info about stimuli
            video_name (list of str): name of video(s) to include
            manipulation (str): 'left_right' or 'miss_goal'
            resized (bool): resize frames of video
            balance_blocks (dict): keys are 'player_name', 'condition_name', 'trial_type'
            block_dur_secs (int): length of task_name (sec)
            num_blocks (int): number of blocks to make
            tile_block (int): determines number of repeats for task_name
            trial_dur (int): length of trial (sec)
            iti_dur (iti): length of iti (sec)
            instruct_dur (int): length of instruct for task_names (sec)
            hand (str): response hand
            replace (bool): sample stim with or without replacement
            display_trial_feedback (bool): display trial-by-trial feedback
    """
    
    def __init__(self):
        super().__init__()
        self.task_name = "action_observation"
        self.logging_file = 'all_clips_annotation_logging.csv'
        self.video_name = ['dynamic_120ms']
        self.manipulation = 'left_right'
        self.resized = True
        self.balance_blocks = {'player_name': ['DC', 'EW'], 'condition_name': ['easy', 'hard'], 'trial_type': ['left', 'right']}
        self.block_dur_secs = 15
        self.num_blocks = 5
        self.tile_block = 1
        self.trial_dur = 2 
        self.iti_dur = .5
        self.instruct_dur = 5
        self.hand = 'left'
        self.key_hand_dict = 'key_hand_dict_binary_left'
        self.replace = True # sample with or without replacement
        self.display_trial_feedback = True

    def _filter_dataframe(self, dataframe):

        def _get_player(x):
            if x.find('DC')>=0:
                player_name = 'DC'
            elif x.find('FI')>=0:
                player_name = 'FI'
            elif x.find('EW')>=0:
                player_name = 'EW'
            else:
                print('player does not exist')
            return player_name

        # remove all filenames where any of the videos have not been extracted
        # and where the player did not accurately hit the target (success=F)
        stims_to_remove = dataframe.query('extracted==False or player_success=="?"')["video_name"].to_list()
        df_filtered = dataframe[~dataframe["video_name"].isin(stims_to_remove)]

        # remove rows without video info
        df_filtered = df_filtered.query(f'condition_name=={self.video_name}')

        # create `player_name`
        df_filtered['player_name'] = df_filtered['video_name'].apply(lambda x: _get_player(x))

        # filter `player_name`
        cond = self.balance_blocks['player_name']
        df_filtered = df_filtered.query(f'player_name=={cond}')

        # figure out the actual hits. certain trials (~14%) were misses. enter the actual hit.
        df_filtered.loc[df_filtered['hit_target'].isnull(), 'hit_target'] = df_filtered['instructed_target']

        return df_filtered
    
    def _create_new_columns(self, dataframe):

        def _get_condition(x):
            if self.manipulation=="miss_goal":
                easy = [1,2,7,8,9,10,15,16]
                hard = [3,4,5,6,11,12,13,14]
            elif self.manipulation=="left_right":
                easy = [1,2,3,4,13,14,15,16]
                hard = [5,6,7,8,9,10,11,12]
            else:
                print('manipulation does not exist')

            if x in easy:
                condition = "easy"
            elif x in hard:
                condition = "hard"
            else:
                condition = float("NaN")
                print(f'{x} not in list')

            return condition
        
        def _get_trial_type(x):
            if self.manipulation=="miss_goal":
                list1= [5,6,7,8,9,10,11,12]
                list2 = [1,2,3,4,13,14,15,16]
                value1 = "goal"
                value2 = "miss"
            elif self.manipulation=="left_right":
                list1 = [1,2,3,4,5,6,7,8]
                list2 = [9,10,11,12,13,14,15,16]
                value1 = True # 'right'
                value2 = False # 'left'
            else:
                print('manipulation does not exist')

            if x in list1:
                trial = value1
            elif x in list2:
                trial = value2
            else:
                trial = float("NaN")
                print(f'{x} not in list')

            return trial

        # make new image column
        if self.resized:
            dataframe['stim'] = dataframe['video_name'] + '_' + dataframe['condition_name'] + '_resized' + '.mp4'
        else:
            dataframe['stim'] = dataframe['video_name'] + '_' + dataframe['condition_name'] + '.mp4'

        # divide targets between easy and hard
        dataframe['condition_name'] = dataframe['hit_target'].apply(lambda x: _get_condition(x))

        # either miss_goal or left_right based on manipulation
        dataframe['trial_type'] = dataframe['hit_target'].apply(lambda x: _get_trial_type(x))

        # get time of extraction for video (round to two decimals)
        dataframe['video_start_time'] = np.round(dataframe['interact_start'] - dataframe['secs_before_interact'], 2)
        dataframe['video_end_time'] = np.round(dataframe['interact_start'] + dataframe['secs_after_interact'], 2)

        dataframe['display_trial_feedback'] = self.display_trial_feedback

        return dataframe  
    
    def _balance_design(self, dataframe):
        # group the dataframe according to `balance_blocks`
        dataframe = dataframe.groupby([*self.balance_blocks], as_index=False).apply(lambda x: x.sample(n=self.num_stims, random_state=self.random_state, replace=self.replace)).reset_index(drop=True)

        # ensure that only `num_trials` are sampled
        dataframe = dataframe.sample(n=self.num_trials, random_state=self.random_state, replace=False).reset_index(drop=True)

        return dataframe
    
    def _get_block_info(self, **kwargs):
        # length (in secs) of the block
        if kwargs.get('block_dur_secs'):
            self.block_dur_secs = kwargs['block_dur_secs']
        
        # repeat the target files
        if kwargs.get('tile_block'):
            self.tile_block = kwargs['tile_block']

        # num of blocks (i.e. target files) to make
        if kwargs.get('num_blocks'):
            self.num_blocks = kwargs['num_blocks'] * self.tile_block

        # get overall number of trials
        self.num_trials = int(self.block_dur_secs / (self.trial_dur + self.iti_dur))  

        # get `num_stims` - lowest denominator across `balance_blocks`
        denominator = np.prod([len(stim) for stim in [*self.balance_blocks.values()]])
        self.num_stims = ceil(self.num_trials / denominator) # round up to nearest int

    def make_targetfile(self, **kwargs):
        """
        makes target file(s) for action observation given parameters in __init__
        """
        # get info about block
        self._get_block_info(**kwargs)

        # return logging file
        fpath = os.path.join(consts.stim_dir, self.task_name, self.logging_file)
        
        # read in stimulus dataframe
        df = pd.read_csv(fpath)

        # filter dataframe
        df_filtered = self._filter_dataframe(dataframe = df)

        # create new columns (`trial_type` etc)
        df_filtered = self._create_new_columns(dataframe = df_filtered)

        seeds = np.arange(self.num_blocks)+1

        # for self.block, self.key in enumerate(self.block_design):
        for self.block in np.arange(self.num_blocks):
            # randomly sample so that conditions (easy and hard) are equally represented
            self.random_state = seeds[self.block]

            # balance the dataframe by `condition_name` and `player_num`
            df_target = self._balance_design(dataframe = df_filtered)

            # remove `df_target` rows from the main dataframe so that we're always sampling from unique rows
            if self.replace==False:
                df_filtered = df_filtered.merge(df_target, how='left', indicator=True)
                df_filtered = df_filtered[df_filtered['_merge'] == 'left_only'].drop('_merge', axis=1)
            
            self.target_dir = os.path.join(consts.target_dir, self.task_name)
            self._save_target_files(df_target)

class TheoryOfMind(Utils):
    """
    This class makes target files for Theory of Mind using parameters set in __init__
        Args:
            task_name (str): 'theory_of_mind'
            logging_file (str): csv file containing info about stimuli
            story_dur (int): length of story (sec)
            question_dur (int): length of question (sec)
            frac (int): proportion of meaningless trials. default is .3.
            balance_blocks (dict): keys are 'condition_name'
            block_dur_secs (int): length of task_name (sec)
            num_blocks (int): number of blocks to make
            tile_block (int): determines number of repeats for task_name
            trial_dur (int): length of trial (sec)
            iti_dur (iti): length of iti (sec)
            instruct_dur (int): length of instruct for task_names (sec)
            hand (str): response hand
            replace (bool): sample stim with or without replacement
            display_trial_feedback (bool): display trial-by-trial feedback
    """
    
    def __init__(self):
        super().__init__()
        self.task_name = 'theory_of_mind'
        self.logging_file = 'theory_of_mind.csv'
        self.story_dur = 10
        self.question_dur = 4
        self.frac = .3
        self.balance_blocks = {'condition_name': ['belief','photo'],'trial_type': [True, False]}
        self.block_dur_secs = 15
        self.num_blocks = 5
        self.tile_block = 1
        self.trial_dur = 14
        self.iti_dur = .5
        self.instruct_dur = 5
        self.hand = 'left'
        self.key_hand_dict = 'key_hand_dict_binary_left'
        self.replace = False
        self.display_trial_feedback = True

    def _filter_dataframe(self, dataframe):
        dataframe = dataframe.query(f'condition=={self.balance_blocks["condition_name"]} and response=={self.balance_blocks["trial_type"]}')
        
        return dataframe

    def _create_new_columns(self, dataframe):
        # add condition column
        # dataframe['condition_name'] = dataframe['condition'].apply(lambda x: self.balance_blocks['condition_name'][x])
        dataframe['condition_name'] = dataframe['condition']
        dataframe['story_dur'] = self.story_dur
        dataframe['question_dur'] = self.question_dur
        dataframe['trial_dur_correct'] = dataframe['story_dur'] + self.iti_dur + dataframe['question_dur']
        dataframe['display_trial_feedback'] = self.display_trial_feedback
        
        responses = self.balance_blocks['trial_type']
        dataframe['trial_type'] = dataframe['response'].apply(lambda x: True if x==responses[0] else False)

        return dataframe

    def _balance_design(self, dataframe):
        # group the dataframe according to `balance_blocks`
        dataframe = dataframe.groupby([*self.balance_blocks], as_index=False).apply(lambda x: x.sample(n=self.num_stims, random_state=self.random_state, replace=self.replace)).reset_index(drop=True)

        # ensure that only `num_trials` are sampled
        dataframe = dataframe.sample(n=self.num_trials, random_state=self.random_state, replace=False).reset_index(drop=True)

        return dataframe
    
    def _get_block_info(self, **kwargs):
        # length (in secs) of the block
        if kwargs.get('block_dur_secs'):
            self.block_dur_secs = kwargs['block_dur_secs']
        
        # repeat the target files
        if kwargs.get('tile_block'):
            self.tile_block = kwargs['tile_block']

        # num of blocks (i.e. target files) to make
        if kwargs.get('num_blocks'):
            self.num_blocks = kwargs['num_blocks'] * self.tile_block

        # get overall number of trials
        self.num_trials = int(self.block_dur_secs / (self.trial_dur + self.iti_dur))  

        # get `num_stims` - lowest denominator across `balance_blocks`
        denominator = np.prod([len(stim) for stim in [*self.balance_blocks.values()]])
        self.num_stims = ceil(self.num_trials / denominator) # round up to nearest int
    
    def make_targetfile(self, **kwargs):
        """
        makes target file(s) for theory of mind given parameters in __init__
        """
        # get info about block
        self._get_block_info(**kwargs)

        # return logging file
        fpath = os.path.join(consts.stim_dir, self.task_name, self.logging_file)
        
        # read in stimulus dataframe
        df = pd.read_csv(fpath)

        # filter dataframe
        df_filtered = self._filter_dataframe(dataframe = df)

        # create new columns (`condition_name` etc)
        df_filtered = self._create_new_columns(dataframe = df_filtered)

        seeds = np.arange(self.num_blocks)+1

        # for self.block, self.key in enumerate(self.block_design):
        for self.block in np.arange(self.num_blocks):
            # randomly sample so that conditions (easy and hard) are equally represented
            self.random_state = seeds[self.block]

            # balance the dataframe by `condition_name` and `player_num`
            df_target = self._balance_design(dataframe = df_filtered)

            # remove `df_target` rows from the main dataframe so that we're always sampling from unique rows
            if self.replace==False:
                df_filtered = df_filtered.merge(df_target, how='left', indicator=True)
                df_filtered = df_filtered[df_filtered['_merge'] == 'left_only'].drop('_merge', axis=1)
            
            # save out target files
            self.target_dir = os.path.join(consts.target_dir, self.task_name)
            self._save_target_files(df_target)

class Rest(Utils):
    """
    This class makes target files for Rest using parameters set in __init__
        Args:
            task_name (str): 'rest'
            rest_dur_secs (int): length of rest (sec), 0 if no rest
            iti_dur (iti): length of iti (sec)
            instruct_dur (int): length of instruct for task_names (sec)
            hand (str): response hand
            num_trials (int): number of trials per block. default is 1
            display_trial_feedback (bool): display trial-by-trial feedback
    """
    def __init__(self):
        super().__init__()
        self.task_name = 'rest'
        self.rest_dur_secs = 10
        self.iti_dur = 0
        self.instruct_dur = 0
        self.hand = "None"
        self.num_trials = 1
        self.display_trial_feedback = False

    def _get_block_info(self, **kwargs):
        # length (in secs) of the block
        if kwargs.get('rest_dur_secs'):
            self.trial_dur = kwargs['rest_dur_secs']
        else:
            self.trial_dur = self.rest_dur_secs

    def _create_new_columns(self):
        start_time = np.round(np.arange(0, self.num_trials*(self.trial_dur+self.iti_dur), self.trial_dur+self.iti_dur), 1)
        data = {"stim": 'fixation', "trial_dur":self.trial_dur, "iti_dur":self.iti_dur, "start_time":start_time, "hand": self.hand}

        dataframe = pd.DataFrame.from_records(data)

        dataframe['display_trial_feedback'] = self.display_trial_feedback

        return dataframe
    
    def make_targetfile(self, **kwargs):
        """
        makes target file(s) for rest given parameters in __init__
        """
        # get info about block
        self._get_block_info(**kwargs)
        

        # save target file
        self.target_name = self.task_name + '_' + str(self.rest_dur_secs) + 'sec.csv'

        # create dataframe
        dataframe = self._create_new_columns()

        target_dir = os.path.join(consts.target_dir, self.task_name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        dataframe.to_csv(os.path.join(target_dir, self.target_name), index = False, header = True)

        return self.target_name

class MakeFiles:
    """
    This class makes run and target files using parameters set in __init__
        Args:
            task_names (list of str): options are 'visual_search', 'n_back', 'social_prediction', 'semantic_prediction', 'action_observation'
            feedback_types (list of str): options are 'rt' and 'acc'
            run_name_prefix (str): prefix of run name
            tile_run (int): determines number of block repeats within a run
            instruct_dur (int): length of instruct for task_names (sec)
            block_dur_secs (int): length of task_name (sec)
            rest_dur_secs (int): length of rest (sec), 0 if no rest
            num_runs (int): number of runs
            counterbalance_runs (bool): counterbalance block order across runs
    """
    def __init__(self):
        self.task_names = ['visual_search', 'theory_of_mind', 'n_back', 'social_prediction', 'semantic_prediction', 'action_observation']
        self.feedback_types = ['rt', 'acc', 'rt', 'acc', 'rt', 'acc']
        #self.task_names = ['visual_search','theory_of_mind']
        #self.feedback_types = ['rt','acc']
        self.run_name_prefix = 'run'
        self.tile_run = 1
        self.instruct_dur = 5
        self.block_dur_secs = 30
        self.rest_dur_secs = 10
        self.num_runs = 5
        self.counterbalance_runs = True
    
    def _create_run_dataframe(self, target_files):
        for iter, target_file in enumerate(target_files):
            
            # load target file
            dataframe = pd.read_csv(target_file)

            start_time = dataframe.iloc[0]['start_time'] + self.cum_time 
            end_time = dataframe.iloc[-1]['start_time'] + dataframe.iloc[-1]['trial_dur'] + self.instruct_dur + self.cum_time

            target_file_name = Path(target_file).name
            num_sec = re.findall(r'\d+(?=sec)', target_file)[0]
            target_num = re.findall(r'\d+(?=.csv)', target_file)[0]
            num_trials = len(dataframe)

            data = {'task_name': self.task_name, 'task_num': self.task_num+1, # 'block_iter': iter+1
                    'num_trials': num_trials, 'target_num': target_num, 'num_sec': num_sec,
                    'target_file': target_file_name, 'start_time': start_time, 'end_time': end_time,
                    'instruct_dur': self.instruct_dur, 'feedback_type': self.feedback_type}

            self.all_data.append(data)
            self.cum_time = end_time
    
    def _save_run_file(self, run_name):
        # make dataframe from a dictionary
        df_run = pd.DataFrame.from_dict(self.all_data)

        # save out to file
        df_run.to_csv(os.path.join(consts.run_dir, run_name), index=False, header=True)
    
    def _add_rest(self):
        run_files = sorted(glob.glob(os.path.join(consts.run_dir, f'*{self.run_name_prefix}*')))

        # make target file
        BlockClass = TASK_MAP['rest']
        block = BlockClass()
        self.target_name = block.make_targetfile(block_dur_secs = self.rest_dur_secs)

        for run_file in run_files:
            dataframe = pd.read_csv(run_file)

            dataframe = self._add_rest_rows(dataframe)

            dataframe.to_csv(run_file, index = False, header = True)
    
    def _counterbalance_runs(self):
        pass
    
    def _check_task_run(self):
        # check if task exists in dict
        exists_in_dict = [True for key in self.target_dict.keys() if self.task_name==key]
        if not exists_in_dict: 
            self.target_dict.update({self.task_name: self.fpaths})

        # create run dataframe
        random.seed(2)
        target_files_sample = [self.target_dict[self.task_name].pop(random.randrange(len(self.target_dict[self.task_name]))) for _ in np.arange(self.tile_run)]

        return target_files_sample
   
    def _insert_row(self, row_number, dataframe, row_value): 
        # Slice the upper half of the dataframe 
        df1 = dataframe[0:row_number] 
    
        # Store the result of lower half of the dataframe 
        df2 = dataframe[row_number:] 
    
        # Insert the row in the upper half dataframe 
        df1.loc[row_number]=row_value 
    
        # Concat the two dataframes 
        df_result = pd.concat([df1, df2]) 
    
        # Reassign the index labels 
        df_result.index = [*range(df_result.shape[0])] 
    
        # Return the updated dataframe 
        return df_result 
    
    def _correct_start_end_times(self, dataframe):

        timestamps = (np.cumsum(dataframe['num_sec'] + dataframe['instruct_dur'])).to_list()

        dataframe['end_time'] = timestamps

        timestamps.insert(0, 0) 
        dataframe['start_time'] = timestamps[:-1]

        return dataframe 
    
    def _add_rest_rows(self, dataframe):

        self.num_rest = (len(self.task_names) * self.tile_run) - 1

        trials_before_rest = np.tile(np.round((len(dataframe) + self.num_rest) /(self.num_rest)), self.num_rest)
        rest = np.cumsum(trials_before_rest).astype(int) - 1

        # row values
        row_dict = {'task_name': 'rest', 'task_num': len(self.task_names) + 1,
                     'num_trials': 1, 'target_num': None, 'num_sec': self.rest_dur_secs,
                    'target_file': self.target_name, 'start_time': None, 'end_time': None,
                    'instruct_dur': 0, 'feedback_type': None}

        # Let's create a row which we want to insert 
        for row_number in rest:
            # row_value = np.tile('rest', len(dataframe.columns))
            row_value = list(row_dict.values())
            if row_number > dataframe.index.max()+1: 
                print("Invalid row_number") 
            else: 
                dataframe = self._insert_row(row_number, dataframe, row_value)

        # update start and end times
        dataframe = self._correct_start_end_times(dataframe)

        return dataframe
    
    def make_targetfiles(self):
        for self.task_name in self.task_names:

            # delete any target files that exist in the folder
            files = glob.glob(os.path.join(consts.target_dir, self.task_name, '*'))
            for f in files:
                os.remove(f)

            # make target files
            BlockClass = TASK_MAP[self.task_name]
            block = BlockClass()
            block.make_targetfile(block_dur_secs = self.block_dur_secs, 
                                num_blocks = self.num_runs,
                                tile_block = self.tile_run)
    
    def make_runfiles(self):
        # delete any run files that exist in the folder
        files = glob.glob(os.path.join(consts.run_dir, '*'))
        for f in files:
            os.remove(f)

        # create run files
        self.target_dict = {}
        for run in np.arange(self.num_runs):
            self.cum_time = 0.0
            self.all_data = []

            for self.task_num, self.task_name in enumerate(self.task_names):

                # get target files for `task_name`
                self.target_dir = os.path.join(consts.target_dir, self.task_name)
                self.fpaths = sorted(glob.glob(os.path.join(self.target_dir, f'*{self.task_name}*')))

                # get feedback type
                self.feedback_type = self.feedback_types[self.task_num]

                # sample tasks
                target_files_sample = self._check_task_run()

                # create run dataframe
                self._create_run_dataframe(target_files = target_files_sample)

            # save run file
            run_name = self.run_name_prefix + '_' +  f'{run+1:02d}' + '.csv'
            self._save_run_file(run_name = run_name)
            print(f'saving out {run_name}')

        # OPTION TO COUNTERBALANCE RUNS
        if self.counterbalance_runs:
            self._counterbalance_runs()

        # OPTION TO ADD REST TO RUNS
        if self.rest_dur_secs > 0:
            self._add_rest()
    
    def make_all(self):
        # create target files
        self.make_targetfiles()

        # create run files
        self.make_runfiles()

#TASK_MAP = {
#    "visual_search": VisualSearch,
#    "n_back": NBack,
#    "social_prediction": SocialPrediction,
#    "semantic_prediction": SemanticPrediction,
#    "action_observation": ActionObservation,
#    "theory_of_mind": TheoryOfMind,
#    "rest": Rest,
#    }

TASK_MAP = {
    "visual_search": VisualSearch,
    "theory_of_mind": TheoryOfMind,
    "n_back": NBack,
    "social_prediction": SocialPrediction,
    "semantic_prediction": SemanticPrediction,
    "action_observation": ActionObservation,
    "rest": Rest,
    }

# make target and run files
mk = MakeFiles()
mk.make_all()