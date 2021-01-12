from pathlib import Path

import os
import re
import pandas as pd
import numpy as np
import random
from PIL import Image 
import moviepy.editor as mp
import cv2
import glob

import experiment_code.constants as consts

class Utils():

    def __init__(self):
        pass

    def _sample_evenly_from_col(self, dataframe, num_stim, column="trial_type", **kwargs):
        if kwargs.get("random_state"):
            random_state = kwargs["random_state"]
        else: 
            random_state = 2
        num_values = len(dataframe[column].unique())
        group_size = int(np.ceil(num_stim / num_values))

        group_data = dataframe.groupby(column).apply(lambda x: x.sample(group_size, random_state=random_state, replace=False))
        group_data = group_data.sample(num_stim, random_state=random_state, replace=False).reset_index(drop=True).sort_values(column)

        return group_data.reset_index(drop=True)

    def _get_target_file_name(self, targetfile_name):
        # figure out naming convention for target files
        target_num = []

        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)
            
        for f in os.listdir(self.target_dir):
            if re.search(targetfile_name, f):
                regex = r"_(\d+).csv"
                target_num.append(int(re.findall(regex, f)[0]))
                
        if target_num==[]:
            outfile_name = f"{targetfile_name}_01.csv" # first target file
        else:
            num = np.max(target_num)+1
            outfile_name = f"{targetfile_name}_{num:02d}.csv" # second or more
        
        return outfile_name
    
    def _save_target_files(self, df_target):
        """ saves out target files
            Args:
                df_target (pandas dataframe)
            Returns:
                modified pandas dataframes `df_target`
        """
        # shuffle and set a seed (to ensure reproducibility)
        df_target = df_target.sample(n=len(df_target), random_state=self.random_state, replace=False)

        start_time = np.round(np.arange(0, self.num_trials*(self.trial_dur+self.iti_dur), self.trial_dur+self.iti_dur), 1)
        data = {"trial_dur":self.trial_dur, "iti_dur":self.iti_dur, "start_time":start_time, "hand": self.hand, "key_hand_dict": self.key_hand_dict}

        df_target = pd.concat([df_target, pd.DataFrame.from_records(data)], axis=1, ignore_index=False, sort=False)

        # get targetfile name
        tf_name = f"{self.task_name}_{self.block_dur_secs}sec" # was {self.num_trials}trials
        tf_name = self._get_target_file_name(tf_name)

        # save out dataframe to a csv file in the target directory (TARGET_DIR)
        df_target.to_csv(os.path.join(self.target_dir, tf_name), index=True, header=True)

        print(f'saving out {tf_name}')