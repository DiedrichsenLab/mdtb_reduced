# import libraries
from pathlib import Path
import os

response_keys = ['a', 's', 'd', 'f', 'j', 'k', 'l', ';']

# assign keys to hands
key_hand_dict_binary_right = {
        True:  [response_keys[4], 'Index'], # index finger
        False: [response_keys[5], 'Middle'],  # middle finger
    }
    
key_hand_dict_binary_left = {
        False: [response_keys[2], 'Middle'], # index finger
        True:  [response_keys[3], 'Index'],  # middle finger
    }

key_hand_dict_sequence_right = {
        1: [response_keys[4], 'Index'], # index finger
        2: [response_keys[5], 'Middle'],  # middle finger
        3: [response_keys[6], 'Ring'], # ring finger
        4: [response_keys[7], 'Pinky'],  # pinky finger
    }
    
key_hand_dict_sequence_left = {
        4: [response_keys[0], 'Pinky'], # pinky finger
        3: [response_keys[1], 'Ring'],  # ring finger
        2: [response_keys[2], 'Middle'], # middle finger
        1: [response_keys[3], 'Index'],  # index finger
    }    

base_dir   = Path(__file__).absolute().parent.parent
stim_dir   = base_dir / "stimuli"
target_dir = base_dir / "target_files"
run_dir    = base_dir / "run_files"

# This is where the result files are being saved
raw_dir    = base_dir/ "data"  

def dircheck(path2dir):
    """
    Checks if a directory exists! if it does not exist, it creates it
    Args:
    dir_path    -   path to the directory you want to be created!
    """

    if not os.path.exists(path2dir):
        print(f"creating {path2dir}")
        os.makedirs(path2dir)
    else:
        print(f"{path2dir} already exists")

def dirtree():
    """
    Create all the directories if they don't already exist
    """
    fpaths = [raw_dir, stim_dir, target_dir, run_dir]
    for fpath in fpaths:
        dircheck(fpath)