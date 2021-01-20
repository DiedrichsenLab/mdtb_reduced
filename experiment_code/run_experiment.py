# import libraries
from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
import time
import math
import glob

from psychopy import visual, core, event, gui # data, logging

import experiment_code.constants as consts
from experiment_code.screen import Screen
from experiment_code.task_blocks import TASK_MAP
from experiment_code.ttl import ttl
import experiment_code.constants as const
# -----------------------------------------------------------------------------

def display_input_box():
    """
    opens up an input box in which you can enter the info for the experiment/run/subj
    Args:
        NONE!
    Returns:
        experiment_info (dict) with the following keys:
        subj_id     :   id user assigns to the subject
        study_name  :   name the user enters in the gui
        run_name    :   name of the run csv file which contains the task names in the order they are represented
    """
    #Set up input box
    inputDlg = gui.Dlg(title = "Run Experiment")
    
    inputDlg.addField('Enter Subject ID:') 
    inputDlg.addField('Enter Study Name:')  # either behavioral or fmri
    inputDlg.addField('Enter Run Name:')

    inputDlg.show()

    # record input variables
    experiment_info = {}
    if gui.OK:
        experiment_info['subj_id']    = inputDlg.data[0]
        experiment_info['study_name'] = inputDlg.data[1]
        experiment_info['run_name']   = inputDlg.data[2]
    else:
        sys.exit()
    
    return experiment_info

def get_runfile_info(run_name, study_name):
    """
    gets info for the run
    Args:
        run_name(str)    -   name of the run csv file entered in the input GUI
        study_name(str)  -   name assigned to the study by the user
    Returns:
        run_info(dict)   -   a dictionary with the info for the run. contains keys:
        run_file(pandas dataframe) : the opened run file in the form of a pandas dataframe
        run_num(int)               : an integer representing the run number (get from the run filename)
        task_nums(numpy array)    : a numpy array with the number of target files in each run file

    """
    run_info = {} # a dictionary with all the info for the run 
    # load run file
    run_info['run_file'] = pd.read_csv(consts.run_dir / study_name / f"{run_name}.csv")

    # get run num
    run_info['run_num'] = int(re.findall(r'\d+', run_name)[0])

    # get number of target files in each runfile
    run_info['task_nums'] = np.arange(len(run_info['run_file']))

    return run_info

def check_runfile_results(experiment_info):
    """
    Checks if a file for behavioral data of the current run already exists
    Args:
        experiment_info(dict)   -   a dictionary with all the info for the experiment (after user inputs info in the GUI)
    Returns:
        run_iter    -   how many times this run has been run:)
    """
    study_name = experiment_info['study_name']
    subj_id    = experiment_info['subj_id']
    run_name   = experiment_info['run_name']

    fpath = consts.raw_dir / study_name / 'raw' / subj_id / f"{study_name}_{subj_id}.csv"
    if os.path.isfile(fpath):
        # load in run_file results if they exist 
        run_file_results = pd.read_csv(fpath)
        if len(run_file_results.query(f'run_name=="{run_name}"')) > 0:
            current_iter = run_file_results.query(f'run_name=="{run_name}"')['run_iter'].max() # how many times has this run_file been executed?
            run_iter = current_iter+1
        else:
            run_iter = 1 
    else:
        run_iter = 1
        run_file_results = pd.DataFrame()
        pass

    return run_iter, run_file_results

def wait_ttl():
    """
    waits for the ttl pulse (only for fmri study)
    Args:
        NONE
    Returns:
        timer_info(dict)    -   a dictionary with time info. keys:
        t0 : the time??
    """
    timer_info = {}
    ttl.reset()
    while ttl.count <= 0:
        ttl.check()

    # start timer
    timer_info['global_clock'] = ttl.clock()
    timer_info['t0'] = timer_info['global_clock'].getTime()

    return timer_info

def start_timer():
    """
    starts the timer for the experiment (for behavioral study)
    Args:
        NONE
    Returns:
        timer_info(dict)    -   a dictionary with all the info for the timer. keys are:
        global_clock : the clock from psychopy?
        t0           : the time???

    """
    timer_info = {}
    timer_info['global_clock'] = core.Clock()
    timer_info['t0']           = timer_info['global_clock'].getTime()

    return timer_info

def get_targetfile_info(study_name, run_file, b):
    """
    gets the target file information of the task b
    Args:
        study_name(str)             -   name assigned to the study
        run_file(pandas dataframe)  -   run csv file opened as a daraframe
        b(int)                      -   task number
    Returns:
        target_taskInfo(dict)  -   a dictionary containing target file info for the current task with keys:
        task_name     : task name
        task_num      : task number
        target_file   : target csv file opened as a pandas dataframe
        run_endTime   : end time of the task run 
        run_startTime : start time of the task run
        instruct_dur  : duration of instruction for the task
    """
    target_binfo = {}
    # what's the task in this target file?
    target_binfo['task_name'] = run_file['task_name'][b]

    # what number is this target file?
    target_binfo['target_num'] = run_file['target_num'][b]
    if not math.isnan(target_binfo['target_num']):
        target_binfo['target_num'] = (f"{int(target_binfo['target_num']):02d}")

    # load target file
    target_binfo['target_file'] = pd.read_csv(consts.target_dir / study_name / target_binfo['task_name'] / run_file['target_file'][b])

    # get end of run
    target_binfo['run_endTime'] = run_file['end_time'][b]

    # get start of run
    target_binfo['run_startTime'] = run_file['start_time'][b]

    # get instruct dur
    target_binfo['instruct_dur'] = run_file['instruct_dur'][b]

    return target_binfo

def get_task(experiment_info, target_binfo, run_info, 
                   screen, run_iter):
    """
    creates a class for the task 
    Args:
        experiment_info(dict)     -   experiment information:subj_id, study_name, run_name
        screen                    -   screen object
        target_binfo              -   target file information (see get_targetfile_info)
        run_info                  -   run information (see get_runfile_info)
        run_iter                  -   if the run has been done more than one time, this represent the number of repetition of a run
    Returns:
        BlockTask   -   a task class with all the att. and methods associated to the current task in the task
    """
    BlockTask = TASK_MAP[target_binfo['task_name']]
    # BlockTask  = BlockTask(screen = screen, 
    #                         target_file = target_binfo['target_file'], 
    #                         run_end  = target_binfo['run_endTime'], task_name = target_binfo['task_name'], 
    #                         study_name = experiment_info['study_name'], 
    #                         run_name = experiment_info['run_name'], target_num = target_binfo['target_num'],
    #                         run_iter = run_iter, run_num = run_info['run_num'])

    BlockTask  = BlockTask(screen = screen, 
                            target_file = target_binfo['target_file'], 
                            run_end  = target_binfo['run_endTime'], task_name = target_binfo['task_name'],  
                            study_name = experiment_info['study_name'], target_num = target_binfo['target_num'])

    return BlockTask

def wait_starttask(timer_info, run_startTime, study_name):
    """
    Wait till it's time to start the task (reads info from target file)
    Args:
        timer_info      -   a timer object
        run_startTime   -   start time of the task in a specific run
        study_name(str) -   'fmri' or 'behavioral'
    """
    while timer_info['global_clock'].getTime() - timer_info['t0'] <= run_startTime:
        if study_name == 'fmri':
            ttl.check()
        elif study_name == 'behavioral':
            pass
            
def wait_instruct(timer_info, run_startTime, instruct_dur, study_name):
    """
    Wait for a specific amount of time for the instructions specified in the target file
    Args:
        timer_info      -   a timer object
        run_starttime   -   start time of the task in a specific run
        instruct_dur    -   duration of the instruction for the current task
        study_name(str) -   'fmri' or 'behavioral'
    """
    wait_time = run_startTime + instruct_dur
    while timer_info['global_clock'].getTime() - timer_info['t0'] <= wait_time: # timed presentation
        
        if study_name == 'fmri':
            ttl.check()
        elif study_name == 'behavioral':
            pass

def wait_endtask(timer_info, run_endTime, study_name):
    """"
    Waits till the timer reaches the end time of the task 
    Args:
        timer_info      -   a timer object 
        run_endTime     -   end time of the task in the run
        study_name(str) -   'fmri' or 'behavioral'
    """
    while timer_info['global_clock'].getTime() - timer_info['t0'] <= run_endTime: # timed presentation

        if study_name == 'fmri':
            ttl.check()
        elif study_name == 'behavioral':
            pass

def save_resp_df(new_resp_df, study_name, subj_id, task_name):
    """
    gets the response dataframe and save it
    Args: 
        new_resp_df -   response dataframe
        study_name  -   study name: fmri or behavioral
        subj_id     -   id assigned to the subject
        task_name   -   name of the task for the current task block
    """
    # collect existing data
    try:
        target_file_results = pd.read_csv(consts.raw_dir /study_name/ 'raw' / subj_id / f"{study_name}_{subj_id}_{task_name}.csv")
        target_resp_df = pd.concat([target_file_results, new_resp_df], axis=0, sort=False)
        # if there is no existing data, just save current data
    except:
        target_resp_df = new_resp_df
        pass
    # save all data 
    target_resp_df.to_csv(consts.raw_dir / study_name/ 'raw' / subj_id / f"{study_name}_{subj_id}_{task_name}.csv", index=None, header=True)

def get_runfile_results(run_file, all_run_response, run_file_results):
    """
    gets the behavioral results of the current run and returns a dataframe to be saved
    Args:
        run_file            -   run file for the current run
        all_run_response    -   list of dictionaries with behavioral results of the current run
        run_file_results    -   the dataframe which represents results if the run has already been done once
    Returns:
        df_run_results(pandas dataframe)    -   a dataframe containing behavioral results
    """
    # save run results 
    new_run_df = pd.concat([run_file, pd.DataFrame.from_records(all_run_response)], axis=1)
    
    try: # collect existing data
        df_run_results = pd.concat([run_file_results, new_run_df], axis=0, sort=False)
        
    except: # if there is no existing data, just save current data
        df_run_results = new_run_df
        pass 

    return df_run_results

def _get_rt(dataframe):
    """
    Takes in the final result dataframe and calculate Reaction time
    Args:
        dataframe(pandas dataframe)   -   final results of the dataframe
    Returns:
        feedback(dict)    -     a dictionary with the calculated measures  
    """
    rt = dataframe.query('corr_resp==True').groupby(['run_name', 'run_iter'])['rt'].agg('mean')

    rt_curr = None
    rt_prev = None

    if not rt.empty:
        rt_curr = int(round(rt[-1] * 1000))
        if len(rt)>1:
            # get rt of prev. run if it exists
            rt_prev = int(round(rt[-2] * 1000))

    feedback = {'curr': rt_curr, 'prev': rt_prev, 'measure': 'ms'} 

    return feedback 

def _get_acc(dataframe):
    """
    Takes in the final result dataframe and calculate Reaction time
    Args:
        dataframe(pandas dataframe)   -   final results of the dataframe
    Returns:
        feedback(dict)    -     a dictionary with the calculated measures  
    """
    acc = dataframe.groupby(['run_name', 'run_iter'])['corr_resp'].agg('mean')

    acc_curr = None
    acc_prev = None
    if not acc.empty:
        acc_curr = int(round(acc[-1] * 100))
        if len(acc)>1:
            # get rt of prev. run if it exists
            acc_prev = int(round(acc[-2] * 100))

    feedback = {'curr': acc_curr, 'prev': acc_prev, 'measure': '%'} 

    return feedback

def _get_feedback_text(task_name, feedback):
    """
    creates a feedback text
    Args:
        task_name  -   name of the task
        feedback    -   feedback calculated based on either RT or ACC
    Returns:
        a string with the feedback
    """
    return f'{task_name}\n\nCurrent score: {feedback["curr"]}{feedback["measure"]}\n\nPrevious score: {feedback["prev"]}{feedback["measure"]}'

def _display_feedback_text(feedback_all, screen):
    """
    used to display the feedback on the screen
    Args:
        feedback_all    -   feedback for the whole run
        screen          -   screen objects
    """
    positions = [(-9, -6), (0, -6), (9, -6),
                (-9, 3), (0, 3), (9, 3)]

    for position, feedback in zip(positions, feedback_all):
        scoreboard = visual.TextStim(screen.window, text = feedback, color = [-1, -1, -1], pos = position, height = 0.05)
        scoreboard.draw()

    screen.window.flip()

    # wait for end-of-task
    event.waitKeys()

    return

def show_scoreboard(subj_dir, run_filename, screen):
    """
    Presents a score board in the end of the run
    Args:
        subj_dir    -   directory where run results of a subject is stored
        run_file    -   run file used in the current run
    """
    # load run file and get tasks
    dataframe_run = pd.read_csv(os.path.join(subj_dir, run_filename))

    # get unique tasks
    tasks = dataframe_run['task_name'].unique().tolist()

    # remove rest from list if it's present - we don't have scores to display
    if 'rest' in tasks:
        tasks.remove('rest')
        
    feedback_all = []
    for b_name in tasks:  

        # get feedback type (accuracy or reaction time)
        feedback_type = dataframe_run[dataframe_run['task_name']==b_name]['feedback_type'].unique()[0]
        
        # load target file dataframe
        dataframe = pd.read_csv(glob.glob(os.path.join(subj_dir , f'*{b_name}*'))[0])

        # determine feedback
        if feedback_type=="rt":
            feedback = _get_rt(dataframe)
        elif feedback_type=="acc":
            feedback = _get_acc(dataframe = dataframe)

        # get feedback text
        feedback_all.append(_get_feedback_text(b_name, feedback))

    # display feedback text
    _display_feedback_text(feedback_all, screen)

    return

def end_experiment(screen):
    """
    just ends the experiment
    """
    # end experiment
    event.waitKeys()

    # quit screen and exit
    screen.window.close()
    core.quit()
    return

def run():
    """
    opens up a GUI with fields for subject id, experiment name, and run file name
    Run file name is the name of a csv file that containg the names of the task with 
    the order in which they will be presented and their start/end times.
    Once the experiment is run, the data will be saved in a file with a filename specified as follows:
    $experimentName_subjId_taskName.csv
    """

    # 1. open up the input box
    exp_info = display_input_box()

    # 2. get the run information
    run_info = get_runfile_info(exp_info['run_name'], exp_info['study_name'])

    # 3. make subject folder in data/raw/<subj_id>
    subj_dir = consts.raw_dir/ exp_info['study_name'] / 'raw' / exp_info['subj_id']
    consts.dircheck(subj_dir)

    # 4. check for existing run file results in a subject folder
    run_iter, run_file_results = check_runfile_results(exp_info)

    # 5. open screen and display fixation cross
    exp_screen = Screen()

    # 6. timer stuff!
    if exp_info['study_name'] == 'fmri':
        # wait for ttl to begin the task
        timer_info = wait_ttl()
    elif exp_info['study_name'] == 'behavioral':
        # start timer
        timer_info = start_timer()

    # 7. initialize a list for responses
    all_run_response = []

    # 8. loop over tasks
    for b in run_info['task_nums']:

        # 8.1 get target info
        target_binfo = get_targetfile_info(exp_info['study_name'], run_info['run_file'], b)

        # 8.2 get the real strat time for each task
        real_start_time = timer_info['global_clock'].getTime() - timer_info['t0']

        # 8.2.1 collect ttl time and counter
        if exp_info['study_name'] == 'fmri':
            ttl_time  = ttl.time - timer_info['t0']
            ttl_count = ttl.count
        elif exp_info['study_name'] == 'behavioral':
            ttl_time  = 0
            ttl_count = 0

        # 8.3 get the task task
        Task_Block = get_task(exp_info, target_binfo, run_info, 
                               exp_screen, run_iter)

        # 8.4 wait for first start task
        wait_starttask(timer_info, target_binfo['run_startTime'], exp_info['study_name'])

        # 8.5 display instructions
        Task_Block.display_instructions()

        # 8.6 wait for a time period equal to instruction duration
        wait_instruct(timer_info, target_binfo['run_startTime'], target_binfo['instruct_dur'], exp_info['study_name'])

        # 8.7.1 run task and collect feedback
        new_resp_df = Task_Block.run()
        # 8.7.2 adding run information to response dataframe
        new_resp_df['run_name'] = exp_info['run_name']
        new_resp_df['run_iter'] = run_iter
        new_resp_df['run_num']  = run_info['run_num']
        # 8.7.3 get the response dataframe and save it
        save_resp_df(new_resp_df, exp_info['study_name'], exp_info['subj_id'], target_binfo['task_name'])

        # 8.8 log results
        # collect real_end_time for each task
        all_run_response.append({
            'real_start_time': real_start_time,
            'real_end_time': (timer_info['global_clock'].getTime() - timer_info['t0']),
            'ttl_counter': ttl_count,
            'ttl_time': ttl_time,
            'run_name': exp_info['run_name'],
            'task_idx': b+1,
            'run_iter': run_iter,
            'run_num': run_info['run_num'],
        })

        # 8.9 wait for end-of-task
        wait_endtask(timer_info, target_binfo['run_endTime'], exp_info['study_name'])

    # 9.1 get the run result as a dataframe
    df_run_results = get_runfile_results(run_info['run_file'], all_run_response, run_file_results)

    # 9.2 save the run results
    run_filename = f"{exp_info['study_name']}_{exp_info['subj_id']}.csv"
    df_run_results.to_csv(subj_dir / run_filename, index=None, header=True)

    # 10. present feedback from all tasks on screen 
    show_scoreboard(subj_dir, run_filename, exp_screen)

    # 11. end experiment
    Task_Block.display_end_run()
    end_experiment(exp_screen)

    return