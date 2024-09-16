from pathlib import Path

import numpy as np
import os
import subprocess as sp
import pickle
import  matplotlib.pyplot as plt

path = "/NeuroLS_DecisionTransformer/run_benchmark.py"
path2 = "/NeuroLS_DecisionTransformer/run_nls_jssp.py"
path3 = "/NeuroLS_DecisionTransformer/data/JSSP/"
MODEL = "30x20/NLSDT/rural-valley-270-2158.pt"
NLSDT_RESULTS_PATH = "/NeuroLS_DecisionTransformer/experiment_outputs/nlsdt/15x15_varied_rtgs/"
NLS_RESULTS_PATH = "/NeuroLS_DecisionTransformer/experiment_outputs/nls/15x15/1"

def vary_rtgs():
    #Run the specified decision transformer model with various rtgs
    mean_makespans = {}
    rtg_range = np.arange(0.25,1.8,0.05)
    os.chdir("/NeuroLS_DecisionTransformer/")
    for i in rtg_range: #Runs the model for each rtg in rtg_range  on the dataset given in directory with flag -g
        out = None
        cmd = 'python ' + path + ' -r ' + path2 + ' -d ' + path3 + f' -g jssp30x20/test -p jssp -m nls -e eval_jssp --args env=jssp30x20_unf -n 200 -x {MODEL}  -a True -f '+ str(round(i,2))
        try:
            print(os.getcwd())
            out = sp.run(cmd.split(),
                         universal_newlines=True,
                         capture_output=True,
                         check=True
                         )
            print(out.stdout)
        except sp.CalledProcessError as e:
            print(f"encountered error for call: {e.cmd}\n")
            print(e.stderr)
        makespan = float(out.stdout.split("makespan")[1])
        mean_makespans[i] = makespan

    lists = sorted(mean_makespans.items()) # Sorts achieved mean makespans of each rtg

    #plot mean makespans
    x, y = zip(*lists)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    plt.plot(x, y)
    plt.title("Mean makespan of various rtgs on 100 instances")
    plt.ylabel('mean makespan')
    plt.xlabel('return-to-go factor')
    plt.savefig('plot_30x20.png')
    plt.show()

    #Save data for later use
    with open('30x20_vary_rtg.pkl', 'wb') as file:
        pickle.dump(mean_makespans, file)

def get_dt_schedules(rtg_range):
    #Retrieves schedules solution files that were created during rtg-variation
    dt_schedules = {}
    for i in rtg_range:
        dt_schedules[i] = {}
        for filename in os.scandir(NLSDT_RESULTS_PATH+str(round(i,2))): # for each rtg_value each .npy file from the respective folder is returned
            if os.path.splitext(filename)[1] == ".npy":
                sequence_data  = np.load(filename, allow_pickle=True)
                dt_schedules[i][filename.name[0:-4]] = sequence_data
    return dt_schedules

def get_nls_schedules():
    #Retrieves schedules solution files that were created during rtg-variation
    no_dt_schedules = {}
    for filename in os.scandir(NLS_RESULTS_PATH): #Uses each .npy file in specified path
        if os.path.splitext(filename)[1] == ".npy":
            sequence_data = np.load(filename, allow_pickle=True)
            no_dt_schedules[filename.name[0:-4]] = sequence_data
    return no_dt_schedules

def calculate_influence_of_rtg_on_hamDistance(actions=False): #If actions is true distance will be calculated on actions not schedules
    schedule_index = 2 if actions else 1 # defines if the Hamming distance should be calculated for the machine sequences or for the action sequences
    dist_string= "selected actions" if actions else "machine sequences"
    mean_distance = {}
    rtg_range = np.arange(0.25,1.8,0.05)
    dt_schedules = get_dt_schedules(rtg_range)
    nls_schedules = get_nls_schedules()
    os.chdir("/NeuroLS_DecisionTransformer/newPlots")
    for i in rtg_range:
        mean_distance[i] = []
        for schedule_id in dt_schedules[i]: #schedule id is the unique filename in the rtg or nls folder
            if schedule_id in nls_schedules: #Ensures that the instance was solved by nls and nlsdt
                mean_distance[i].append(hamming_distance(np.asarray(dt_schedules[i][schedule_id][schedule_index]).flatten(), np.asarray(nls_schedules[schedule_id][schedule_index]).flatten()))
        #mean_distance[i] = mean_distance[i]/len(nls_schedules)

    lists = sorted(mean_distance.items())
    x, y = zip(*lists)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    fig, ax = plt.subplots()
    bp = ax.boxplot(mean_distance.values(), meanline=True, showmeans=True ,patch_artist=True, boxprops=dict(facecolor="lightblue",color="lightblue"))
    ax.set_xticklabels([round(x,2) for x in mean_distance.keys()])
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    plt.grid(True)
    plt.title(f"Hamming distance of {dist_string} for various rtgs on 100 instances")
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'], loc="upper right")
    plt.ylabel('Hamming distance')
    plt.xlabel('return-to-go factor')
    plt.savefig(f'Hamming{actions}_15x15.png')
    plt.show()
    print(mean_distance)

def hamming_distance(sequence1, sequence2):
    counter = 0
    for i in range(len(sequence1)):
        if sequence1[i] != sequence2[i]: counter += 1
    return counter

def start_time_distance(dt, nls):
    sum = 0
    for i in range(len(nls)):
        sum += abs(dt[i]*100 - nls[i]*100)
    return sum/len(nls)

def calculate_influence_of_rtg_on__starttime_distance(rtg_range):
    mean_distance = {}
    dt_start_times = get_dt_schedules(rtg_range)
    nls_start_times = get_nls_schedules()
    for i in rtg_range:
        mean_distance[i] = []
        for schedule_id in dt_start_times[i]: #schedule id is the unique filename in the rtg or nls folder
            if schedule_id in nls_start_times: #Ensures that the instance was solved by nls and nlsdt
                mean_distance[i].append(start_time_distance(dt_start_times[i][schedule_id][0], nls_start_times[schedule_id][0]))

    lists = sorted(mean_distance.items())
    x, y = zip(*lists)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    fig, ax = plt.subplots()
    bp = ax.boxplot(mean_distance.values(), meanline=True, showmeans=True, patch_artist=True,
                    boxprops=dict(facecolor="lightblue", color="lightblue"))
    ax.set_xticklabels([round(x, 2) for x in mean_distance.keys()])
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    plt.grid(True)
    #plt.plot(x, y)
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'], loc="upper right")
    plt.title(" Start time distance for various rtgs on 100 instances")
    plt.ylabel('Start time distance')
    plt.xlabel('return-to-go factor')
    plt.savefig('std_15x15.png')
    plt.show()

if __name__ == '__main__':
    rtg_range = np.arange(0.25,1.8,0.05)
    #vary_rtgs()
    ''' For these experiments the nls.zip and nlsdt.zip in the experiment_outputs directory need to be unzipped.
        Or alternatively create new files npy of the scheme in tianshou_utils.py:648 for instances to be compared
    '''
    calculate_influence_of_rtg_on_hamDistance(actions=True)
    calculate_influence_of_rtg_on_hamDistance(actions=False)
    calculate_influence_of_rtg_on__starttime_distance(rtg_range)