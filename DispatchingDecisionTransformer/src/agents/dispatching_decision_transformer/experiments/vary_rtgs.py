import copy
import numpy as np
from src.agents.dispatching_decision_transformer.dataset_creation import load_config, load_data_splits
from src.agents.dispatching_decision_transformer.DispatchingDecisionTransformer import DispatchingDecisionTransformer
from src.agents.dispatching_decision_transformer.utils.plot_utils import plot_rtg_variations,create_boxplot
from src.agents.dispatching_decision_transformer.utils.loading_utils import load_mppo, solve_mppo
from src.environments.environment_loader import EnvironmentLoader

def hamming_distance(list1, list2):
    counter = 0
    for i in range(len(list1)):
        if list1[i] != list2[i]: counter += 1
    return counter


def start_time_distance(env_mppo, env_dt):
    assert (len([tsk for tsk in env_mppo.tasks if tsk.done == True]) == 36 and len([tsk for tsk in env_dt.tasks if tsk.done == True]) == 36)
    sum = 0
    for i in range(len(env_mppo.tasks)):
        sum += abs(env_mppo.tasks[i].started - env_dt.tasks[i].started)
    return sum/36


def vary_rtgs(modelname,context_length, rtg_range):
    #Runs the selected DTmodel and the origin-agent with the defined rtgs on 100 datapoints each
    config = load_config()
    train_data, test_data, val_data = load_data_splits(config)
    test_data = test_data[:100]
    dt = DispatchingDecisionTransformer(model_path=modelname, context_length=context_length)
    mean_makespan_dt = {}
    mean_hamming = {}
    mean_start_time = {}
    mppo_actions = []
    mppo_makespans = []
    mppo_envs = []
    dt_envs = []
    #First solve all instances with the mppo
    for i in range(100):
        env, _ = EnvironmentLoader.load(config, data=test_data[i:i+1]) #We create an own environment for each datapoint to make solutions 1:1 comparable.
        env_dt = copy.deepcopy(env) #Deep copy environment for dt so the instances to solve are identical
        dt_envs.append(env_dt)
        model = load_mppo(config, env, './base_networks/6x6_agent_mppo')
        mppo_action, mpp_makespan = solve_mppo(env, model)
        mppo_envs.append(env) #We save the environment because we need it to calculate the start_time_distance
        mppo_makespans.append(mpp_makespan)
        mppo_actions.append(mppo_action)
    print(f"Mppo mean makespan: {np.mean(mppo_makespans)}")

    for lower_bound_factor in rtg_range:
        if lower_bound_factor == 1.0:
            ham_distances_better = []
            st_distances_better = []
            ham_distances_worse = []
            st_distances_worse = []
            ham_distances_eq = []
            st_distances_eq = []
        makespans = []
        ham_distances = []
        start_time_distances = []
        for i in range(100):
            makespan, dt_actions = dt.solve_environment(dt_envs[i], lower_bound_factor) # Solve instance with dt
            makespans.append(makespan)
            #calculate distance between mppo and dt for current instance
            start_time_distances.append(start_time_distance(mppo_envs[i],dt_envs[i]))
            ham_distances.append(hamming_distance(mppo_actions[i], dt_actions))

            # Only plot distances for the 100 instances for rtg 1
            if lower_bound_factor == 1.0:
                if (mppo_makespans[i] > makespan):
                    ham_distances_better.append(hamming_distance(mppo_actions[i], dt_actions))
                    print("hamming:", hamming_distance(mppo_actions[i], dt_actions))
                    st_distances_better.append(start_time_distance(mppo_envs[i], dt_envs[i]))
                elif (mppo_makespans[i] < makespan):
                    ham_distances_worse.append(hamming_distance(mppo_actions[i], dt_actions))
                    print("hamming:", hamming_distance(mppo_actions[i], dt_actions))
                    st_distances_worse.append(start_time_distance(mppo_envs[i], dt_envs[i]))
                else:
                    ham_distances_eq.append(hamming_distance(mppo_actions[i], dt_actions))
                    print("hamming:", hamming_distance(mppo_actions[i], dt_actions))
                    st_distances_eq.append(start_time_distance(mppo_envs[i], dt_envs[i]))
        if lower_bound_factor == 1.0:
            create_boxplot(ham_distances_better, ham_distances_worse, ham_distances_eq, st_distances_better,
                           st_distances_worse, st_distances_eq, context_length)


        mean_makespan_dt[lower_bound_factor] = (np.mean(makespans))
        mean_hamming[lower_bound_factor] = (np.mean(ham_distances))
        mean_start_time[lower_bound_factor] = (np.mean(start_time_distances))
        print(f"{lower_bound_factor}: Makespan {mean_makespan_dt[lower_bound_factor]}")
        print(f"{lower_bound_factor} hamming: {mean_hamming[lower_bound_factor]}")
        print(f"{lower_bound_factor}: start-time {mean_start_time[lower_bound_factor]}")
        print(f"----------------------------------")

    result = {}
    result["makespan"] = sorted(mean_makespan_dt.items())
    result["hamming"] = sorted(mean_hamming.items())
    result["start_time"] = sorted(mean_start_time.items())

    return result


if __name__ == '__main__':
    import os
    os.chdir("..")
    print(os.getcwd())
    models = {"CL6.pt": 6, "CL2.pt": 2, "CL30.pt": 30}
    model_results = {}
    rtg_range = np.arange(1.0, 1.05, 0.05) # Set wanted range
    for modelname,context_length in models.items():
        model_results[context_length] = vary_rtgs(modelname, context_length, rtg_range)
    plot_rtg_variations(model_results)
