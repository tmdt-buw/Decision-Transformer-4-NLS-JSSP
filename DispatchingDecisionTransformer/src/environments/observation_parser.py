import numpy as np
import copy
from typing import List

from src.data_generator.task import Task
from src.agents.heuristic.heuristic_agent import HeuristicSelectionAgent

# from src.agents.heuristic.bottleneck_identification.bottleneck import BottleneckIdentification

OBS_VALUES = ['job_index', 'task_index', 'machines', 'tools', 'runtime',
              'deadline']  # Todo: Christian, brauchen wir das noch?


class ObservationParser:
    @staticmethod
    def parse(env, observation_strategy) -> np.ndarray:
        """
        Parses the current state of the env to an observation vector. according to the 'observation_strategy'.

        :param env: environment from where the state is parsed to the observation
        :param observation_strategy: string that defines the observation strategy

        :return: the observation as a numpy array

        """
        # optional check if the observation strategy is implemented and usable for the current env and networks
        observation = getattr(ObservationParser, observation_strategy)(env)

        return observation

    @staticmethod
    def simulate_action_steps(env):
        """
        simulate the action steps for the next task of each job
        :return: makespan differences caused by each schedulable next task
        """
        next_tasks = env.get_next_tasks()
        makespan_differences = np.ones(env.num_jobs) * env.max_runtime
        for job_id, task in enumerate(next_tasks):
            if task:
                # check task preceding in the job (if it is not the first task within the job)
                if task.task_index == 0:
                    start_time_of_preceding_task = 0
                else:
                    preceding_task = env.tasks[env.task_job_mapping[(job_id, task.task_index - 1)]]
                    start_time_of_preceding_task = preceding_task.finished

                # check earliest possible time to schedule according to preceding task and needed machine
                start_time = max(start_time_of_preceding_task, env.ends_of_machine_occupancies[task.selected_machine])
                end_time = start_time + task.runtime
                old_makespan = max(env.ends_of_machine_occupancies)
                new_makespan = max(old_makespan, end_time)
                makespan_differences[job_id] = new_makespan - old_makespan

        return makespan_differences / env.max_runtime

    # feature section
    # TODO: should we create a separate class or file for this?

    def basic(env) -> np.ndarray:
        """
        basic observation strategy. Consists of:
        - runtime of next tasks per job
        - task index of next task per job

        :return: scaled vector of the observation

        """
        obs = []
        next_tasks = env.get_next_tasks()

        # assemble information on next tasks - note that the observation is ordered by job id!
        for task in next_tasks:
            if task is not None:
                # append scaled information
                obs.append(task.runtime / env.max_runtime)
                obs.append(task.task_index / (env.num_tasks + 1))
            else:
                # append dummies
                obs.append(1)  # maximal scaled runtime -> 1
                obs.append(1)  # maximal scaled task index -> 1

        obs = np.array(obs)

        return obs

    def full_raw(env) -> [float]:
        """
        full raw observation. Consists of:
        - remaining processing times of all tasks on each machine
        - remaining processing times of tasks on each job
        - processing time of the next task per job
        - which machine is used the for next task per job

        :return: scaled, concatenated array of the observation

        """
        # all information, normalized
        # (1) remaining time of operations currently being processed on each machine (not compatible with our offline
        # interaction logic
        # (2) sum of all task processing times still to be processed on each machine
        remaining_processing_times_on_machines = np.zeros(env.num_machines)
        # (3) sum of all task processing times left on each job
        remaining_processing_times_per_job = np.zeros(env.num_jobs)
        # (4) processing time of respective next task on job (-1 if job is done)
        operation_time_of_next_task_per_job = np.zeros(env.num_jobs)
        # (5) machine used for next task (altered for FJJSP compatability to one-hot encoded multibinary representation)
        machines_for_next_task_per_job = np.zeros((env.num_jobs, env.num_machines))
        # (6) time passed at any given moment. Not really applicable to the offline scheduling case.

        # feature assembly
        next_tasks = env.get_next_tasks()
        for task in env.tasks:
            if task.done:
                pass
            if not task.done:
                remaining_processing_times_on_machines[np.argwhere(task.machines)] += task.runtime
                remaining_processing_times_per_job[task.job_index] += task.runtime
                if task == next_tasks[task.job_index]:  # next task of the job
                    operation_time_of_next_task_per_job[task.job_index] += task.runtime
                    machines_for_next_task_per_job[task.job_index] = task.machines

        # normalization
        remaining_processing_times_on_machines /= (env.num_jobs * env.max_runtime)
        remaining_processing_times_per_job /= (env.num_tasks * env.max_runtime)
        operation_time_of_next_task_per_job /= env.max_runtime

        observation = np.concatenate([
            remaining_processing_times_on_machines,
            remaining_processing_times_per_job,
            operation_time_of_next_task_per_job,
            machines_for_next_task_per_job.flatten()
        ])

        return observation

    def mtr_comparison(env) -> np.ndarray:
        """
        mtr comparison observation. Consists of:
        - jobs are compared to each other by how far they have been processed. 1 if a job is further, -1 if it is less
          far, 0 if they are at the same position

        :return: scaled vector of the observation

        """
        # mtr observation
        mtr_array = np.zeros(int((env.num_jobs ** 2 - env.num_jobs) / 2))
        place_counter = 0
        for i in range(env.num_jobs):
            for j in range(i + 1, env.num_jobs):
                if env.job_task_state[i] > env.job_task_state[j]:
                    mtr_array[place_counter] = 1
                if env.job_task_state[i] < env.job_task_state[j]:
                    mtr_array[place_counter] = -1
                place_counter += 1

        return mtr_array

    def mtr_info(env):
        """
        mtr info observation. Consists of:
        - information about the current task index (only information needed for mtr)

        :return: scaled vector of the observation

        """
        info_array = []
        for job in np.arange(env.num_jobs):
            t_idx = env.job_task_state[job] if env.job_task_state[job] < env.max_task_index else env.max_task_index
            next_task_in_job = copy.copy(env.tasks[env.task_job_mapping[job, t_idx]])

            info_array.append(next_task_in_job.task_index / (env.num_tasks + 1))

        return info_array

    def spt_comparison(env):
        """
        spt comparison observation. Consists of:
        - jobs are compared to each other by their (overall) remaining runtime. 0 if a job has a longer remaining
          runtime, -1 if it has a short remaining runtime, 0 if both have the same remaining runtime

        :return: scaled vector of the observation

        """

        # sum of all task processing times left on each job
        remaining_processing_times_per_job = np.zeros(env.num_jobs)
        for task in env.tasks:
            if task.done:
                pass
            if not task.done:
                remaining_processing_times_per_job[task.job_index] += task.runtime

        # create comparison obs
        spt_array = np.zeros(env.num_jobs)
        place_counter = 0
        for i in range(env.num_jobs):
            for j in range(i + 1, env.num_jobs):
                if remaining_processing_times_per_job[i] < remaining_processing_times_per_job[j]:
                    spt_array[place_counter] = 1
                if remaining_processing_times_per_job[i] > remaining_processing_times_per_job[j]:
                    spt_array[place_counter] = -1
                place_counter += 1

        return spt_array

    def mtr_one_hot(env):
        """
        mtr one hot observation. Consists of:
        - result of regular mtr heuristic

        :return: one hot encoded vector of the observation

        """
        heuristic_agent = HeuristicSelectionAgent()
        heuristic = 'MTR'
        selected_action = heuristic_agent(env.tasks, env.get_action_mask(), heuristic)
        mtr_array = env.to_one_hot(selected_action, env.num_jobs)

        obs = np.array(mtr_array).flatten()

        return obs

    def spt_one_hot(env):
        """
        spt one hot observation. Consists of:
        - result of regular spt heuristic

        :return: one hot encoded vector of the observation

        """
        heuristic_agent = HeuristicSelectionAgent()
        heuristic = 'SPT'
        selected_action = heuristic_agent(env.tasks, env.get_action_mask(), heuristic)
        mtr_array = env.to_one_hot(selected_action, env.num_jobs)

        obs = np.array(mtr_array).flatten()

        return obs

    def heuristics_one_hot(env):
        """
        heuristics one hot observation. Consists of:
        - result of all heuristic/algorithms apart from rand

        :return: one hot encoded vector for each of the results

        """
        heuristic_res = []
        heuristic_agent = HeuristicSelectionAgent()
        heuristic_list = ['EDD', 'SPT', 'MTR', 'LTR']
        for heuristic in heuristic_list:
            selected_action = heuristic_agent(env.tasks, env.get_action_mask(), heuristic)
            heuristic_res.append(env.to_one_hot(selected_action, env.num_jobs))

        obs = np.array(heuristic_res).flatten()

        return obs

    def runtime_relation(env):
        """
        runtime relation observation. Consists of:
        - puts the remaining runtime for the selected job (job_idx) in relation to all other jobs. 0 if one of the jobs
          is finished.

        :return: scaled vector of the observation

        """

        # sum of all task processing times left on each job
        remaining_processing_times_per_job = np.zeros(env.num_jobs)
        for task in env.tasks:
            if task.done:
                pass
            if not task.done:
                remaining_processing_times_per_job[task.job_index] += task.runtime

        # create relation array
        relation_array = np.zeros(env.num_jobs)
        place_counter = 0
        for i in range(env.num_jobs):
            for j in range(i + 1, env.num_jobs):
                # needed to avoid "divide by zero error" if remaining processing time is zero
                if remaining_processing_times_per_job[i] == 0 or remaining_processing_times_per_job[j] == 0:
                    relation_array[place_counter] = 0
                else:
                    relation_array[place_counter] = remaining_processing_times_per_job[i] / \
                                                    remaining_processing_times_per_job[j]
                place_counter += 1

        return relation_array

    def lookahead_makespan(env) -> np.ndarray:
        """
        lookahead makespan observation. Consists of:
        - how much does the makespan increase if the next task of each job is scheduled

        :return: scaled vector of the observation

        """
        # lookhead makespan observation
        next_tasks = env.get_next_tasks()

        makespan_differences = np.ones(env.num_jobs)
        for job_id, task in enumerate(next_tasks):
            if task:
                # check task preceding in the job (if it is not the first task within the job)
                if task.task_index == 0:
                    start_time_of_preceding_task = 0
                else:
                    preceding_task = env.tasks[env.task_job_mapping[(job_id, task.task_index - 1)]]
                    start_time_of_preceding_task = preceding_task.finished

                # check earliest possible time to schedule according to preceding task and needed machine
                start_time = max(start_time_of_preceding_task, env.ends_of_machine_occupancies[task.selected_machine])
                end_time = start_time + task.runtime
                old_makespan = max(env.ends_of_machine_occupancies)
                new_makespan = max(old_makespan, end_time)
                makespan_differences[job_id] = new_makespan - old_makespan

        lookahead_feature = makespan_differences / env.max_runtime

        return lookahead_feature

    def lookahead_makespan_v2(env):
        ends_machines = env.ends_of_machine_occupancies
        ends_jobs = np.zeros(env.num_jobs)

        next_tasks = env.get_next_tasks()

        previous_tasks = env.get_previous_tasks()
        for job in range(env.num_jobs):
            if previous_tasks[job] != 0:
                ends_jobs[job] = previous_tasks[job].finished
            else:
                ends_jobs[job] = 0

        obs = []

        current_makespan = env.get_makespan()

        for task in next_tasks:
            if task:
                r = task.runtime
                m = task.selected_machine
                start = max(ends_machines[m], ends_jobs[task.job_index])
                end = start + r
                obs.append((end - current_makespan) / env.max_runtime)
            else:
                obs.append(1)

        return obs

    def lookahead_makespan_copy(env):
        """
        lookahead makespan copy observation. Consists of:

        * how much does the makespan increase if the next task of each job is scheduled
        * using a copy of the current state and performing a dummy step with each possible next task to calculate the
          diffs -> not really usable due to computational inefficiency

        :return: scaled vector of the observation

        """
        next_tasks = env.get_next_tasks()
        makespan_diff = []

        # perform next possible actions one by one and calculate diff to new makespan after job execution
        for task in next_tasks:
            if task:
                machine = task.selected_machine
                job_id = task.job_index
                old_makespan = env.get_makespan()
                env_copy = copy.deepcopy(env)
                env_copy.execute_action(job_id, task, machine)
                new_makespan = env_copy.get_makespan()
                makespan_diff.append((new_makespan - old_makespan) / env.max_runtime)
            else:
                dummy_makespan = env.get_makespan() + env.max_runtime
                makespan_diff.append((dummy_makespan - env.get_makespan()) / env.max_runtime)

        # return those diffs as an observation
        return makespan_diff

    def lookahead_machine_idle(env):
        """
        lookahead machine idle observation. Consists of:
        - white spaces of each machine for current state and next tasks that could be scheduled

        :return: total white spaces after task execution for each job as a scaled vector OR: scaled diffs

        """

        # slot dict for current state
        slot_dict = {}
        # for each machine (key) each task logs start and end time (value) if processed on the machine
        for machine in range(env.num_machines):
            slot_dict[f'{machine}'] = [(0, 0)]
        for task in env.tasks:
            if task.done:
                slot_dict[f'{task.selected_machine}'].append((task.started, task.finished))

        # calculate white spaces for current state
        white_spaces = np.zeros(env.num_machines)
        for machine in range(env.num_machines):
            # sort dict values of current machine (=key)
            slot_dict[f'{machine}'].sort()

            # calculate white spaces for current machine
            white_space = 0
            for i in range(len(slot_dict[f'{machine}']) - 1):
                # case where no tasks have been started on the current machine yet
                if len(slot_dict[f'{machine}']) > 1:
                    # case where first task of the current machine is not started at timestep 0
                    if slot_dict[f'{machine}'][i][1] == 0 and slot_dict[f'{machine}'][i + 1][0] != 0:
                        white_space += slot_dict[f'{machine}'][i + 1][0]
                    else:
                        # calculate white spaces using the differenz of second value of the current tuple and
                        # first value of the next tuple
                        white_space += slot_dict[f'{machine}'][i + 1][0] - slot_dict[f'{machine}'][i][1]
            white_spaces[machine] = white_space

        # initialize slot_ahead creating a dict for each job
        slot_ahead = [dict() for x in range(env.num_jobs)]
        for i in range(env.num_jobs):
            slot_ahead[i] = copy.deepcopy(slot_dict)

        # update slot_ahead using all the next tasks (jobs)
        next_tasks = env.get_next_tasks()
        previous_tasks = env.get_previous_tasks()
        ends_machines = env.ends_of_machine_occupancies
        ends_jobs = np.zeros(env.num_jobs)

        for job in range(env.num_jobs):
            if previous_tasks[job] != 0:
                ends_jobs[job] = previous_tasks[job].finished
            else:
                ends_jobs[job] = 0

        for task in next_tasks:
            if task:
                start = max(ends_machines[task.selected_machine], ends_jobs[task.job_index])
                end = start + task.runtime
                slot_ahead[task.job_index][f'{task.selected_machine}'].append((start, end))

        # calculate new whitespaces according to slot_ahead
        new_whitespaces = np.zeros((env.num_jobs, env.num_machines))
        for job in range(env.num_jobs):
            for machine in range(env.num_machines):
                slot_ahead[i][f'{machine}'].sort()
                white_space = 0
                for i in range(len(slot_ahead[job][f'{machine}']) - 1):
                    # case where no tasks have been started on the current machine yet
                    if len(slot_ahead[job][f'{machine}']) > 1:
                        # case where first task of the current machine is not started at timestep 0
                        if slot_ahead[job][f'{machine}'][i][1] == 0 and slot_ahead[job][f'{machine}'][i + 1][0] != 0:
                            white_space += slot_ahead[job][f'{machine}'][i + 1][0]
                        else:
                            # calculate white spaces using the differenz of second value of the current tuple and
                            # first value of the next tuple
                            white_space += slot_ahead[job][f'{machine}'][i + 1][0] - slot_ahead[job][f'{machine}'][i][1]
                    new_whitespaces[job][machine] = white_space

        # calculate white spaces diffs for each job that could be scheduled and return it as an observation
        diffs = np.zeros(env.num_jobs)
        for i in range(env.num_jobs):
            diffs[i] = sum(new_whitespaces[i]) - sum(white_spaces)

        # calculate total white spaces for each job and scale it using min/max scaling
        white_spaces_total = np.zeros(env.num_jobs)
        for i in range(env.num_jobs):
            if max(new_whitespaces[i] != 0):
                # new white spaces for each job = old+new whitespaces, scaled using max total white space
                white_spaces_total[i] = (sum(white_spaces) + sum(new_whitespaces[i])) / (
                    max(new_whitespaces[i] + max(white_spaces)))

        # return white_spaces_total

        # alternative obs using only the white space diffs caused by the different jobs
        diffs = np.zeros(env.num_jobs)
        for i in range(env.num_jobs):
            diffs[i] = sum(new_whitespaces[i]) - sum(white_spaces)

        # scaling
        if max(white_spaces_total) != 0:
            diffs /= np.max(new_whitespaces)

        return diffs

    def lookahead_runtime_job(env):
        """
        lookahead remaining runtime per job observation. Consists of:
        - remaining runtimes per job for each next task that could be scheduled

        :return: scaled vector of the observation

        """

        next_tasks = env.get_next_tasks()

        obs = []

        # sum of all task processing times left on each job
        remaining_processing_times_per_job = np.zeros(env.num_jobs)
        for task in env.tasks:
            if task.done:
                pass
            if not task.done:
                remaining_processing_times_per_job[task.job_index] += task.runtime

        # calculate remaining runtimes per job if each next task is executed
        for task in next_tasks:
            if task:
                remaining_time = remaining_processing_times_per_job[task.job_index]
                if remaining_time != 0:
                    remaining_processing_times_per_job[task.job_index] -= task.runtime
                    obs.append(remaining_processing_times_per_job[task.job_index] / (env.num_tasks * env.max_runtime))
            else:
                obs.append(0)

        # return obs
        return obs

    def lookahead_runtime_machine(env):
        """
        lookahead remaining runtime per machine observation. Consists of:
        - remaining runtimes per machine for each next task that could be scheduled

        :return: scaled vector of the observation

        """

        next_tasks = env.get_next_tasks()
        remaining_processing_times_on_machines = np.zeros(env.num_machines)

        obs = []

        # sum of all task processing times left on each machine
        for task in env.tasks:
            if task.done:
                pass
            if not task.done:
                remaining_processing_times_on_machines[np.argwhere(task.machines)] += task.runtime

        # calculate remaining runtimes per machine if each next task is executed
        for task in next_tasks:
            if task:
                remaining_time = remaining_processing_times_on_machines[task.selected_machine]
                if remaining_time != 0:
                    remaining_processing_times_on_machines[task.selected_machine] -= task.runtime
                    obs.append(remaining_processing_times_on_machines[task.selected_machine] /
                               (env.num_jobs * env.max_runtime))
            else:
                obs.append(0)

        # normalize and return obs
        return obs

    def lookahead_mtr(env):
        """
        lookahead mtr observation. Consists of:
        - todo

        :return: todo

        """
        # TODO Merlin, neu schreiben
        # get current tasks remaining for each job (tr = tasks remaining)
        tr_counter = np.zeros(env.num_jobs)
        for task in env.tasks:
            if not task.done:
                tr_counter[task.job_index] += 1

        # current decision (not really needed (?))
        # selected_job = np.argmax(tr_counter)
        # current_decision = env.to_one_hot(selected_job, env.num_jobs)

        # tasks remaining if each next task is scheduled
        # tr = [ [task1 is executed(size of num_jobs)] ... [task_i is executed]  ]
        #      outer size = num_tasks, inner size = num_jobs
        tr_counter_ahead = np.zeros((env.num_tasks, env.num_jobs))
        for i in range(env.num_tasks):
            tr_counter_ahead[i] = tr_counter.copy

        for i, task in enumerate(env.get_next_tasks()):
            # würde bedeuten, dass tasks direkt fertig sind,
            # remaining tasks direkt einer weniger sind im job (alternative?)
            # vielleicht dann zusätzlich remaining runtime als obs?
            tr_counter_ahead[i][task.job_index] -= 1

        # decision for each next task
        ahead_decision = np.zeros((env.num_tasks, env.num.jobs))
        for i in range(env.num_tasks):
            selected_job_ahead = np.argmax(tr_counter_ahead[i])
            ahead_decision[i] = env.to_one_hot(selected_job_ahead, env.num_jobs)

        # alternativ: nicht wirklich mtr zurückgeben, sondern differenz zwischen tasks remaining?
        # oder diff zwischen den entscheidungen (new-old / num_all_tasks)
        # tr_diffs = np.zeros((env.num_tasks, env.num_jobs))
        # for task in range(env.num_tasks):
        #       for job in range(num_jobs):
        #           tr_diffs[task][job] = tr_counter_ahead[task][job] - tr_counter[job]
        # return tr_diffs / num_all_tasks
        return ahead_decision.flatten()

    @classmethod
    def all_tasks(cls, env):

        tasks = copy.deepcopy(env.tasks)
        feature_list = []

        # normalize
        for task in tasks:
            # normalize
            task.job_index = task.job_index / (env.num_jobs - 1)
            task.runtime = task.runtime / env.max_runtime
            task.task_index = task.task_index / (env.num_tasks - 1)
            task.deadline = task.deadline / env.max_deadline

            # get object attributes according to constant
            features = []
            # extract necessary attributes and append (extend) to features
            for value in OBS_VALUES:
                feature = getattr(task, value)
                if type(feature) == list:
                    features.extend(feature)
                else:
                    features.append(feature)

            # append features of object to list for all
            feature_list.append(features)

        observation = np.array(feature_list)

        return observation

    @classmethod
    def all_tasks_lateness(cls, env):

        # compute and normalize lateness
        tasks = copy.deepcopy(env.tasks)
        feature_list = []
        bneck = BottleneckIdentification(env.tasks)
        bneck.identify(graph=env.sol_graph.graph)
        latenesses = np.array(bneck.latenessOfTask, dtype=np.float)
        max_value = np.max(latenesses)
        min_value = np.min(latenesses)
        if max_value != min_value:
            bn = (latenesses - min_value) / (max_value - min_value)
        else:
            bn = np.zeros_like(latenesses)

        # compute and normalize remaining runtime per job
        remaining_runtime_per_job = np.zeros(env.num_jobs)
        for task in tasks:
            # if task not done yet, add runtime to job remaining runtime
            if not task.done:
                remaining_runtime_per_job[task.job_index] += task.runtime

        for i, remaining_runtime in enumerate(remaining_runtime_per_job):
            # normalize with max = max_runtime*num_tasks -> maximum possible runtime per job
            remaining_runtime_per_job[i] = remaining_runtime / (env.max_runtime * env.num_tasks)

        # normalize
        for lateness, task in zip(bn, tasks):

            # store job_index
            job_index = task.job_index

            # normalize
            task.job_index = task.job_index / (env.num_jobs - 1)
            task.runtime = task.runtime / env.max_runtime
            task.task_index = task.task_index / (env.num_tasks - 1)
            task.deadline = task.deadline / env.max_deadline

            # get object attributes according to constant
            features = []
            # extract necessary attributes and append (extend) to features
            for value in OBS_VALUES:
                feature = getattr(task, value)
                if type(feature) == list:
                    features.extend(feature)
                else:
                    features.append(feature)

            # append lateness and runtime per job
            features.append(lateness)
            features.append(remaining_runtime_per_job[job_index])

            # append features of object to list for all
            feature_list.append(features)

        observation = np.array(feature_list)

        return observation

    @classmethod
    def all_tasks_lookahead(cls, env):

        # compute and normalize lateness
        tasks = copy.deepcopy(env.tasks)
        feature_list = []
        lookahead_feature = cls.simulate_action_steps(env)

        # compute and normalize remaining runtime per job
        remaining_runtime_per_job = np.zeros(env.num_jobs)
        for task in tasks:
            # if task not done yet, add runtime to job remaining runtime
            if not task.done:
                remaining_runtime_per_job[task.job_index] += task.runtime

        for i, remaining_runtime in enumerate(remaining_runtime_per_job):
            # normalize with max = max_runtime*num_tasks -> maximum possible runtime per job
            remaining_runtime_per_job[i] = remaining_runtime / (env.max_runtime * env.num_tasks)

        for task in tasks:

            # store job_index
            job_index = task.job_index

            # normalize
            task.job_index = task.job_index / (env.num_jobs - 1)
            task.runtime = task.runtime / env.max_runtime
            task.task_index = task.task_index / (env.num_tasks - 1)
            task.deadline = task.deadline / env.max_deadline

            # get object attributes according to constant
            features = []
            # extract necessary attributes and append (extend) to features
            for value in OBS_VALUES:
                feature = getattr(task, value)
                if type(feature) == list:
                    features.extend(feature)
                else:
                    features.append(feature)

            features.append(lookahead_feature[job_index])
            features.append(remaining_runtime_per_job[job_index])

            # append features of object to list for all
            feature_list.append(features)

        observation = np.array(feature_list)

        return observation

    @classmethod
    def lookahead_combined(cls, env):

        tasks = copy.deepcopy(env.tasks)
        features = []

        # get 2d feature (feature for each task)
        for task in tasks:

            # normalize
            task.job_index = task.job_index / (env.num_jobs - 1)
            task.runtime = task.runtime / env.max_runtime
            task.task_index = task.task_index / (env.num_tasks - 1)
            task.deadline = task.deadline / env.max_deadline

            # get object attributes according to constant, extract necessary attributes and append features
            for value in OBS_VALUES:
                feature = getattr(task, value)
                if type(feature) == list:
                    features.extend(feature)
                else:
                    features.append(feature)

        # get 1d features
        lookahead = cls.lookahead_makespan(env)
        next_task_runtime = cls.basic(env)

        # concatenate
        features = np.concatenate((features, lookahead, next_task_runtime))

        return features

    def lookahead_makespan_v2(env):
        ends_machines = env.ends_of_machine_occupancies
        ends_jobs = np.zeros(env.num_jobs)

        next_tasks = env.get_next_tasks()

        previous_tasks = env.get_previous_tasks()
        for job in range(env.num_jobs):
            if previous_tasks[job] != 0:
                ends_jobs[job] = previous_tasks[job].finished
            else:
                ends_jobs[job] = 0

        obs = []

        current_makespan = env.get_makespan()

    def switch_info_v1(env):
        # start of task
        # end of task
        # end of machine
        # ideas: utilization before task, utilization after task, machine utilization
        # ideas:

        started_list = []
        finished_list = []
        machine_list = []

        for task in env.tasks:
            started_list.append(task.started / env.benchmark_makespan)
            finished_list.append(task.finished / env.benchmark_makespan)
            machine_list.append(env.ends_of_machine_occupancies[task.selected_machine] / env.benchmark_makespan)

        # assemble features to array
        features = np.concatenate((started_list, finished_list, machine_list))

        return features

    def switch_info_v2(env):
        """Infos on each task concatenated and structured for encoder policy. The order is equal to the order in
        self.tasks"""
        features = []

        tasks = copy.deepcopy(env.tasks)

        # find gap before task
        utilization_features = {}
        for m, machine_sequence in enumerate(env.machine_sequences):
            starts_ends = np.zeros((len(machine_sequence) + 1, 2))
            task_list = []
            for t, task in enumerate(machine_sequence):
                starts_ends[t + 1, 0] = task.started
                starts_ends[t + 1, 1] = task.finished
                task_list.append((task.task_index, task.job_index))

            gaps = starts_ends[1:, 0] - starts_ends[:-1, 1]
            cumulative_gaps = np.cumsum(gaps)
            with np.errstate(invalid='ignore'):  # ignore division by zero
                utilizations = cumulative_gaps / starts_ends[1:, 0]
            utilizations = np.nan_to_num(utilizations)

            for e, entry in enumerate(task_list):
                utilization_features[entry] = [gaps[e], utilizations[e]]

        # find machine utilizations before tasks

        for task in tasks:
            task_features = []
            task_features.append(task.started / env.benchmark_makespan)
            task_features.append(task.finished / env.benchmark_makespan)
            task_features.append(task.runtime / env.max_runtime)
            task_features.append(task.job_index / (env.num_jobs - 1))
            task_features.append(task.task_index / (env.num_tasks - 1))
            task_features.extend(task.machines)
            task_features.extend(utilization_features[(task.task_index, task.job_index)])
            features.append(task_features)

        return np.array(features)

    @classmethod
    def sensible_obs_self_att(cls, env):
        """sensible basic observation for next task information
        consists (per task) of:

        * processing time of the task
        * remaining processing time of the job
        * remaining processing steps in the job
        * remaining processing time of the next required machine
        * remaining processing steps of the next required machine
        * earliest starting point of the task
        * earliest finishing time of the task

         """

        next_tasks = env.get_next_tasks()
        remaining_processing_times_per_job, remaining_processing_steps_per_job, \
        remaining_processing_times_on_machines, remaining_processing_steps_on_machines, \
        earliest_starting_points_of_tasks, earliest_finishing_times_of_tasks = cls.get_normalized_features(env)

        obs = []
        for task in next_tasks:
            task_obs = []
            if task:
                task_obs.append(task.runtime / env.max_runtime)
                task_obs.append(remaining_processing_times_per_job[task.job_index])
                task_obs.append(remaining_processing_steps_per_job[task.job_index])
                task_obs.append(remaining_processing_times_on_machines[task.selected_machine])
                task_obs.append(remaining_processing_steps_on_machines[task.selected_machine])
                task_obs.append(earliest_starting_points_of_tasks[task.job_index])
                # task_obs.append(earliest_finishing_times_of_tasks[task.job_index])
            else:
                task_obs.extend([0] * 5)
                task_obs.extend([1] * 1)
            obs.append(task_obs)
        observation = np.array(obs)

        return observation

    @staticmethod
    def get_normalized_features(env):
        """
        helper fucntion to compute features for observations
        :param env:
        :return:
        """
        # processing_time_of_task
        remaining_processing_times_per_job = np.zeros(env.num_jobs)
        remaining_processing_steps_per_job = np.zeros(env.num_jobs)
        remaining_processing_times_on_machines = np.zeros(env.num_machines)
        remaining_processing_steps_on_machines = np.zeros(env.num_machines)
        earliest_starting_points_of_tasks = np.zeros(env.num_tasks)
        earliest_finishing_times_of_tasks = np.zeros(env.num_tasks)

        # feature assembly
        next_tasks = env.get_next_tasks()
        previous_tasks = env.get_previous_tasks()

        for task in env.tasks:
            if task.done:
                pass
            if not task.done:
                # machine infos
                remaining_processing_times_on_machines[np.argwhere(task.machines)] += task.runtime
                remaining_processing_steps_on_machines[np.argwhere(task.machines)] += 1
                # job infos
                remaining_processing_times_per_job[task.job_index] += task.runtime
                remaining_processing_steps_per_job[task.job_index] += 1
                if task in next_tasks:
                    # earliest_starting_point_of_task
                    if task.task_index == 0:
                        earliest_starting_point = 0
                    else:
                        earliest_starting_points_of_tasks[task.job_index] = \
                            max(env.ends_of_machine_occupancies[task.selected_machine],
                                previous_tasks[task.job_index].finished)

                    # earliest_finishing_time_of_task
                    earliest_finishing_times_of_tasks[task.job_index] = \
                        earliest_starting_points_of_tasks[task.job_index] + task.runtime

        # normalization
        remaining_processing_times_on_machines /= (env.num_jobs * env.max_runtime)
        remaining_processing_steps_on_machines /= (env.num_jobs)
        remaining_processing_times_per_job /= (env.num_tasks * env.max_runtime)
        remaining_processing_steps_per_job /= env.num_tasks
        if max(earliest_starting_points_of_tasks) - min(earliest_starting_points_of_tasks) == 0:
            earliest_starting_points_of_tasks = np.zeros_like(earliest_starting_points_of_tasks)
        else:
            earliest_starting_points_of_tasks = \
                (earliest_starting_points_of_tasks - min(earliest_starting_points_of_tasks)) / \
                (max(earliest_starting_points_of_tasks) - min(earliest_starting_points_of_tasks))
        if max(earliest_finishing_times_of_tasks) - min(earliest_finishing_times_of_tasks) == 0:
            earliest_finishing_times_of_tasks = np.zeros_like(earliest_finishing_times_of_tasks)
        else:
            earliest_finishing_times_of_tasks = \
                (earliest_finishing_times_of_tasks - min(earliest_finishing_times_of_tasks)) / \
                (max(earliest_finishing_times_of_tasks) - min(earliest_finishing_times_of_tasks))

        return remaining_processing_times_per_job, remaining_processing_steps_per_job, \
               remaining_processing_times_on_machines, remaining_processing_steps_on_machines, \
               earliest_starting_points_of_tasks, earliest_finishing_times_of_tasks

    def sensible_obs(env):
        """sensible basic observation for next task information
        consists (per task) of:

        * processing time of the task
        * remaining processing time of the job
        * remaining processing steps in the job
        * remaining processing time of the next required machine
        * remaining processing steps of the next required machine
        * earliest starting point of the task
        * earliest finishing time of the task

         """
        # processing_time_of_task
        remaining_processing_times_per_job = np.zeros(env.num_jobs)
        remaining_processing_steps_per_job = np.zeros(env.num_jobs)
        remaining_processing_times_on_machines = np.zeros(env.num_machines)
        remaining_processing_steps_on_machines = np.zeros(env.num_machines)
        earliest_starting_points_of_tasks = np.zeros(env.num_tasks)
        earliest_finishing_times_of_tasks = np.zeros(env.num_tasks)

        # feature assembly
        next_tasks = env.get_next_tasks()
        previous_tasks = env.get_previous_tasks()
        for task in env.tasks:
            if task.done:
                pass
            if not task.done:
                # machine infos
                remaining_processing_times_on_machines[np.argwhere(task.machines)] += task.runtime
                remaining_processing_steps_on_machines[np.argwhere(task.machines)] += 1
                # job infos
                remaining_processing_times_per_job[task.job_index] += task.runtime
                remaining_processing_steps_per_job[task.job_index] += 1
                if task in next_tasks:
                    # earliest_starting_point_of_task
                    if task.task_index == 0:
                        earliest_starting_point = 0
                    else:
                        earliest_starting_points_of_tasks[task.job_index] = \
                            max(env.ends_of_machine_occupancies[task.selected_machine],
                                previous_tasks[task.job_index].finished)

                    # earliest_finishing_time_of_task
                    earliest_finishing_times_of_tasks[task.job_index] = \
                        earliest_starting_points_of_tasks[task.job_index] + task.runtime

        # normalization
        remaining_processing_times_on_machines /= (env.num_jobs * env.max_runtime)
        remaining_processing_steps_on_machines /= (env.num_jobs)
        remaining_processing_times_per_job /= (env.num_tasks * env.max_runtime)
        remaining_processing_steps_per_job /= env.num_tasks
        if max(earliest_starting_points_of_tasks) - min(earliest_starting_points_of_tasks) == 0:
            earliest_starting_points_of_tasks = np.zeros_like(earliest_starting_points_of_tasks)
        else:
            earliest_starting_points_of_tasks = \
                (earliest_starting_points_of_tasks - min(earliest_starting_points_of_tasks)) / \
                (max(earliest_starting_points_of_tasks) - min(earliest_starting_points_of_tasks))
        if max(earliest_finishing_times_of_tasks) - min(earliest_finishing_times_of_tasks) == 0:
            earliest_finishing_times_of_tasks = np.zeros_like(earliest_finishing_times_of_tasks)
        else:
            earliest_finishing_times_of_tasks = \
                (earliest_finishing_times_of_tasks - min(earliest_finishing_times_of_tasks)) / \
                (max(earliest_finishing_times_of_tasks) - min(earliest_finishing_times_of_tasks))

        obs = []
        for task in next_tasks:
            if task:
                obs.append(task.runtime / env.max_runtime)
                obs.append(remaining_processing_times_per_job[task.job_index])
                obs.append(remaining_processing_steps_per_job[task.job_index])
                obs.append(remaining_processing_times_on_machines[task.selected_machine])
                obs.append(remaining_processing_steps_on_machines[task.selected_machine])
                obs.append(earliest_starting_points_of_tasks[task.job_index])
                obs.append(earliest_finishing_times_of_tasks[task.job_index])
            else:
                obs.extend([0] * 5)
                obs.extend([1] * 2)

        observation = np.array(obs)

        return observation


if __name__ == "__main__":
    # test instance for 3x4 with 5 machines and tools -> 12 tasks
    instance = []
    for i in range(12):
        instance.append(Task(2, 3, [1, 0, 1, 0, 1], [0, 1, 0, 0, 0], 6, 6, runtime=9))
    parser = ObservationParser()
    print(parser.parse(instance))
