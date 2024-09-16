from pathlib import Path

# Functional imports
import os
import pickle
from src.data_generator.task import Task
import numpy as np


# Constants
TAILLARD_RAW_DIRECTORY: Path = Path(__file__).parent.parent.parent / 'data' / 'benchmark_instances' / 'taillard_raw'
TAILLARD_INSTANCES_DIRECTORY: Path = Path(__file__).parent.parent.parent / 'data' / 'benchmark_instances' / 'taillard'


class TaillardFactory:
    """
    Loads instances from Taillard file
    :param taillard_num: number of the instance (1-10)
    """

    @classmethod
    def load_instances_from_data(cls, taillard_num: int, write_to_file: bool = True):

        with open(TAILLARD_RAW_DIRECTORY / f"ta{taillard_num}.txt") as handle:
            data = handle.read()
        data_arr = data.split()

        # check the problem size
        if taillard_num in range(1, 11):
            num_jobs = 15
            num_machines = 15
        elif taillard_num in range(11, 21):
            num_jobs = 20
            num_machines = 15
        elif taillard_num in range(21, 31):
            num_jobs = 20
            num_machines = 20
        elif taillard_num in range(31, 41):
            num_jobs = 30
            num_machines = 15
        elif taillard_num in range(41, 51):
            num_jobs = 30
            num_machines = 20
        elif taillard_num in range(51, 61):
            num_jobs = 50
            num_machines = 15
        elif taillard_num in range(61, 71):
            num_jobs = 50
            num_machines = 20
        elif taillard_num in range(71, 81):
            num_jobs = 100
            num_machines = 20
        else:
            raise TypeError("Taillard_num needs to be between 1 and 80")

        # Create arrays for runtimes and task-machine mapping
        for index, value in enumerate(data_arr):
            if value == 'Times':
                runtimes_index = index
            if value == 'Machines' and index >= 14:
                machines_index = index

        runtimes_all = data_arr[runtimes_index+1:machines_index]

        machine_map_all = data_arr[machines_index+1:]

        # Slice runtimes after num_jobs values and convert it to 2D array
        # [[j0_t0_runtime,..][jn_tn_runtime,...]]
        runtimes = np.zeros((num_jobs, num_machines))
        job_counter = 0
        last_index = 0
        for index, value in enumerate(runtimes_all):
            # index > 0 is needed cause 0 mod 0 is 0
            if (index + 1) % num_machines == 0 and index > 0:
                runtimes[job_counter] = runtimes_all[last_index:(index+1)]
                last_index = (index +1)
                job_counter += 1

        # Slice task-machine values after num_jobs and convert it to 2D array
        # [[j0,t0_machine_num,..][jn,tn_machine_num]]
        machine_map = np.zeros((num_jobs, num_machines))
        job_counter = 0
        last_index = 0
        for index, value in enumerate(machine_map_all):
            if (index+1) % num_machines == 0 and index > 0:
                machine_map[job_counter] = machine_map_all[last_index:(index+1)]
                last_index = (index +1)
                job_counter += 1

        # create instances
        instance = []
        num_tasks = num_machines
        num_tools = 0
        instance_hash = hash(tuple(instance))

        def to_one_hot(x: int, max_size: int) -> np.array:
            """
            Convert to One Hot encoding
            :param x: Index which value should be 1
            :param max_size: Size of the one hot encoding vector
            :return: One hot encoded vector
            """
            one_hot = np.zeros(max_size)
            one_hot[x] = 1

            return one_hot

        # for machines: -1 is required because our machine logic includes Machine 0, taillard instances which
        # is not the case in taillard instances
        for j in range(num_jobs):
            for t in range(num_tasks):
                task = Task(
                    job_index=j,
                    task_index=t,
                    machines=to_one_hot(int(machine_map[j][t]-1), num_machines).astype(int).tolist(),
                    tools=[],
                    deadline=0,
                    done=0,
                    finished=0,
                    runtime=int(runtimes[j][t]),
                    instance_hash=hash(tuple(runtimes.flatten())),
                    _n_machines=num_machines,
                    _n_tools=num_tools,
                    started=0,
                    selected_machine=int(machine_map[j][t]-1)
                )
                task._deadline=0
                task.running=0
                instance.append(task)

        instance = [instance]

        if write_to_file:
            if not os.path.exists(TAILLARD_INSTANCES_DIRECTORY / f"{num_jobs}x{num_machines}"):
                os.makedirs(TAILLARD_INSTANCES_DIRECTORY / f"{num_jobs}x{num_machines}")
            with open(TAILLARD_INSTANCES_DIRECTORY / f"{num_jobs}x{num_machines}" / f"ta{taillard_num}.pkl", 'wb') as handle:
                pickle.dump(instance, handle, protocol=pickle.HIGHEST_PROTOCOL)


# call function to create taillard instances
for i in range(1, 81):
    TaillardFactory.load_instances_from_data(i)
print("Taillard instances have been generated.")

