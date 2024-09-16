"""
    Takes a JSSP instance and identifies the bottleneck machine.
"""
from typing import Dict,List
from .carlier1982 import CarlierBuB, SingleMachineJob   
from .longestpath import LongestPathDAG

class SolutionGraph:

    def __init__(self, tasks):

        self.machinesScheduled = [] # List of tasks already scheduled on the mth machine

        # nr of jobs equals the job index of the last task
        nrJobs = tasks[-1].job_index +1
        nrMachines = len(tasks[-1].machines)

        # Dummy start and end node + one node for each task (operation)
        nrNodes = 2 + nrJobs*nrMachines
        self.graph = [] # Graph as a (Task+1) x (Task+1) Matrix
        for _ in range(nrNodes):
            self.graph.append([None]*nrNodes)
        
        # Step 1) set edges for operations of the same job
        # sets the precedence constraints for each task on each machine
        # i.e sets the length of the edge from task i to i+1 to its duration.
        self.graph = self.__setJobEdges(nrJobs,self.graph,tasks)
        
        # Step 2) for operations already scheduled, set the edgeweights to all operations on the same machine 
        # that are not already scheduled. i.e. every operation can only take place after the last operation
        self.machinesScheduled = [[] for _ in range(nrMachines)]
        self.machinesOpen = [[] for _ in range(nrMachines)]
        for i,task in enumerate(tasks):
            if task.done == True:
                self.machinesScheduled[task.selected_machine].append(i)
            else:
                self.machinesOpen[task.selected_machine].append(i)

        # Sorts the tasks on every machine in order according to task-start, i.e. the sequence of tasks in list m
        # equals the sequence of fulfillment on that machine
        for m in self.machinesScheduled:
            if len(m) >0:
                m.sort(key=lambda x: tasks[x].started)
        
        # Sets precedence constrains for tasks that follow after each other on the same machine
        for m in self.machinesScheduled:
            if len(m) <= 1:
                continue

            for t in range(len(m)-1):
                taskNr = m[t]
                idxGraph = taskNr+1
                succTaskNr = m[t+1]
                idxGraphSucc = succTaskNr+1
                self.graph[idxGraph][idxGraphSucc] = tasks[taskNr].runtime

        # Finally defines that all remaining tasks on the machine can only start after the last
        # task has started
        for i,schedTasks in enumerate(self.machinesScheduled):
            if len(schedTasks)>0:
                idxInGraphScheduled = schedTasks[-1] +1
                lastTask = tasks[schedTasks[-1]]
                for o in self.machinesOpen[i]:
                    idxInGraphUnscheduled = o+1
                    self.graph[idxInGraphScheduled][idxInGraphUnscheduled] = lastTask.runtime

       # Up to this point: the matrix representing the edges in our graph contains
       # - edges resulting from precedence releations of a job (i.e. order in which tasks have to be fulfilled)
       # - edges from tasks already scheduled on the same machine
       # - edges from tasks that need to be scheduled on each machine and can only follow after each tasks on
       # the machine has been carried out.
    
    def appendTaskToMachine(self, task):

        """
         Appends a single task to the current task sequence on a machine. In doing so:
         - adds the task to "machinesScheduled"
         - removes the task from "machinesOpen"
         - adds the precedence edges to the graph
        """

        i = (task.job_index)*6+task.task_index

        
        # Adds task to set of machine that are already scheduled and removes it from "to be scheduled" tasks
        self.machinesScheduled[task.selected_machine].append(i)
        self.machinesOpen[task.selected_machine].remove(i)

        taskNr = i
        idxGraph = taskNr+1

        # Remaining tasks on that machine can only start after "task" has finished
        # Note: edge between last task on machine and newly added task ahs already been set by this loop in 
        # a previous iteration
        for o in self.machinesOpen[task.selected_machine]:
            idxInGraphUnscheduled = o+1
            self.graph[idxGraph][idxInGraphUnscheduled] = task.runtime
    
    def removeLastTaskFromMachine(self, m: int) -> None:

        taskNr = self.machinesScheduled[m].pop()
        idxGraph = taskNr+1

        # Delete edges between last task and unscheduled tasks on m
        for o in self.machinesOpen[m]:
            idxInGraphUnscheduled = o+1
            self.graph[idxGraph][idxInGraphUnscheduled] = None

        # Reinsert the task in set of unscheduled tasks on machine
        self.machinesOpen[m].append(taskNr)
    


        

    def __setJobEdges(self,nrJobs,graph,tasks):
        for idx in range(len(tasks)):

            idxInGraph = idx+1
            task = tasks[idx]
            if idxInGraph % nrJobs == 0:
                # last task of a job
                graph[idxInGraph][len(graph)-1] = task.runtime
                continue

            if idxInGraph % nrJobs == 1:
                # First task of a job
                graph[0][idxInGraph] = 0

            # duration of current operation --> edge to next operation of job
            graph[idxInGraph][idxInGraph+1] = task.runtime
        
        return graph


class BottleneckIdentification:

    def __init__(self, tasks):
        self.tasks = tasks
        self.longestPathLength = -1
        self.machineMakespan = []
        self.bottleneckMachine = -1
        self.machineMaximumLateness = [] # the maximum latenes on each machine, possibly <0 if all tasks can be scheduled in time
        self.machineMaximumLatenessTask = [] # the task on machine i having maximum lateness 
        self.latenessOfTask = [0]*len(self.tasks) # Vector of length len(tasks), storing the lateness of task i

    
    def identify(self, graph) -> None:

        # Steph 1) Generate Graph
        # Generates the graph including a precedence relations for tasks of jobs 
        # as well as precedence relations for tasks that are already scheduled and not yet scheduled
        
        longest = LongestPathDAG(graph,0)

        # Makespan = longest path from pseudonode to final pseudonode
        self.longestPathLength = longest.distTo[len(graph)-1]

        # Step 2) Solve the problem machinewise, i.e. determine the minimum makespan on each machine 
        # with heads and tail 
        nrMachines = len(self.tasks[-1].machines)
        machineTasks = [[] for _ in range(nrMachines)]
        for i,task in enumerate(self.tasks):
            machineTasks[task.selected_machine].append(i)

        

        largestMakespan = 0
        for m, assignedTasks in enumerate(machineTasks):

            # Solves the single machine problem for each machine in order to identify bottleneck + makespans
            singleMaschineJobs = self.__setupMachineJobs(assignedTasks, longest, graph)
            bub = CarlierBuB(singleMaschineJobs)
            singleMakespan = bub.solve()
            self.machineMakespan.append(singleMakespan)
            
            # Maximum Lateness and lateness related operations
            # Get the lateness of each task on the machine
            taskLatenesses = self.__getTaskLatenessInSchedule(self.longestPathLength, bub.bestSolution.schrageSequence)
            # get the maximum lateness on the machine
            self.machineMaximumLateness.append(max(taskLatenesses))
            # store the lateness of each task 
            for idx,entry in enumerate(bub.bestSolution.schrageSequence):
                job = entry[1]
                self.latenessOfTask[job.nr] = taskLatenesses[idx]
            
            

            
            # Update bottleneck machine
            if singleMakespan > largestMakespan:
                largestMakespan = singleMakespan
                self.bottleneckMachine = m
    
    def __setupMachineJobs(self, tasks: List[int], longestPathFromStart: LongestPathDAG, graph: List[List[int]]) -> Dict:
        """
            Sets up a set of SingleMachineJobs based on the current schedule/instance. Determines
            r = releasedate of task i
            q = tail (everything that takes place after finishing i)
            d = runtime of task i
            in order to feed it to the CarlierBuB Approach

            @param tasks: The set of taskindices from tasks on a machine
            @param longestPathFromStart: the class containing all distances from the source noed (0) to all other nodes
            @param graph: The graph representing the instance/current schedule
        """

        jobs = {}
        for i in tasks:
            task = self.tasks[i]
            nr = i
            d = task.runtime
            r = longestPathFromStart.distTo[nr+1] # L(0,i) longest Path to task (idx in graph +1 because of dummy startnode)
            l = LongestPathDAG(graph, nr+1)
            q = l.distTo[len(graph)-1] - task.runtime# longest path from nr to End (minus d_j)

            j = SingleMachineJob(nr,q,d,r) # Nr of job (=nr of task)
            jobs[nr] = j
        
        return jobs

    def __getTaskLatenessInSchedule(self,longestPath, sequence) -> List[int]:
        lateness = []

        for e in sequence:
            # The duedate of a job within the onemachine problem is:
            # L(0,n) - L(i,n) + d_i
            start = e[0]
            job = e[1]
            end = start + job.d
            lin = job.q + job.d
            duedate = longestPath-lin+job.d
            lateness.append(end-duedate)

        return lateness