from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from sys import maxsize as MAX_SIZE
from typing import Dict, List

"""

    Notation:
        - N*: Set of unscheduled operations on the machine to be scheduled
        - d_i: duration of operation i
        - t_i: starttime of operation i
        - r_i: "head" of a job i.e. it's releasedate that results from scheduled predecessors on other machines
        - q_i: "tail" of a job i.e. duration of operations that follow the job
        - E_k: set of pairs of operations performed on machine k
    
    Resulting MIP:
        min t_n

        s.t.

        tn >= t_i + d_i + q_i   forall i in N*
        t_i >= r_i              forall i in N*
        t_j - t_i >= d_i or t_i - t_j >= d_j      forall (i,j) in E_k
        t_i >= 0
"""

@dataclass
class SingleMachineJob:
    nr: int # The job's id
    q: int # The job's tail on machine k
    d: int # The job's duration on machine k
    r: int # The job's releasedate on machine k


class SchrageSchedule:

    """
        A simple Upperbound based on the "Most work remaining" idea.
        Also known as a "Schrage Schedule".

        During the solutionprocess the critical path, i.e. the set of jobs that
        define the makespan, is determined "on the fly".

        Note: the implementation does not follow the O(n log n) implementation
        of Carlier (1982)
        
    """

    def __init__(self):
        self.schrageSequence = []
        self.schrageMakeSpan = 0
        self.critialJobset = []


    def solve(self, jobs: Dict) -> None:

        # Initialization: Copy the jobs and empty lists in order to avoid reference errors.
        unschedJobs = [jobs[j] for j in jobs]
        self.schrageSequence = []
        self.schrageMakeSpan = 0

        # Stores the longest path and its length for each job in the sequence
        longestPathTo = {}

        # Sort jobs based on their releasedates ("a_i" in Carlier(1982) = "r_i" here)
        unschedJobs = sorted(unschedJobs, key=lambda x: x.r)
        lastJobInCritialPath = unschedJobs[0].nr

        # Initialize t: smallest releasedate (first entry in the sorted list)
        t = unschedJobs[0].r
        
        while(len(unschedJobs)>0):

            # Among jobs with a releasedate <=t, determine the job having the largest q_i
            j = self.__getNextJob(unschedJobs, t)
            longestPathTo[j.nr] = {"pred": None, "length": None}
            self.__updateLongestPath(longestPathTo, j, self.schrageSequence)
            self.schrageSequence.append((t,j))

            # Check if SchrageMakespan and set of critical jobs can be updated
            # This is the case, if t + d_j + q_j > current Makespan
            if t + j.d + j.q >= self.schrageMakeSpan:
                # TODO: Verhindert diese Prüfung "maximale Knotenmengen (maximal sets?)" ?
                # Note: ">=" allows to select "maximal sets" of jobs on the critical path
                self.schrageMakeSpan = t + j.d + j.q
                lastJobInCritialPath = j.nr

            if len(unschedJobs) == 0:
                break

            # Update the time, i.e. set the start of the next job
            t = max(t + j.d, unschedJobs[0].r)
        
        # Finally determine the critical path and the nodes on it
        self.critialJobset = self.__criticalJobSet(lastJobInCritialPath, longestPathTo)
    
    def __criticalJobSet(self, lastJob, longestPaths):
        """
            Recursively determines the set of critical jobs indices.
        """

        critJob = [lastJob]
        job = lastJob
        while(longestPaths[job]["pred"] is not job):
            job = longestPaths[job]["pred"]
            critJob.append(job)
        critJob.reverse()
        return critJob
        

    def __updateLongestPath(self, longestPathTo : Dict, nextJob: SingleMachineJob, sequence : List[int]):
        """
            We solve the DP to determine the longest path while generating the SchrageSequence.
            Whenever we add a new job, we have two options determining the longest path to that jobnode
            in the associated directed graph. 

            Either the longest path corresponds to the direct edge from 0 to the node "j", i.e.
            the releasedate of j is larger than the makespan of the current sequence.

            Or (if t > r_j) the job is released before the sequence has finished and hence can 
            only start after the finish of the last job j' of the incumbent sequence. 
            In this case the longest path to j' plus d_j' determines the longest path to j.

            Implementation is as follow:
            For each job in the incumbent schedule, we save the length of the longest path to that job as
            well as the direct predecessor on that path. If the predecessor is the job itself, it is
            implied, that the releasedate determines the longest path.

            @param longestPathTo: Dictionary with entrys "predecesso" and "length" (of the longest known path) for each job(index)
            @param nextJob: Index of the next job to be appended to the SchrageSchedule
            @param sequence: the current schrageSequence
        """

        if(len(sequence) == 0):
            longestPathTo[nextJob.nr]["pred"] = nextJob.nr
            longestPathTo[nextJob.nr]["length"] = nextJob.r
            return

        lastJob = sequence[-1][1]
        lengthFromLastJob = longestPathTo[lastJob.nr]["length"] + lastJob.d
        if lengthFromLastJob < nextJob.r:
            # The releasedate determines the longest path to the next job
            longestPathTo[nextJob.nr]["pred"] = nextJob.nr
            longestPathTo[nextJob.nr]["length"] = nextJob.r
        else:
            
            longestPathTo[nextJob.nr]["pred"] = lastJob.nr
            longestPathTo[nextJob.nr]["length"] = lengthFromLastJob




    def __getNextJob(self, unschedJobs: List[SingleMachineJob], t : int) -> SingleMachineJob:

        """
            Among jobs with a releasedate <= t, determines the job having
            the largest q_i. Assumes that the jobs in unschedJobs are sorted
            according to increasing releasedates
        """

        maxQ = 0
        j = 0
        for i,job in enumerate(unschedJobs):

            if job.r > t:
                break

            if job.q > maxQ:
                j = i
                maxQ = job.q

        return unschedJobs.pop(j)

class Node:
    def __init__(self):
        self.lb = 0
        self.jobs = []
    
    def __lt__(self,other):
        return self.lb > other.lb
    def __eq__(self,other):
        return self.lb == other.lb
    def __ne__(self,other):
        return not(self.lb == other.lb)
    def __gt__(self,other):
        return self.lb < other.lb
    
    def copy(self):
        # Creates a deepcopy of the class (the jobs are still references)
        c = Node()
        c.lb = self.lb
        c.jobs = self.jobs.copy()
        return c

class CarlierBuB:

    def __init__(self, jobs : List[SingleMachineJob]):
        self.upperBound = MAX_SIZE
        self.jobs = jobs
        self.searchTree = []
        self.bestSolution = ""
    
    def solve(self):

        # Initialisation:
        n = Node()
        n.jobs = self.jobs.copy()
        n.lb = 0

        self.searchTree = [n]

        while(len(self.searchTree)>0):

            # Set next node
            n = self.searchTree.pop()

            childNodes = self.__branch(n)

            for child in childNodes:
                self.__addNodeToSearchTree(child)

        return self.upperBound


    def __addNodeToSearchTree(self,n: Node) -> None:
        
        index = bisect_left(self.searchTree, n)
        self.searchTree.insert(index,n)

    def pruneSearchTree(self):
        """
            Removes the nodes in the search tree is a lowerbound >= to the
            current upperbound.

            @param searchTree: The search tree of the BuB procedure
        """
        pseudoNode = Node()
        pseudoNode.lb = self.upperBound
        index = bisect_right(self.searchTree, pseudoNode)
        if index == len(self.searchTree):
            self.searchTree = []
        else:
            self.searchTree = self.searchTree[index:]

    def __branch(self,n : Node) -> List[Node]:
        """
        @param n: node to be branched on
        @param searchTree: search tree of the BuB Procedure
        """
        childNodes = []

        # Generate the schrageschedule based on node n
        s = SchrageSchedule()
        s.solve(n.jobs)
        
        L = s.schrageMakeSpan
        critJobs = s.critialJobset
        
        # Update best bound if possible
        if(L < self.upperBound):
            self.bestSolution = s
            self.upperBound = L
            self.pruneSearchTree()

        # Determine c and J (a list of job indices on critical path)
        # if no c exists --> the node represents an optimum solution
        c, J = self.__getCAndJ(critJobs, n.jobs)
        if c is None: 
            # No further branching, schragesolution is optimal for given node
            # Branching cannot result in a better solution
            return childNodes
        
        # Otherwise: Generate two childnodes:
        # - in Node a : c is processed before every job in J
        # - in Node b : c is processed after every job in J

        cJob = n.jobs[c]
        sumD = sum([n.jobs[j].d for j in J])

        a = n.copy()
        newQ = max(cJob.q, sumD+n.jobs[J[-1]].q)
        modJob = SingleMachineJob(nr=cJob.nr, r=cJob.r,d=cJob.d,q=newQ)
        a.jobs[c] = modJob
        a.lb = self.__getLowerbound(n.lb, c, J, a.jobs)
        if(a.lb < self.upperBound):
            childNodes.append(a)

        b = n.copy()
        newR = max(cJob.r, min([n.jobs[j].r for j in J]) + sumD)
        modJob = SingleMachineJob(nr=cJob.nr,r=newR,d=cJob.d, q=cJob.q)
        b.jobs[c] = modJob
        b.lb = self.__getLowerbound(n.lb, c, J, b.jobs)
        if(b.lb < self.upperBound):
            childNodes.append(b)

        return childNodes

    def __getCAndJ(self, critialJobSequence : List[int], jobs : Dict):
        """
        Carlier (1982) p. 44 (Proof of theorem)
        Job c is the job at the highest position in the sequence where q_c < q_p
        with q_p being the tail of the last job in the sequence.

        If no such job exists, the sequence is optimal.

        @param criticalJobSequence: List of indicies of the critical jobs in the list "jobs"
        @param jobs: List of all singlemachine jobs for the given instance
        """
        if(len(critialJobSequence)==1):
            # Only a single job in the critial jobsequence
            return None, critialJobSequence

        c = None
        J = critialJobSequence
        q_p = jobs[critialJobSequence[-1]].q
        for i in range(len(critialJobSequence)-1):
            job = jobs[critialJobSequence[i]]
            if job.q < q_p:
                c = job.nr 
                J = critialJobSequence[i+1:] # jobs after c in the sequence

        return c,J


    def __getLowerbound(self,L, c, J, jobs):

        jLb = self.__jobSetLowerbound(J,jobs)
        candJLb = self.__jobSetLowerbound(J+[c],jobs)
        return max(L,jLb,candJLb)
    
    def __jobSetLowerbound(self, jobIdx, jobs):
        min_ai = MAX_SIZE
        min_qi = MAX_SIZE
        sum_di = 0
        for i in jobIdx:
            j = jobs[i]
            min_ai = j.r if j.r < min_ai else min_ai
            min_qi = j.q if j.q < min_qi else min_qi
            sum_di += j.d
        
        return min_ai + sum_di + min_qi

if __name__ == '__main__':
    # UNIT TESTING
    # Instance from Carlier(1982)
    jobs = {
        1 : SingleMachineJob(nr=1,r=10, d=5, q=7),
        2 : SingleMachineJob(nr=2,r=13,d=6,q=26),
        3 : SingleMachineJob(nr=3,r=11,d=7,q=24),
        4 : SingleMachineJob(nr=4,r=20,d=4,q=21),
        5 : SingleMachineJob(nr=5,r=30,d=3,q=8),
        6 : SingleMachineJob(nr=6,r=0,d=6,q=17),
        7 : SingleMachineJob(nr=7,r=30,d=2,q=0),
    }


    bub = CarlierBuB(jobs)
    makespan = bub.solve()