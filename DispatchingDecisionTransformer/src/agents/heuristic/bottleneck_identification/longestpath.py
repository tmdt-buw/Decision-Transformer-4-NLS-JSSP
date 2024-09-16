"""

    Longest Path in a directed acyclic graph.
    The structure is based on Sedgewick's Implementation, see
    https://algs4.cs.princeton.edu/44sp/AcyclicLP.java.html

"""

from typing import List
from sys import maxsize as MAX_SIZE

class DepthFirstOrder:

    def __init__(self, graph : List[List[int]]):
        """
        @param graph: NxN Matrix of distances (distance i,j = None implies that there is no edge between i and j)
        """
        self.graph = graph
        self.preCounter = 0
        self.postCounter = 0
        self.pre = [0 for n in range(len(graph))]
        self.post = [0 for n in range(len(graph))]
        self.postorder = []
        self.preorder = []
        self.marked = [False for v in range(len(graph))]
        for n in range(len(graph)):
            if not self.marked[n]:
                self.__dfs(graph,n)
        
    def __dfs(self, graph : List[List[int]], n : int):
        """
        Recursive depth first implementation
        @param graph: NxN Distance Matrix (None-entry implies that there is no edge between i and j)
        @param n: node index to be considered
        """

        self.marked[n] = True
        self.preCounter +=1 
        self.pre[n] = self.preCounter
        self.preorder.append(n)

        for o,w in enumerate(graph[n]):
            if w is None:
                # no edge between o and n exists
                continue
            if self.marked[o]:
                continue
            self.__dfs(graph, o)

        self.postorder.append(n)
        self.postCounter += 1
        self.post[n] = self.postCounter
    
    def reversePost(self):
        rev = self.postorder[:]
        rev.reverse()
        return rev
        
        

class Topological:
    """
        Skips all feasibility checks from Sedgewick and directly implements "DepthFirstOrder".
        Crashes if the graph has cycles.

        TODO: Check if there exists a topological order.
    """

    def __init__(self, graph : List[List[int]]):
        self.graph = graph

    def order(self):
        """
            Returns a list representing the topological order of nodes in a graph.
        """
        dfso = DepthFirstOrder(self.graph)
        return dfso.reversePost()



class LongestPathDAG:

    def __init__(self, graph : List[List[int]], s : int):
        """
        Calculates the longest path from a given startnode s to every other node in the graph.
        @param graph: NxN Matrix of distances (distance i,j = None implies that there is no edge between i and j)
        @param s: index of the source node
        """
        self.graph = graph
        # distTo[v] = distance  of longest s->v path
        self.distTo = [-MAX_SIZE for n in range(len(graph))] 
        self.distTo[s] = 0
        
        # edgeTo[v] = last edge on longest s->v path
        self.edgeTo = [None for n in range(len(graph))]

        # Determine topological order
        top = Topological(graph=graph)
        order = top.order()

        # Determines the longest paths to each node in topological order
        for i in order:
            for j,w in enumerate(self.graph[i]):
                if w is None:
                    # No edge between i and j exists
                    continue
                self.relax(i,j)
        
    def relax(self,i: int, j: int):

        # checks if the length to j from i is larger than the best known value
        weight = self.graph[i][j]
        if(self.distTo[j] < self.distTo[i] + weight):
            self.distTo[j] = self.distTo[i] + weight
            self.edgeTo[j] = i


# Unit testing
if __name__ == '__main__':
    # UNIT TESTING
    # Instance from Carlier(1982) as DAG
    graph = [
        [None,10,13,11,20,30,0,30,None], # Knoten 0 (Pseudo)
        [None,None,5,None,None,None,None,None,12], # Knoten 1
        [None,None,None,6,None,None,None,None,32], # Knoten 2
        [None,None,None,None,7,None,None,None,31], # Knoten 3
        [None,None,None,None,None,4,None,None,25], # Knoten 4
        [None,None,None,None,None,None,None,3,11], # Knoten 5
        [None,6,None,None,None,None,None,None,23], # Knoten 6
        [None,None,None,None,None,None,None,None,2], # Knoten 7
        [None,None,None,None,None,None,None,None,None], # Knoten 8 (Pseudo)
    ]


    top = Topological(graph)
    print(top.order())

    d = LongestPathDAG(graph,0)
    print(d.distTo)