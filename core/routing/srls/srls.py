import itertools
import time

import numpy as np

from .constraint import MaxLoad
from .demand import DemandsData
from .neighborhood import *
from .state import *

sys.path.append('../../')
from core.routing.solver import Solver
from core.routing.te_util import edge_in_segment


class SRLS(Solver):
    def __init__(self, graph, sp, capacity, segments, timeout, verbose):
        super(SRLS, self).__init__(graph, timeout, verbose)

        labels = []
        srcs = []
        dest = []
        bws = []
        for i in range(self.num_node):
            for j in range(self.num_node):
                if i != j:
                    srcs.append(i)
                    dest.append(j)
                    bws.append(0)

        decisionDemands = DemandsData(labels, srcs, dest, bws)

        self.sp = sp
        self.nDemands = decisionDemands.nDemands
        self.capacity = capacity
        self.decisionDemands = decisionDemands
        self.edgeDemandState = EdgeDemandStateTree(self.nDemands, self.num_edge, self.capacity)
        self.pathState = PathState(decisionDemands)
        self.flowState = FlowStateRecomputeDAG(self.num_node, self.num_edge, sp, self.pathState, decisionDemands)
        self.flowStateOnCommit = FlowStateRecomputeDAGOnCommit(self.num_node, self.num_edge, sp, self.pathState,
                                                               decisionDemands,
                                                               self.edgeDemandState)
        self.maxLoad = MaxLoad(self.num_node, self.num_edge, capacity, self.flowState, sp)
        self.bestPaths = SavedState(self.pathState)

        self.pathState.addTrial(self.flowState)
        self.pathState.addTrial(self.flowStateOnCommit)
        self.pathState.addTrial(self.maxLoad)
        self.pathState.addTrial(self.bestPaths)

        self.neighborhoods = [Reset(self.pathState), Remove(self.pathState), Insert(self.num_node, self.pathState),
                              Replace(self.num_node, self.pathState)]
        self.kickNeighborhoods = [Reset(self.pathState), Remove(self.pathState)]

        self.segments = segments

    def selectDemand(self):
        edge = self.maxLoad.selectRandomMaxEdge()
        return self.edgeDemandState.selectRandomDemand(edge)

    def visitNeighborhood(self, neighborhood, setter):
        nBestMoves = 0
        bestNeighborhoodLoad = 999999999.0
        improvementFound = False
        neighborhood.setNeighborhood(setter)
        while neighborhood.hasNext():
            neighborhood.next()
            neighborhood.apply()

            if self.pathState.nChanged > 0 & self.pathState.check():
                score = self.maxLoad.score()

                if score == bestNeighborhoodLoad:
                    nBestMoves += 1
                    if random.randint(0, nBestMoves) == 0:
                        neighborhood.saveBest()
                else:
                    if score < bestNeighborhoodLoad:
                        nBestMoves = 1
                        improvementFound = True
                        neighborhood.saveBest()
                        bestNeighborhoodLoad = self.maxLoad.score()
                self.pathState.revert()

        return improvementFound

    def kick(self, demand):
        choice = random.randint(0, self.kickNeighborhoods)
        neighborhood = self.kickNeighborhoods[choice]
        if self.visitNeighborhood(neighborhood, demand):
            neighborhood.applyBest()
            self.pathState.update()
            self.pathState.commit()

    def startMoving(self):
        startTime = time.time()
        bestLoad = self.maxLoad.score()
        nIterations = 0
        bestIteration = 0
        while time.time() - startTime < self.timeout:

            nIterations += 1
            if (self.maxLoad.score() > bestLoad) & (nIterations > (bestIteration + 1000)):
                self.bestPaths.restorePath()
                self.pathState.update()
                self.pathState.commit()
                bestIteration = nIterations - 1

            demand = self.selectDemand()

            if (self.maxLoad.score == bestLoad) & (nIterations > (bestIteration + 3)):
                bestIteration = nIterations

                self.kick(demand)

            improvementFound = False
            pNeighborhood = 0

            while (improvementFound is False) and (pNeighborhood < len(self.neighborhoods)):
                neighborhood = self.neighborhoods[pNeighborhood]
                improvementFound = self.visitNeighborhood(neighborhood, demand)

                if improvementFound:

                    neighborhood.applyBest()
                    self.pathState.update()
                    self.pathState.commit()

                    if self.maxLoad.score() < bestLoad:
                        self.bestPaths.savePath()
                        bestLoad = self.maxLoad.score()
                        bestIteration = nIterations

                pNeighborhood += 1

    def solve(self, tm, solution=None, eps=1e-12):
        try:
            self.modifierTrafficMatrix(tm)
            self.startMoving()

            self.bestPaths.restorePath()
            self.pathState.update()
            self.pathState.commit()
        except:
            print('ERROR in p2_srls_solver --> pass')
        solution = self.extractRoutingPath()

        return solution

    def modifierDemands(self, newDemand):
        diffs = []
        for i in range(len(self.decisionDemands.demandTraffics)):
            diffs.append(newDemand.demandTraffics[i] - self.decisionDemands.demandTraffics[i])
            self.decisionDemands.demandTraffics[i] += diffs[i]
        for demand in range(self.nDemands):
            path = self.pathState.path(demand)
            pDetour = self.pathState.size(demand) - 1

            while pDetour > 0:
                pDetour -= 1
                src = path[pDetour]
                dest = path[pDetour + 1]
                self.flowState.modify(src, dest, diffs[demand])
                self.flowStateOnCommit.modify(demand, src, dest, diffs[demand])
        self.flowState.update()
        self.flowStateOnCommit.update()
        self.flowStateOnCommit.commit()
        self.flowState.commit()
        self.maxLoad.initialize()
        self.maxLoad.commit()

    def modifierTrafficMatrix(self, tf):
        labels = []
        srcs = []
        dest = []
        bws = []

        for i in range(len(tf)):
            for j in range(len(tf)):
                if i != j:
                    srcs.append(i)
                    dest.append(j)
                    bws.append(tf[i][j])
        demandData = DemandsData(labels, srcs, dest, bws)
        self.modifierDemands(demandData)

    def init_solution(self):
        init_solution = np.zeros(shape=(self.num_node, self.num_node, self.num_node))
        for src in range(self.num_node):
            for dst in range(self.num_node):
                init_solution[src, dst, src] = 1

        return init_solution

    def extractRoutingPath(self):
        paths = []
        for i in range(self.num_node):
            A = []
            for j in range(self.num_node):
                A.append([])
            paths.append(A)

        for i in range(self.pathState.nDemands):
            paths[self.pathState.source(i)][self.pathState.destination(i)] = self.pathState.paths[i].currentPath

        self.routing_mx = self.getRoutingMatrix(paths)

        return self.conver_solution(paths)

    def conver_solution(self, solutions):
        converted_solution = np.zeros(shape=(self.num_node, self.num_node, self.num_node))
        for src in range(self.num_node):
            for dst in range(self.num_node):
                if src == dst:
                    converted_solution[src, dst, src] = 1
                if len(solutions[src][dst]) == 2:
                    converted_solution[src, dst, src] = 1
                if len(solutions[src][dst]) > 2:
                    converted_solution[src, dst, int(solutions[src][dst][1])] = 1

        return converted_solution

    def evaluate(self, tm, solution):
        # extract utilization

        values = [0] * self.num_edge
        for i, j, k in itertools.product(range(self.num_node), range(self.num_node), range(self.num_node)):
            if solution[i, j, k] > 0 and i != j:
                path_ik = self.sp.pathEdges[i][k]
                path_kj = self.sp.pathEdges[k][j]
                #
                if i != k:
                    for edge in path_ik[0]:
                        values[edge] += tm[i, j] / self.capacity.capacity[edge]
                if k != j:
                    for edge in path_kj[0]:
                        values[edge] += tm[i, j] / self.capacity.capacity[edge]
        return max(values)

    def getRoutingMatrix(self, routingSolution):
        nLinks = self.num_edge
        nNodes = self.num_node
        routingMatrix = np.zeros((nLinks, nNodes ** 2))
        for i in range(nNodes):
            for j in range(nNodes):
                if i != j:
                    for k in range(len(routingSolution[i][j]) - 1):
                        n = routingSolution[i][j][k]
                        m = routingSolution[i][j][k + 1]
                        for path in self.sp.pathEdges[m][n]:
                            for link in path:
                                routingMatrix[link][i * nNodes + j] = 1
        return routingMatrix

    def getLinkload(self, tm, solution):
        values = [0] * self.num_edge
        routingMatrix = np.zeros((self.num_edge, self.num_node ** 2))

        for i, j, k in itertools.product(range(self.num_node), range(self.num_node), range(self.num_node)):
            if solution[i, j, k] > 0 and i != j:
                path_ik = self.sp.pathEdges[i][k]
                path_kj = self.sp.pathEdges[k][j]
                #
                if i != k:
                    for edge in path_ik[0]:
                        values[edge] += tm[i, j] / self.capacity.capacity[edge]
                        routingMatrix[edge][i * self.num_node + j] = 1

                if k != j:
                    for edge in path_kj[0]:
                        values[edge] += tm[i, j] / self.capacity.capacity[edge]
                        routingMatrix[edge][i * self.num_node + j] = 1

        return values, routingMatrix
