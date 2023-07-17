from abc import abstractmethod

from .ArrayState import ArrayStateFloat


class FlowStateChecker(ArrayStateFloat):
    def __init__(self, nNodes, nEdges, pathState, demandsData):
        ArrayStateFloat.__init__(self, nEdges)
        self.nNodes = nNodes
        self.nEdges = nEdges
        self.pathState = pathState
        self.demandsData = demandsData

    def check(self):
        self.updateState()
        ArrayStateFloat.check(self)

    @abstractmethod
    def modify(self, src, dest, bw):
        pass

    def initialize(self):
        demand = self.pathState.nDemands
        while demand > 0:
            demand -= 1
            path = self.pathState.path(demand)
            pos = self.pathState.size(demand) - 1

            while pos > 0:
                pos -= 1
                src = path[pos]
                dest = path[pos + 1]
                self.modify(src, dest, self.demandsData.demandTraffics[demand])

    def updateState(self):
        pChanged = self.pathState.nChanged
        changed = self.pathState.changed
        while pChanged > 0:
            pChanged -= 1
            demand = changed[pChanged]
            bandwidth = self.demandsData.demandTraffics[demand]

            currentPath = self.pathState.path(demand)
            currentSize = self.pathState.size(demand)
            oldPath = self.pathState.oldPath(demand)
            oldSize = self.pathState.oldSize(demand)

            minSize = min(oldSize, currentSize)

            firstDiff = 1

            while firstDiff < minSize:
                if currentPath[firstDiff] != oldPath[firstDiff]:
                    break
                firstDiff += 1

            endCurrent = currentSize - 2
            endOld = oldSize - 2

            while (firstDiff < endCurrent) & (firstDiff < endOld) & (currentPath[endCurrent] == oldPath[endOld]):
                endCurrent -= 1
                endOld -= 1

            p = firstDiff - 1
            while p <= endCurrent:
                self.modify(currentPath[p], currentPath[p + 1], bandwidth)
                p += 1

            q = firstDiff - 1
            while q <= endOld:
                self.modify(oldPath[q], oldPath[q + 1], -bandwidth)
                q += 1


class FlowStateRecomputeDAG(FlowStateChecker):
    def __init__(self, nNodes, nEdges, sp, pathState, demandsData):
        FlowStateChecker.__init__(self, nNodes, nEdges, pathState, demandsData)
        self.sp = sp
        self.initialize()
        self.commitState()

    def modify(self, src, dest, bw):
        if src != dest:
            paths = self.sp.pathEdges[src][dest]
            nPaths = self.sp.nPaths[src][dest]
            increment = bw / nPaths
            for path in paths:
                for edge in path:
                    self.updateValue(edge, self.values[edge] + increment)


class FlowStateRecomputeDAGOnCommit(ArrayStateFloat):
    def __init__(self, nNodes, nEdges, sp, pathState, demandsData, edgeDemandState):
        ArrayStateFloat.__init__(self, nEdges)
        self.nNodes = nNodes
        self.nEdges = nEdges
        self.sp = sp
        self.pathState = pathState
        self.demandsData = demandsData
        self.edgeDemandState = edgeDemandState
        self.initialize()
        self.commit()

    def check(self):
        return True

    def updateState(self):
        pass

    def commit(self):
        self.updateFlowState()
        ArrayStateFloat.commit(self)

    def initialize(self):
        demand = self.pathState.nDemands
        while demand > 0:
            demand -= 1
            path = self.pathState.path(demand)
            pos = self.pathState.size(demand) - 1

            while pos > 0:
                pos -= 1
                src = path[pos]
                dest = path[pos + 1]
                self.modify(demand, src, dest, self.demandsData.demandTraffics[demand])

    def updateFlowState(self):
        pChanged = self.pathState.nChanged
        changed = self.pathState.changed
        while pChanged > 0:
            pChanged -= 1
            demand = changed[pChanged]
            bandwidth = self.demandsData.demandTraffics[demand]

            currentPath = self.pathState.path(demand)
            currentSize = self.pathState.size(demand)
            oldPath = self.pathState.oldPath(demand)
            oldSize = self.pathState.oldSize(demand)

            minSize = min(oldSize, currentSize)

            firstDiff = 1

            while firstDiff < minSize:
                if currentPath[firstDiff] != oldPath[firstDiff]:
                    break
                firstDiff += 1

            endCurrent = currentSize - 2
            endOld = oldSize - 2

            while (firstDiff < endCurrent) & (firstDiff < endOld) & (currentPath[endCurrent] == oldPath[endOld]):
                endCurrent -= 1
                endOld -= 1

            p = firstDiff - 1
            while p <= endCurrent:
                self.modify(demand, currentPath[p], currentPath[p + 1], bandwidth)
                p += 1

            q = firstDiff - 1
            while q <= endOld:
                self.modify(demand, oldPath[q], oldPath[q + 1], -bandwidth)
                q += 1

    def modify(self, demand, src, dest, bw):
        if src != dest:
            paths = self.sp.pathEdges[src][dest]
            nPaths = self.sp.nPaths[src][dest]
            increment = bw / nPaths
            for path in paths:
                for edge in path:
                    self.updateValue(edge, self.values[edge] + increment)
                    self.edgeDemandState.updateEdgeDemand(edge, demand, increment)
