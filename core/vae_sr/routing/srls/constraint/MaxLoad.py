from ..state import Trial, Objective


class MaxLoad(Trial, Objective):
    def __init__(self, nNodes, nEdges, capacityData, flowState, sp):
        self.nNodes = nNodes
        self.nEdges = nEdges
        self.capacityData = capacityData
        self.flowState = flowState
        self.sp = sp
        self.nMaxLoad = 0
        self.maxLoad = 0.0
        self.maxEdge = 0
        self.initialize()
        self.oldNMaxload = self.nMaxLoad
        self.oldMaxload = self.maxLoad
        self.oldMaxEdge = self.maxEdge

    def initialize(self):
        self.nMaxLoad = 0
        self.maxLoad = 0.0

        flow = self.flowState.values
        edge = self.nEdges
        while edge > 0:
            edge -= 1
            load = flow[edge] * self.capacityData.invCapacity[edge]
            if load > self.maxLoad:
                self.maxEdge = edge
                self.maxLoad = load
                self.nMaxLoad = 1
            else:
                if load == self.maxLoad:
                    self.nMaxLoad += 1

    def selectRandomMaxEdge(self):
        return self.maxEdge

    def update(self):
        flow = self.flowState.values
        oldFlow = self.flowState.oldValues
        changed = self.flowState.deltaElements
        p = self.flowState.nDelta

        while p > 0:
            p -= 1
            edge = changed[p]
            if flow[edge] != oldFlow[edge]:
                load = flow[edge] * self.capacityData.invCapacity[edge]
                oldLoad = oldFlow[edge] * self.capacityData.invCapacity[edge]

                if load > self.maxLoad:
                    self.maxEdge = edge
                    self.maxLoad = load
                    self.nMaxLoad = 1
                else:
                    if load == self.maxLoad:
                        self.nMaxLoad += 1
                    else:
                        if oldLoad == self.maxLoad:
                            self.nMaxLoad -= 1
        if self.nMaxLoad == 0:
            self.initialize()

    def commit(self):
        self.oldMaxload = self.maxLoad
        self.oldNMaxload = self.nMaxLoad
        self.oldMaxEdge = self.maxEdge

    def revert(self):
        self.maxLoad = self.oldMaxload
        self.nMaxLoad = self.oldNMaxload
        self.maxEdge = self.oldMaxEdge

    def check(self):
        self.update()
        improved = (self.maxLoad < self.oldMaxload)
        if improved == False:
            self.revert()
        return improved

    def score(self):
        return self.maxLoad
