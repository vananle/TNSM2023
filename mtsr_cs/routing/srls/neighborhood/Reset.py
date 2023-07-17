from ..core import Neighborhood


class Reset(Neighborhood):
    def __init__(self, pathState):
        self.pathState = pathState
        self.demand = -1
        self.source = -1
        self.destination = -1
        self.position = 0
        self.size = 0
        self.maxDetourSize = self.pathState.maxDetourSize

    def setNeighborhood(self, demand):
        self.demand = demand
        self.source = self.pathState.source(demand)
        self.destination = self.pathState.destination(demand)
        self.size = self.pathState.size(demand)
        self.neverTried = True

    def hasNext(self):
        return (self.size > 2) & (self.neverTried)

    def next(self):
        self.neverTried = False

    def apply(self):
        self.pathState.reset(self.demand)

    def saveBest(self):
        pass

    def applyBest(self):
        self.apply()
