from ..core import Neighborhood


class Remove(Neighborhood):
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
        self.position = 0
        self.size = self.pathState.size(demand)
        self.storedPosition = 0

    def hasNext(self):
        return (self.size > 2) & (self.position < self.size - 2)

    def next(self):
        self.position += 1

    def apply(self):
        self.pathState.remove(self.demand, self.position)

    def saveBest(self):
        self.storedPosition = self.position

    def applyBest(self):
        self.position = self.storedPosition
        self.apply()
