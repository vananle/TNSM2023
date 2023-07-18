from ..core import Neighborhood


class Replace(Neighborhood):
    def __init__(self, nNodes, pathState):
        self.nNodes = nNodes
        self.pathState = pathState
        self.demand = -1
        self.source = -1
        self.destination = -1
        self.position = 0
        self.node = 0
        self.size = 0
        self.maxDetourSize = self.pathState.maxDetourSize

    def setNeighborhood(self, demand):
        self.demand = demand
        self.source = self.pathState.source(demand)
        self.destination = self.pathState.destination(demand)
        self.position = 0
        self.node = self.nNodes - 1
        self.size = self.pathState.size(demand)
        self.storedPosition = 0
        self.storedNode = self.node

    def hasNext(self):
        return ((self.node == self.nNodes - 1) & (self.position == self.size - 2)) == False

    def next(self):
        self.node += 1
        if self.node >= self.nNodes:
            self.node = 0
            self.position += 1

    def apply(self):
        self.pathState.replace(self.demand, self.node, self.position)

    def saveBest(self):
        self.storedPosition = self.position
        self.storedNode = self.node

    def applyBest(self):
        self.position = self.storedPosition
        self.node = self.storedNode
        self.apply()
