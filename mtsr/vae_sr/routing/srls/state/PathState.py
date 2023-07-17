import os
import sys

lib_path = os.path.abspath(os.path.join('demand'))
sys.path.append(lib_path)

from .TrialState import TrialState


class PathState(TrialState):
    def __init__(self, demands):
        TrialState.__init__(self)
        self.demands = demands
        self.nDemands = demands.nDemands
        self.maxDetourSize = 6
        self.paths = []
        self.nChanged = 0
        self.markedChanged = [False] * self.nDemands
        self.changed = [-1] * self.nDemands

        for i in range(self.nDemands):
            self.paths.append(Path(i, self.source(i), self.destination(i), self.maxDetourSize))

    def source(self, demand):
        return self.demands.demandSrcs[demand]

    def destination(self, demand):
        return self.demands.demandDests[demand]

    def size(self, demand):
        return self.paths[demand].currentSize

    def path(self, demand):
        return self.paths[demand].currentPath

    def oldSize(self, demand):
        return self.paths[demand].savedSize

    def oldPath(self, demand):
        return self.paths[demand].savedPath

    def insert(self, demand, node, position):
        self.addChanged(demand)
        self.paths[demand].insert(node, position)

    def replace(self, demand, node, position):
        self.addChanged(demand)
        self.paths[demand].replace(node, position)

    def remove(self, demand, position):
        self.addChanged(demand)
        self.paths[demand].remove(position)

    def setPath(self, demand, newPath, newSize):
        self.addChanged(demand)
        self.paths[demand].setPath(newPath, newSize)

    def reset(self, demand):
        self.addChanged(demand)
        self.paths[demand].reset()

    def commitState(self):
        while self.nChanged > 0:
            self.nChanged -= 1
            demand = self.changed[self.nChanged]
            self.markedChanged[demand] = False

    def revertState(self):
        while self.nChanged > 0:
            self.nChanged -= 1
            demand = self.changed[self.nChanged]
            self.paths[demand].restore()
            self.markedChanged[demand] = False

    def updateState(self):
        pass

    def addChanged(self, demand):
        if self.markedChanged[demand] == False:
            self.paths[demand].save()
            self.markedChanged[demand] = True
            self.changed[self.nChanged] = demand
            self.nChanged += 1


class Path:
    def __init__(self, demand, source, destination, maxSize):
        self.demand = demand
        self.source = source
        self.destination = destination
        self.maxSize = maxSize
        assert maxSize >= 2

        self.currentSize = 2
        self.currentPath = []
        self.currentPath.append(source)
        self.currentPath.append(destination)

        self.savedSize = 2
        self.savedPath = []
        self.savedPath.append(source)
        self.savedPath.append(destination)

    def insert(self, node, position):
        assert (position > 0)
        assert (position < self.currentSize)
        assert (self.currentSize < self.maxSize)
        self.currentSize += 1
        self.currentPath.insert(position, node)

    def replace(self, node, position):
        assert (position > 0)
        assert (position < self.currentSize - 1)
        self.currentPath[position] = node

    def reset(self):
        self.currentPath = []
        self.currentPath.append(self.source)
        self.currentPath.append(self.destination)
        self.currentSize = 2

    def remove(self, position):
        self.currentPath.pop(position)
        self.currentSize -= 1

    def setPath(self, newPath, newSize):
        assert (newPath[0] == self.source)
        assert (newPath[newSize - 1] == self.destination)
        assert (newSize <= self.maxSize)
        self.currentSize = newSize
        self.currentPath = newPath.copy()

    def save(self):
        self.savedSize = self.currentSize
        self.savedPath = self.currentPath.copy()

    def restore(self):
        self.currentSize = self.savedSize
        self.currentPath = self.savedPath.copy()

    def __str__(self) -> str:
        return 'current  ' + str(self.currentPath) + '    saved  ' + str(self.savedPath)
