import os
import sys

lib_path = os.path.abspath(os.path.join('state'))
sys.path.append(lib_path)
from .TrialState import TrialState


class SavedState(TrialState):
    def __init__(self, pathState):
        TrialState.__init__(self)
        self.pathState = pathState
        self.nDemands = pathState.nDemands
        self.changed = [False] * self.nDemands
        self.changedStack = [-1] * self.nDemands
        self.nChanged = 0
        self.lengthPaths = [0] * self.nDemands
        self.paths = []
        for i in range(self.nDemands):
            self.paths.append(pathState.path(i).copy())
            self.lengthPaths[i] = len(self.paths[i])

    def changedPath(self, demand):
        if self.changed[demand] == False:
            self.changedStack[self.nChanged] = demand
            self.nChanged += 1

        self.changed[demand] = True

    def savePath(self):
        count = 0
        while self.nChanged > 0:
            self.nChanged -= 1
            demand = self.changedStack[self.nChanged]
            path = self.pathState.path(demand)
            size = self.pathState.size(demand)
            self.changed[demand] = False

            pathChanged = (size != self.lengthPaths[demand])

            if pathChanged != True:
                p = size
                while p > 0:
                    p -= 1
                    pathChanged = pathChanged | (path[p] != self.paths[demand][p])

            if pathChanged == True:
                count += 1

            self.paths[demand] = path.copy()
            self.lengthPaths[demand] = size

        return count

    def restorePath(self):
        while self.nChanged > 0:
            self.nChanged -= 1
            demand = self.changedStack[self.nChanged]
            self.changed[demand] = False
            self.pathState.setPath(demand, self.paths[demand], self.lengthPaths[demand])

    def check(self):
        return True

    def commit(self):
        currentChanged = self.pathState.changed
        p = self.pathState.nChanged
        while p > 0:
            p -= 1
            demand = currentChanged[p]
            self.changedPath(demand)

    def revert(self):
        pass

    def update(self):
        pass

# def showPaths(paths):
#     print('********************')
#     for path in paths:
#         print(path)
# demands = DemandsData([1,2,3,4],[4,3,2,1],[3,2,3,2],[1,1,1,1])
# pathState = PathState(demands)
# showPaths(pathState.paths)
# pathState.insert(1,4,1)

# (pathState.paths)
