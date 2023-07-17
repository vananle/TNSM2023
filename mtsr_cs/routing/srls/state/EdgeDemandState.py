import math
import random
from abc import abstractmethod


class EdgeDemandState:
    def __init__(self, nDemands, nEdges):
        self.nDemands = nDemands
        self.nEdges = nEdges

    @abstractmethod
    def updateEdgeDemand(self, edge, demand, flowDelta):
        pass

    @abstractmethod
    def selectRandomDemand(self, edge):
        pass

    @abstractmethod
    def flowOnEdgeDemand(self, edge, demand):
        pass

    def restrictDemands(self, set):
        nDemand = len(set)
        edge = self.nEdges

        while edge > 0:
            edge -= 1
            demand = nDemand
            while demand > 0:
                demand -= 1
                if (set[demand] != True) & self.flowOnEdgeDemand(edge, demand != 0.0):
                    self.updateEdgeDemand(edge, demand, -self.flowOnEdgeDemand(edge, demand))


class EdgeDemandStateTree(EdgeDemandState):

    def __init__(self, nDemands, nEdges, capa):
        super().__init__(nDemands, nEdges)
        self.capacityData = capa
        logDemands = math.ceil(math.log2(nDemands))
        self.baseDemand = 1 << logDemands
        self.treeFlowOnEdgeDemand = []
        self.epsilon = 1e-6
        for i in range(nEdges):
            self.treeFlowOnEdgeDemand.append([0.0] * (self.baseDemand * 2))

    def __str__(self):
        return str(self.treeFlowOnEdgeDemand)

    def flowOnEdgeDemand(self, edge, demand):
        return self.treeFlowOnEdgeDemand[edge][demand + self.baseDemand]

    def modifyFlowDemand(self, edge, demand, newFlow):
        assert (newFlow > -self.epsilon)
        self.treeFlowOnEdgeDemand[edge][demand + self.baseDemand] = newFlow
        self.modifyFlowDemandTree(self.treeFlowOnEdgeDemand[edge], (self.baseDemand + demand) >> 1)

    def modifyFlowDemandTree(self, tree, node):
        left = node << 1
        right = left + 1
        tree[node] = tree[left] + tree[right]
        assert (tree[node] > -self.epsilon)
        if node > 1:
            self.modifyFlowDemandTree(tree, node >> 1)

    def updateEdgeDemand(self, edge, demand, flowDelta):
        oldFlow = self.flowOnEdgeDemand(edge, demand)
        newFlow = oldFlow + flowDelta
        self.modifyFlowDemand(edge, demand, newFlow)

    def selectRandomDemand(self, edge):

        r = random.random() * self.treeFlowOnEdgeDemand[edge][1] - self.epsilon
        selectedNode = self.selectDemand(self.treeFlowOnEdgeDemand[edge], 1, 2 * self.baseDemand, r)
        selectedDemand = selectedNode - self.baseDemand

        assert ((selectedDemand < self.nDemands) & (selectedDemand >= 0))
        return selectedDemand

    def selectDemand(self, tree, node, limit, r):
        left = node << 1
        if left >= limit:
            return node
        else:
            if r <= tree[left]:
                return self.selectDemand(tree, left, limit, r)
            else:
                right = left + 1
                return self.selectDemand(tree, right, limit, r - tree[left])

    def showIF(self):
        print('******************')
        for tree in self.treeFlowOnEdgeDemand:
            print(tree)
        print('******************')

# tree = EdgeDemandStateTree(5, 2 , None)

# tree.showIF()

# tree.updateEdgeDemand(0,3,200)
# tree.updateEdgeDemand(0,4,200)

# tree.showIF()

# print(tree.selectRandomDemand(0))
