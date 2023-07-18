from abc import abstractmethod


class Neighborhood:
    @abstractmethod
    def setNeighborhood(self, demand):
        pass

    @abstractmethod
    def hasNext(self):
        pass

    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def apply(self):
        pass

    @abstractmethod
    def saveBest(self):
        pass

    @abstractmethod
    def applyBest(self):
        pass
