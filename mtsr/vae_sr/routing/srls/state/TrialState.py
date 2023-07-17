from abc import abstractmethod


class Trial:
    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def check(self):
        pass

    @abstractmethod
    def revert(self):
        pass

    @abstractmethod
    def commit(self):
        pass


class Objective:
    @abstractmethod
    def score(self):
        pass


class TrialState(Trial):

    def __init__(self):
        self.nTrial = 0
        self.maxTrial = 16
        self.trials = []

    @abstractmethod
    def updateState(self):
        pass

    @abstractmethod
    def commitState(self):
        pass

    @abstractmethod
    def revertState(self):
        pass

    def addTrial(self, trial):
        if self.nTrial == self.maxTrial:
            self.maxTrial <<= 1

        self.nTrial += 1
        self.trials.append(trial)

    def update(self):
        self.updateState()

        pTrial = 0
        while pTrial < self.nTrial:
            self.trials[pTrial].update()
            pTrial += 1

    def check(self):
        pTrial = 0
        while (pTrial < self.nTrial):
            if self.trials[pTrial].check() == False:
                break
            pTrial += 1

        _pass = (pTrial == self.nTrial)

        if _pass != True:
            while pTrial > 0:
                pTrial -= 1
                self.trials[pTrial].revert()

            self.revertState()

        return _pass

    def revert(self):
        self.revertAll()
        self.revertState()

    def commit(self):
        self.commitAll()
        self.commitState()

    def revertAll(self):
        pTrial = self.nTrial
        while pTrial > 0:
            pTrial -= 1
            self.trials[pTrial].revert()

    def commitAll(self):
        pTrial = 0
        while pTrial < self.nTrial:
            self.trials[pTrial].commit()
            pTrial += 1
