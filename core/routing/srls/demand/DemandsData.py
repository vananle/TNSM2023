class DemandsData:
    def __init__(self, demandLabels, demandSrcs, demandDests, demandTraffics):
        self.demandLabels = demandLabels
        self.demandSrcs = demandSrcs
        self.demandDests = demandDests
        self.demandTraffics = demandTraffics
        self.nDemands = len(self.demandSrcs)
        self.demands = list(range(self.nDemands))
