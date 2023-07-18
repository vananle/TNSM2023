class CapacityData:
    def __init__(self, capacity):
        self.capacity = capacity
        self.invCapacity = []
        for capa in capacity:
            self.invCapacity.append(1 / capa)
