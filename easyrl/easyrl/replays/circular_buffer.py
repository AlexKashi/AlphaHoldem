import random
from copy import deepcopy


class CyclicBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.cur_pos = 0

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        return self.buffer[item]

    def append(self, data):
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.cur_pos] = data
        self.cur_pos = int((self.cur_pos + 1) % self.capacity)

    def sample(self, batch_size):
        if batch_size == len(self.buffer):
            return deepcopy(self.buffer)
        else:
            data = random.sample(self.buffer, batch_size)
        return deepcopy(data)

    def get_all(self):
        return deepcopy(self.buffer)

    def clear(self):
        self.buffer.clear()
