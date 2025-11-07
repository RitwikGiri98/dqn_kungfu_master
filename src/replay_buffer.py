# src/replay_buffer.py
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, obs_shape):
        self.capacity = capacity; self.idx = 0; self.full = False
        self.s  = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.a  = np.zeros((capacity,), dtype=np.int64)
        self.r  = np.zeros((capacity,), dtype=np.float32)
        self.ns = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.d  = np.zeros((capacity,), dtype=np.bool_)
    def __len__(self): return self.capacity if self.full else self.idx
    def push(self, s, a, r, ns, done):
        i = self.idx
        self.s[i], self.a[i], self.r[i], self.ns[i], self.d[i] = s, a, r, ns, done
        self.idx = (i + 1) % self.capacity; self.full |= self.idx == 0
    def sample(self, batch):
        idxs = np.random.randint(0, len(self), size=batch)
        s  = self.s[idxs].astype(np.float32) / 255.0
        ns = self.ns[idxs].astype(np.float32) / 255.0
        return s, self.a[idxs], self.r[idxs], ns, self.d[idxs].astype(np.float32)
