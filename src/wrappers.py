# src/wrappers.py
import numpy as np, cv2, gymnasium as gym
import ale_py  # <- keep this; it ensures ALE is registered
from collections import deque

class Resize84Gray(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 255, (84, 84), dtype=np.uint8)
    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs

class ActionRepeatMax(gym.Wrapper):
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat
        self._buf = deque(maxlen=2)
    def step(self, action):
        total = 0.0; term = trunc = False; info = {}
        for _ in range(self.repeat):
            obs, r, term, trunc, info = self.env.step(action)
            self._buf.append(obs); total += r
            if term or trunc: break
        obs = np.maximum(self._buf[0], self._buf[-1]) if len(self._buf) == 2 else obs
        return obs, total, term, trunc, info
    def reset(self, **kw):
        self._buf.clear()
        return self.env.reset(**kw)

class RewardClip(gym.RewardWrapper):
    def reward(self, r): return float(np.sign(r))

class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env); self.k = k; self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, (k, *shp), dtype=np.uint8)
    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        for _ in range(self.k): self.frames.append(obs)
        return self._get(), info
    def step(self, a):
        obs, r, term, trunc, info = self.env.step(a)
        self.frames.append(obs)
        return self._get(), r, term, trunc, info
    def _get(self): return np.stack(self.frames, axis=0)

def make_kungfu_env(render_mode=None, stack=4, repeat=4, clip_rewards=True):
    env = gym.make("ALE/KungFuMaster-v5", render_mode=render_mode)
    env = ActionRepeatMax(env, repeat=repeat)
    env = Resize84Gray(env)
    if clip_rewards: env = RewardClip(env)
    env = FrameStack(env, k=stack)
    return env
