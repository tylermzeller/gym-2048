# Gym 2048 Environment

A 2048 environment for RL experiments with [Gym](https://gym.openai.com/)

## Quickstart
Requires gym and numpy
```bash
$ pip install gym==0.10.4 numpy==1.14.2
```

Install using `setup.py`
```bash
$ python setup.py install
```

Usage
```python
import gym
import gym_2048

if __name__=='__main__':
  env = gym.make('2048-v0')
  assert env.observation_space.shape[1] == 4*4
  assert env.action_space.n == 4
```
