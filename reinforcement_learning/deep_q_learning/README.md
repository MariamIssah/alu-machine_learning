Deep Q-Learning — Breakout

This folder contains a minimal setup to train and run a Deep Q-Network (DQN)
agent to play Atari Breakout using Keras and Keras-RL.

Files:

- `train.py`: trains a `DQNAgent` using `SequentialMemory` and `EpsGreedyQPolicy`.
  Saves weights to `policy.h5` when finished.
- `play.py`: loads `policy.h5` and runs evaluation episodes with a `GreedyQPolicy`.

Quickstart

1. Install dependencies (on Ubuntu 16.04 as required by the project):

```bash
pip install --user gym[atari]
pip install --user keras==2.2.4 keras-rl==0.4.2
```

2. Train (this can take many hours for good performance — environment variable
   `DQN_TRAIN_STEPS` can be used to override steps):

```bash
python3 train.py
# or shorter quick run
DQN_TRAIN_STEPS=5000 python3 train.py
```

3. Play (loads `policy.h5`):

```bash
python3 play.py
```

Notes

- The training script uses standard Atari preprocessing (84x84 grayscale,
  stacked frames) when available via gym wrappers. If those wrappers are
  unavailable it falls back to `Breakout-v0` but results may be degraded.
- For a policy that reliably scores > 10, many training steps are required.

Author: Alexa Orrico (Holberton School)
