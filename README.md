<h1>ISSP</span></h1>

A PyTorch implementation of "Rethinking Offline Reinforcement Learning with Implicit State-State Planning".


## Evaluation Video

<img src="Videos/halfcheetah-random-v2.gif" width="16%"><img src="Videos/halfcheetah-medium-v2.gif" width="16%"><img src="Videos/halfcheetah-medium-replay-v2.gif" width="16%"><img src="Videos/halfcheetah-medium-expert-v2.gif" width="16%"><img src="Videos/halfcheetah-expert-v2.gif" width="16%">

<img src="Videos/hopper-random-v2.gif" width="16%"><img src="Videos/hopper-medium-v2.gif" width="16%"><img src="Videos/hopper-medium-replay-v2.gif" width="16%"><img src="Videos/hopper-medium-expert-v2.gif" width="16%"><img src="Videos/hopper-expert-v2.gif" width="16%">

<img src="Videos/walker2d-random-v2.gif" width="16%"><img src="Videos/walker2d-medium-v2.gif" width="16%"><img src="Videos/walker2d-medium-replay-v2.gif" width="16%"><img src="Videos/walker2d-medium-expert-v2.gif" width="16%"><img src="Videos/walker2d-expert-v2.gif" width="16%">

<img src="Videos/antmaze-umaze-v2.gif" width="16%"><img src="Videos/antmaze-umaze-diverse-v2.gif" width="16%"><img src="Videos/antmaze-medium-play-v2.gif" width="16%"><img src="Videos/antmaze-medium-diverse-v2.gif" width="16%">

<img src="Videos/antmaze-large-play-v2.gif" width="16%"><img src="Videos/antmaze-large-diverse-v2.gif" width="16%">

----

## Getting started

We provide examples on how to train and evaluate ISSP agents. 

### TrainingPreparing

PyTorch 1.10 with Python 3.7
MuJoCo 2.00 with mujoco-py 2.1.2.14
d4rl 1.1 with version2 (v2) datasets

### Training

See below examples on how to train OBAC on a single task (e.g. antmaze-large-diverse-v2).

```python
python main.py --env antmaze-large-diverse-v2
```
We recommend using default hyperparameters. See `main.py` for a full list of arguments.
----
