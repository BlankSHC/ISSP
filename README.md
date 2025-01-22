<h1>ISSP</span></h1>

A PyTorch implementation of "Rethinking Offline Reinforcement Learning via Implicit State-State Planning".


## Evaluation Video

<img src="Videos/halfcheetah-random-v2.gif" width="19.5%">   <img src="Videos/halfcheetah-medium-v2.gif" width="19.5%">   <img src="Videos/halfcheetah-medium-replay-v2.gif" width="19.5%">   <img src="Videos/halfcheetah-medium-expert-v2.gif" width="19.5%">   <img src="Videos/halfcheetah-expert-v2.gif" width="19.5%">
From left to right: `halfcheetah-random-v2`,   `halfcheetah-medium-v2`,   `halfcheetah-medium-replay-v2`,   `halfcheetah-medium-expert-v2`,   `halfcheetah-expert-v2`

<img src="Videos/hopper-random-v2.gif" width="19.5%">  <img src="Videos/hopper-medium-v2.gif" width="19.5%">  <img src="Videos/hopper-medium-replay-v2.gif" width="19.5%">  <img src="Videos/hopper-medium-expert-v2.gif" width="19.5%"> <img src="Videos/hopper-expert-v2.gif" width="19.5%">
From left to right: `hopper-random-v2`,   `hopper-medium-v2`,   `hopper-medium-replay-v2`,   `hopper-medium-expert-v2`,   `hopper-expert-v2`

<img src="Videos/walker2d-random-v2.gif" width="19.5%">  <img src="Videos/walker2d-medium-v2.gif" width="19.5%">  <img src="Videos/walker2d-medium-replay-v2.gif" width="19.5%">  <img src="Videos/walker2d-medium-expert-v2.gif" width="19.5%">  <img src="Videos/walker2d-expert-v2.gif" width="19.5%">
From left to right: `walker2d-random-v2`,   `walker2d-medium-v2`,   `walker2d-medium-replay-v2`,   `walker2d-medium-expert-v2`,   `walker2d-expert-v2`

<img src="Videos/antmaze-umaze-v2.gif" width="19.5%">  <img src="Videos/antmaze-umaze-diverse-v2.gif" width="19.5%">  <img src="Videos/antmaze-medium-play-v2.gif" width="19.5%">  <img src="Videos/antmaze-medium-diverse-v2.gif" width="19.5%">  
From left to right:  `antmaze-umaze-v2`, `antmaze-umaze-diverse-v2`, `antmaze-medium-play-v2`, `antmaze-medium-diverse-v2`

<img src="Videos/antmaze-large-play-v2.gif" width="19.5%"> <img src="Videos/antmaze-large-diverse-v2.gif" width="19.5%">  
From left to right:  `antmaze-large-play-v2`, `antmaze-large-diverse-v2`

----

## Getting started

We provide requirements and examples on how to train and evaluate ISSP agents. 

### Preparing

PyTorch == 1.10  
gym == 0.19.5  
MuJoCo == 2.00  
mujoco-py == 2.0.2.8  
d4rl == 1.1

### Training and evaluating

See below examples on how to train and evaluate ISSP on a single task (e.g. antmaze-large-diverse-v2).

```python
python main.py --env antmaze-large-diverse-v2
```
We recommend using default hyperparameters. See `main.py` for a full list of hyperparameters.

