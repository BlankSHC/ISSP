<h1>ISSP</span></h1>

A PyTorch implementation of "Rethinking Offline Reinforcement Learning with Implicit State-State Planning".


## Evaluation Video

 <img src="Videos/halfcheetah-random-v2.gif" width="18%">   <img src="Videos/halfcheetah-medium-v2.gif" width="18%">   <img src="Videos/halfcheetah-medium-replay-v2.gif" width="18%">   <img src="Videos/halfcheetah-medium-expert-v2.gif" width="18%">   <img src="Videos/halfcheetah-expert-v2.gif" width="18%">
 <p>halfcheetah-random-v2, halfcheetah-medium-v2, halfcheetah-medium-replay-v2, halfcheetah-medium-expert-v2, halfcheetah-expert-v2</p>

  <img src="Videos/hopper-random-v2.gif" width="18%"><p>hopper-random-v2</p><img src="Videos/hopper-medium-v2.gif" width="18%"><p>hopper-medium-v2.gif</p>
  <img src="Videos/hopper-medium-replay-v2.gif" width="18%"><p>hopper-medium-replay-v2</p>
  <img src="Videos/hopper-medium-expert-v2.gif" width="18%"><p>hopper-medium-expert-v2</p>
  <img src="Videos/hopper-expert-v2.gif" width="18%"><p>hopper-expert-v2</p>

  <img src="Videos/walker2d-random-v2.gif" width="18%"><p>walker2d-random-v2</p>
  <img src="Videos/walker2d-medium-v2.gif" width="18%"><p>walker2d-medium-v2</p>
  <img src="Videos/walker2d-medium-replay-v2.gif" width="18%"><p>walker2d-medium-replay-v2</p>
  <img src="Videos/walker2d-medium-expert-v2.gif" width="18%"><p>walker2d-medium-expert-v2</p>
  <img src="Videos/walker2d-expert-v2.gif" width="18%"><p>walker2d-expert-v2</p>

  <img src="Videos/antmaze-umaze-v2.gif" width="18%"><p>antmaze-umaze-v2</p>
  <img src="Videos/antmaze-umaze-diverse-v2.gif" width="18%"><p>antmaze-umaze-diverse-v2</p>
  <img src="Videos/antmaze-medium-play-v2.gif" width="18%"><p>antmaze-medium-play-v2</p>
  <img src="Videos/antmaze-medium-diverse-v2.gif" width="18%"><p>antmaze-medium-diverse-v2</p>

  <img src="Videos/antmaze-large-play-v2.gif" width="18%"><p>antmaze-large-play-v2</p>
  <img src="Videos/antmaze-large-diverse-v2.gif" width="18%"><p>antmaze-large-diverse-v2</p>

----

## Getting started

We provide requirements and examples on how to train and evaluate ISSP agents. 

### Preparing

PyTorch == 1.10  
gym == 0.18  
MuJoCo == 2.00  
mujoco-py == 2.0.2.8  
d4rl == 1.1

### Training

See below examples on how to train OBAC on a single task (e.g. antmaze-large-diverse-v2).

```python
python main.py --env antmaze-large-diverse-v2
```
We recommend using default hyperparameters. See `main.py` for a full list of hyperparameters.
