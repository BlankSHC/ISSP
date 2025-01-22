import os
import datetime
import numpy as np
import pandas as pd
import torch
import gym
import argparse
import d4rl

import utils
from logger import logger
import ISSP
from Video_Recorder import VideoRecorder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Runs policy for X episodes and returns D4RL score
def eval_policy(iter,policy, env_name, seed, min_s, max_s, seed_offset=100, eval_episodes=100, video_path=None):
	BEN = datetime.datetime.now()
	eval_env = gym.make(env_name)
	eval_env.seed(seed+seed_offset)

	avg_reward = 0.	
	for episode_n in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = 2 * (np.array(state).reshape(1, -1) - min_s) / (max_s - min_s) - 1   
			action = policy.select_action(state)
			state, reward, done, info = eval_env.step(action)
			avg_reward += reward
			if 'antmaze' in env_name:
				episode_return += (reward - 1.0)
			else:
				episode_return += reward

 
	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	# Print evaluation metrics
	print(f"Evaluation return over {eval_episodes} episodes: {avg_reward:.2f}    D4RL score: {d4rl_score:.2f}")
	print("**************************************************************")
	return avg_reward, d4rl_score

# Hyperparameter configurations for different environments
hyperparameters = {
    'halfcheetah-random-v2':         {'lr': 3e-4, 'alp': 0.4,  'lamb': 10.0, 'tau1': 0.2, 'tau2': 0.3},
    'halfcheetah-medium-v2':         {'lr': 3e-4, 'alp': 0.4,  'lamb': 10.0, 'tau1': 0.2, 'tau2': 0.3},
    'halfcheetah-medium-replay-v2':  {'lr': 3e-4, 'alp': 0.4,  'lamb': 2.5,  'tau1': 0.2, 'tau2': 0.3},
    'halfcheetah-medium-expert-v2':  {'lr': 3e-4, 'alp': 0.04, 'lamb': 2.5,  'tau1': 0.2, 'tau2': 0.3},
    'halfcheetah-expert-v2':         {'lr': 3e-4, 'alp': 0.04, 'lamb': 2.5,  'tau1': 0.2, 'tau2': 0.3},
    'hopper-random-v2':              {'lr': 3e-4, 'alp': 0.4,  'lamb': 10.0, 'tau1': 0.2, 'tau2': 0.3},
    'hopper-medium-v2':              {'lr': 3e-4, 'alp': 0.4,  'lamb': 10.0, 'tau1': 0.2, 'tau2': 0.3},
    'hopper-medium-replay-v2':       {'lr': 3e-4, 'alp': 0.4,  'lamb': 10.0, 'tau1': 0.2, 'tau2': 0.3},
    'hopper-medium-expert-v2':       {'lr': 3e-4, 'alp': 0.4,  'lamb': 10.0, 'tau1': 0.2, 'tau2': 0.3},
    'hopper-expert-v2':              {'lr': 3e-4, 'alp': 0.4,  'lamb': 10.0, 'tau1': 0.2, 'tau2': 0.3},
    'walker2d-random-v2':            {'lr': 3e-4, 'alp': 0.4,  'lamb': 10.0, 'tau1': 0.2, 'tau2': 0.3},  
    'walker2d-medium-v2':            {'lr': 3e-4, 'alp': 0.4,  'lamb': 10.0, 'tau1': 0.2, 'tau2': 0.3},
    'walker2d-medium-replay-v2':     {'lr': 3e-4, 'alp': 0.4,  'lamb': 10.0, 'tau1': 0.2, 'tau2': 0.3},
    'walker2d-medium-expert-v2':     {'lr': 3e-4, 'alp': 0.4,  'lamb': 10.0, 'tau1': 0.2, 'tau2': 0.3},
    'walker2d-expert-v2':            {'lr': 3e-4, 'alp': 0.4,  'lamb': 10.0, 'tau1': 0.2, 'tau2': 0.3}, 
    'antmaze-umaze-v2':              {'lr': 3e-4, 'alp': 0.4,  'lamb': 2.5,  'tau1': 0.2, 'tau2': 0.3},
    'antmaze-umaze-diverse-v2':      {'lr': 3e-4, 'alp': 0.4,  'lamb': 2.5,  'tau1': 0.2, 'tau2': 0.3},
    'antmaze-medium-play-v2':        {'lr': 3e-4, 'alp': 0.4,  'lamb': 2.5,  'tau1': 0.9, 'tau2': 0.9},
    'antmaze-medium-diverse-v2':     {'lr': 3e-4, 'alp': 0.4,  'lamb': 2.5,  'tau1': 0.9, 'tau2': 0.9},
    'antmaze-large-play-v2':         {'lr': 3e-4, 'alp': 0.4,  'lamb': 2.5,  'tau1': 0.9, 'tau2': 0.9},
    'antmaze-large-diverse-v2':      {'lr': 3e-4, 'alp': 0.4,  'lamb': 2.5,  'tau1': 0.9, 'tau2': 0.9},
}

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	# Experiment setup
	parser.add_argument("--policy", default="ISSP")                 # Policy name
	parser.add_argument("--env", default="halfcheetah-expert-v2")   # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.05)             # Noise added to target policy during critic update
	parser.add_argument("--policy_clip", default=1.0)               # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	# ISSP-specific parameters
	parser.add_argument("--lr", default=3e-4, type=float)	        # learning rate
	parser.add_argument("--alp", default=0.4, type=float)           # Tactor loss
	parser.add_argument("--lamb", default=2.5, type=float)          # Eactor loss
	parser.add_argument("--tau1", default=0.2, type=float)          # tau1 for Q1
	parser.add_argument("--tau2", default=0.3, type=float)          # tau2 for Q2
	args = parser.parse_args()


	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists(f"./Results"):
		os.makedirs(f"./Results")
	if not os.path.exists(f"./Results/times"):
		os.makedirs(f"./Results/times")
	# video_path = f"./Videos/{args.policy}_{args.env}_{args.seed}"
	# if not os.path.exists(video_path):
	# 	os.makedirs(video_path)
	video_path = None
	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]  
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		"lr": hyperparameters[args.env]['lr'],
		"alp": hyperparameters[args.env]['alp'],
		"lamb": hyperparameters[args.env]['lamb'],
		"tau1": hyperparameters[args.env]['tau1'],
		"tau2": hyperparameters[args.env]['tau2'],
	}

	# Initialize policy
	policy = ISSP.ISSP(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")
	

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

	if 'antmaze' in args.env:
		replay_buffer.reward = replay_buffer.reward - 1.0
		eval_episodes_num = 10
	else:
		eval_episodes_num = 10

	min_s = np.min(replay_buffer.state, 0)
	max_s = np.max(replay_buffer.state, 0)
	replay_buffer.normalize_states_minmax(min=min_s, max=max_s)


	#Training + Evaluation
	evaluations = []
	for t in range(int(args.max_timesteps)):
		loss_metric = policy.train(replay_buffer, args.env, min_s, max_s, args.batch_size)
		if (t + 1) % (args.eval_freq * 2) == 0:
			print(f"Time steps: {(t+1)/5000-1}    Env: {args.env}    Seed: {args.seed}")
			logger.record_tabular('Critic Loss', np.mean(loss_metric['critic_loss']))
			logger.record_tabular('IDM Loss', np.mean(loss_metric['IDM_loss']))
			logger.record_tabular('Tactor Loss', np.mean(loss_metric['Tactor_loss']))
			logger.record_tabular('Tactor Gradient Loss', np.mean(loss_metric['Tactor_gradient_loss']))
			logger.record_tabular('Tstate BC Loss', np.mean(loss_metric['Tstate_BC_loss']))
			logger.record_tabular('Eactor Loss', np.mean(loss_metric['Eactor_loss']))
			logger.record_tabular('Eactor Gradient Loss', np.mean(loss_metric['Eactor_gradient_loss']))
			logger.record_tabular('Estate BC Loss', np.mean(loss_metric['Estate_BC_loss']))
			logger.dump_tabular()
			evaluations.append(eval_policy(t+1, policy, args.env, args.seed, min_s, max_s, eval_episodes=eval_episodes_num, video_path=video_path))
			log = pd.DataFrame(evaluations)
			log.to_csv(f"./Results/{file_name}.csv")
			if args.save_model: policy.save(f"./models/{file_name}")
