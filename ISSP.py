import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 256)
		self.l4 = nn.Linear(256, 256)              
		self.l5 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = F.relu(self.l3(a))
		a = F.relu(self.l4(a))                            
		return self.max_action * torch.tanh(self.l5(a))

class Tactor(nn.Module):
        def __init__(self, state_dim):
                super(Tactor, self).__init__()

                self.l1 = nn.Linear(state_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.l3 = nn.Linear(256, 256)
                self.l4 = nn.Linear(256, 256)
                self.l5 = nn.Linear(256, state_dim)

        def forward(self, state):
                next_state = F.relu(self.l1(state))
                next_state = F.relu(self.l2(next_state))
                next_state = F.relu(self.l3(next_state))
                next_state = F.relu(self.l4(next_state))
                return self.l5(next_state)


class IDM(nn.Module):
        def __init__(self, state_dim, action_dim, max_action, is_discrete):
                super(IDM, self).__init__()

                self.l1 = nn.Linear(2 * state_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.l3 = nn.Linear(256, 256)
                self.l4 = nn.Linear(256, 256)
                self.l5 = nn.Linear(256, action_dim)
                self.max_action = max_action
                self.is_discrete = is_discrete
                

        def forward(self, state, next_state):
                ss = torch.cat([state, next_state], 1)
                a = F.relu(self.l1(ss))
                a = F.relu(self.l2(a))
                a = F.relu(self.l3(a))
                a = F.relu(self.l4(a))

                if self.is_discrete:
                    return torch.nn.Softmax()(self.l5(a))
                else:
                    return self.max_action * torch.tanh(self.l5(a))


class Critic(nn.Module):
        def __init__(self, state_dim, action_dim):
                super(Critic, self).__init__()

                # Q1 architecture
                self.l1 = nn.Linear(state_dim + action_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.l3 = nn.Linear(256, 256)
                self.l4 = nn.Linear(256, 256)
                self.l5 = nn.Linear(256, 1)

                # Q2 architecture
                self.l6 = nn.Linear(state_dim + action_dim, 256)
                self.l7 = nn.Linear(256, 256)
                self.l8 = nn.Linear(256, 256)
                self.l9 = nn.Linear(256, 256)
                self.l10 = nn.Linear(256, 1)

        def forward(self, state, action):
                ss = torch.cat([state, action], 1)

                q1 = F.relu(self.l1(ss))
                q1 = F.relu(self.l2(q1))
                q1 = F.relu(self.l3(q1))
                q1 = F.relu(self.l4(q1))
                q1 = self.l5(q1)

                q2 = F.relu(self.l6(ss))
                q2 = F.relu(self.l7(q2))
                q2 = F.relu(self.l8(q2))
                q2 = F.relu(self.l9(q2))
                q2 = self.l10(q2)
                return q1, q2


        def Q1(self, state, action):
                ss = torch.cat([state, action], 1)

                q1 = F.relu(self.l1(ss))
                q1 = F.relu(self.l2(q1))
                q1 = F.relu(self.l3(q1))
                q1 = F.relu(self.l4(q1))
                q1 = self.l5(q1)

                return q1


def asymmetric_l2_loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class ISSP(object):
        def __init__(
                self,
                state_dim,
                action_dim,
                max_action,
                is_discrete=False,
                discount=0.99,
                tau=0.005,
                policy_noise=0.05,
                policy_clip=1.0,
                policy_freq=2,
                lr=3e-4,
                alp=2.0,
                lamb=0.5,
                tau1=0.1,
                tau2=0.2,
        ):

                self.IDM = IDM(state_dim, action_dim, max_action, is_discrete).to(device)
                self.IDM_optimizer = torch.optim.Adam(self.IDM.parameters(), lr=lr)

                self.critic = Critic(state_dim, action_dim).to(device)
                self.critic_target = copy.deepcopy(self.critic)
                self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

                self.Tactor = Tactor(state_dim).to(device)
                self.Tactor_target = copy.deepcopy(self.Tactor)
                self.Tactor_optimizer = torch.optim.Adam(self.Tactor.parameters(), lr=lr)

                self.Eactor = Tactor(state_dim).to(device)
                self.Eactor_target = copy.deepcopy(self.Eactor)
                self.Eactor_optimizer = torch.optim.Adam(self.Eactor.parameters(), lr=lr)


                self.max_action = max_action
                self.is_discrete = is_discrete
                self.discount = discount
                self.tau = tau
                self.policy_noise = policy_noise
                self.policy_clip = policy_clip
                self.policy_freq = policy_freq
                self.lr = lr
                self.alp = alp
                self.lamb = lamb
                self.tau1 = tau1
                self.tau2 = tau2
                
                self.total_it = 0


        def select_action(self, state):
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                next_state = state + self.Eactor(state).detach()
                action = self.IDM(state, next_state)

                if self.is_discrete:
                    action = np.argmax(action)

                return action.cpu().data.numpy().flatten()
        

        def select_goal(self, state):
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                next_state = state + self.Tactor(state).detach()

                return next_state.cpu().data.numpy().flatten()

        def train(self, replay_buffer, env, min_s, max_s, batch_size=100, dynamics_only=False):
                # Sample replay buffer 
                metric = {'critic_loss': [], 'IDM_loss': [],\
                          'Tactor_loss': [], 'Tactor_gradient_loss': [],  'Tstate_BC_loss': [], \
                          'Eactor_loss': [], 'Eactor_gradient_loss': [],  'Estate_BC_loss': []}
                state, action, next_state, reward, not_done = replay_buffer.sample(batch_size) 
                          

                # my_beta = self.beta 
                if not dynamics_only:
                    self.total_it += 1
                    with torch.no_grad():
                            # Select action according to policy and add clipped noise
                            next_next_state = next_state + self.Tactor_target(next_state)
                            G_state_noise = (torch.randn_like(next_state) * self.policy_noise)
                            next_next_state = (next_next_state + G_state_noise).clamp(-self.policy_clip, self.policy_clip)
                            inverse_action = self.IDM(next_state, next_next_state)

                            # Compute the target Q value
                            Gtarget_Q1, Gtarget_Q2 = self.critic_target(next_state, inverse_action)
                            Gtarget_Q = torch.min(Gtarget_Q1, Gtarget_Q2)
                            Gtarget_Q = reward + not_done * self.discount * Gtarget_Q


                    # Get current Q estimates
                    current_Q1, current_Q2 = self.critic(state, action)

                    critic_loss = asymmetric_l2_loss(Gtarget_Q - current_Q1, self.tau1).mean() \
                                   + asymmetric_l2_loss(Gtarget_Q - current_Q2, self.tau2).mean()

                    # Optimize the critic
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                ########################### 2-1 IDM ###################################
                # Compute actor loss 
                IDM_state_noise = (torch.randn_like(state) * self.policy_noise)
                IDM_next_state = (next_state + IDM_state_noise).clamp(-self.policy_clip, self.policy_clip)
                predicted_action = self.IDM(state, IDM_next_state)
                IDM_loss = F.mse_loss(predicted_action, action)


                # Optimize the actor
                self.IDM_optimizer.zero_grad()
                IDM_loss.backward()
                self.IDM_optimizer.step()

                # Delayed actor updates
                if self.total_it % self.policy_freq == 0 and not dynamics_only:

                    ######################### guided actor ###################################
                    Tactor_prediction = state + self.Tactor(state)
                    Tstate_BC_loss = F.mse_loss(next_state, Tactor_prediction)

                    TAinverse_action = self.IDM(state, Tactor_prediction)
                    GQ1 = self.critic.Q1(state, TAinverse_action)
                    Tactor_gradient_loss = -GQ1.mean() / GQ1.abs().mean().detach()
                    
                    Tactor_loss = self.alp * Tactor_gradient_loss + Tstate_BC_loss
    
                    # Optimize the Tactor 
                    self.Tactor_optimizer.zero_grad()
                    Tactor_loss.backward()
                    self.Tactor_optimizer.step()

                    ######################### executed actor ###################################
                    Eactor_prediction = state + self.Eactor(state)
                    Estate_BC_loss = F.mse_loss(next_state, Eactor_prediction)

                    EAinverse_action = self.IDM(state, Eactor_prediction)
                    G2Q1 = self.critic.Q1(state, EAinverse_action)
                    Eactor_gradient_loss = -G2Q1.mean() / G2Q1.abs().mean().detach()
                    
                    Eactor_loss = self.lamb * self.alp * Eactor_gradient_loss + Estate_BC_loss
    
                    # Optimize the Tactor 
                    self.Eactor_optimizer.zero_grad()
                    Eactor_loss.backward()


                    # Update the frozen target Tactors
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    for param, target_param in zip(self.Tactor.parameters(), self.Tactor_target.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    for param, target_param in zip(self.Eactor.parameters(), self.Eactor_target.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    metric['critic_loss'].append(critic_loss.item())
                    metric['IDM_loss'].append(IDM_loss.item())                  
                    metric['Tactor_loss'].append(Tactor_loss.item())
                    metric['Tactor_gradient_loss'].append(Tactor_gradient_loss.item())
                    metric['Tstate_BC_loss'].append(Tstate_BC_loss.item())
                    metric['Eactor_loss'].append(Eactor_loss.item())
                    metric['Eactor_gradient_loss'].append(Eactor_gradient_loss.item())
                    metric['Estate_BC_loss'].append(Estate_BC_loss.item())
                return metric

        def save(self, filename):
                torch.save(self.critic.state_dict(), filename + "_critic")
                torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
                torch.save(self.IDM.state_dict(), filename + "_IDM")
                torch.save(self.IDM_optimizer.state_dict(), filename + "_IDM_optimizer")
                torch.save(self.Tactor.state_dict(), filename + "_Tactor")
                torch.save(self.Tactor_optimizer.state_dict(), filename + "_Tactor_optimizer")
                torch.save(self.Tactor.state_dict(), filename + "_Eactor")
                torch.save(self.Tactor_optimizer.state_dict(), filename + "_Eactor_optimizer")

        def load(self, filename):
                self.critic.load_state_dict(torch.load(filename + "_critic"))
                self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
                self.IDM.load_state_dict(torch.load(filename + "_IDM"))
                self.IDM_optimizer.load_state_dict(torch.load(filename + "_IDM_optimizer"))
                self.Tactor.load_state_dict(torch.load(filename + "_Tactor"))
                self.Tactor_optimizer.load_state_dict(torch.load(filename + "_Tactor_optimizer"))
                self.Tactor.load_state_dict(torch.load(filename + "_Eactor"))
                self.Tactor_optimizer.load_state_dict(torch.load(filename + "_Eactor_optimizer"))
                self.Tactor_target = copy.deepcopy(self.Tactor)
                self.critic_target = copy.deepcopy(self.critic)
