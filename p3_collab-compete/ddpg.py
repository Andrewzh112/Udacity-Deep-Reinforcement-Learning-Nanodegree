import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6) # replay buffer size
BATCH_SIZE = 256       # minibatch size
GAMMA = 0.99           # discount factor
TAU = 5e-2             # for soft update of target parameters
LR_ACTOR = 1e-3        # learning rate of the actor
LR_CRITIC = 1e-3       # learning rate of the critic
WEIGHT_DECAY = 0       # L2 weight decay

LEARN_INTERVAL = 20       
LEARN_PASSES = 10            
GRAD_CLIPPING = 1.0          

EPSILON = 1.0         
EPSILON_DECAY = 1e-6

NOISE_AMPLIFICATION = 1        
NOISE_AMPLIFICATION_DECAY = 1  


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ddpg_agent():
    """Interacts with and learns from the environment"""
    
    def __init__(self, state_size, action_size, agent_id, random_seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.agent_id = agent_id
        
        self.epsilon = EPSILON
        
        # Actor network
        self.actor_local = Actor(state_size=state_size, action_size=action_size, seed=random_seed).to(device)
        self.actor_target = Actor(state_size=state_size, action_size=action_size, seed=random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        
        # Critic network
        self.critic_local = Critic(state_size=state_size, action_size=action_size, seed=random_seed).to(device)
        self.critic_target = Critic(state_size=state_size, action_size=action_size, seed=random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        self.hard_update(self.actor_local, self.actor_target)
        self.hard_update(self.critic_local, self.critic_target)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        
        self.noise_amplification = NOISE_AMPLIFICATION
        self.noise_amplification_decay = NOISE_AMPLIFICATION_DECAY

                
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy"""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            action += self.epsilon * self.noise.sample()
            self._decay_noise_amplification()
            
        return np.clip(action, -1, 1)
    
    
    def reset(self):
        self.noise.reset()
        
        
    def learn(self, experiences, gamma, next_actions, actions_pred):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        agent_id_tensor = torch.tensor([self.agent_id - 1]).to(device)

        # Update critic
        # get predicted next-state actions and Q values from target models
        Q_targets_next = self.critic_target(next_states, next_actions)
        # compute Q targets for current states(y_i)
        Q_targets = rewards.index_select(1, agent_id_tensor) + \
               (GAMMA * Q_targets_next *  (1 - dones.index_select(1, agent_id_tensor)))
        # compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # gradient clipping for critic
        if GRAD_CLIPPING > 0:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), GRAD_CLIPPING)
        self.critic_optimizer.step()
        
        # update actor
        # compute actor loss
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # update target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.epsilon -= EPSILON_DECAY
        self.noise.reset()

        
    def hard_update(self, local_model, target_model):
        """Hard update model parameters.
        θ_target = θ_local
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Params
        ======
        local_model: PyTorch model from which weights will be copied
        target_model: PyToch model to which weights will be copied
        tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def _decay_noise_amplification(self):
        """decaying exploration noise amplification."""
        self.noise_amplification *= self.noise_amplification_decay

        
class OUNoise:
    """Ornsten-Uhlenbeck process to add noise"""
    def __init__(self, action_dimension, seed=0, scale=1, mu=0, theta=0.15, sigma=0.5):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.action_dimension)
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()
    
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size) # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences =  random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """"Return the current size of internal memory"""
        return len(self.memory)