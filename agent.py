import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, alghoritm="REINFORCE"):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()
        self.alghoritm = alghoritm

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)

        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        if alghoritm == "AC":
        
            self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
            self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
            self.fc3_critic = torch.nn.Linear(self.hidden, 1)


        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)
        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic network
        if self.alghoritm == "AC":
            x_critic = self.tanh(self.fc1_critic(x))
            x_critic = self.tanh(self.fc2_critic(x_critic))
            value = self.fc3_critic(x_critic) # State value
            return normal_dist, value

        
        return normal_dist


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer_actor = torch.optim.Adam(
            list(self.policy.fc1_actor.parameters()) + 
            list(self.policy.fc2_actor.parameters()) +
            list(self.policy.fc3_actor_mean.parameters()) +
            [self.policy.sigma],
            lr=5e-4
        )
        if self.policy.alghoritm == "AC":
            self.optimizer_critic = torch.optim.Adam(
                list(self.policy.fc1_critic.parameters()) + 
                list(self.policy.fc2_critic.parameters()) + 
                list(self.policy.fc3_critic.parameters()),
                lr=5e-4
            )

        self.I = 1
        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.actions = [] # added
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self, alghoritm="REINFORCE"):
        actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1) # added
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.actions, self.action_log_probs, self.rewards, self.done = [], [], [], [], [], []

        #
        # TASK 2:
        #   - compute discounted returns
        #   - compute policy gradient loss function given actions and returns
        #   - compute gradients and step the optimizer
        #

        if alghoritm == "REINFORCE":
            discounted_returns = discount_rewards(rewards, self.gamma).to(self.train_device).squeeze(-1)
            
            for t in range(len(discounted_returns)):
                g = discounted_returns[t] if not done[t] else 0
                delta = g - 20
                log_prob = self.policy(states[t]).log_prob(actions[t]).sum()
                
                loss = -(self.gamma**t) * delta * log_prob

                self.optimizer_actor.zero_grad()
                loss.backward()
                self.optimizer_actor.step()

        #
        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer
        #

        if alghoritm == "AC":
            if done[0]: # if state is terminal, we don't need to update the policy
                self.i = 1
                return
            
            _, state_value = self.policy(states)
            _, next_state_value = self.policy(next_states) if not done else 0

            one_step_return = rewards + self.gamma * next_state_value
            advantage_term = one_step_return - state_value

            # Critic loss
            critic_loss = F.mse_loss(one_step_return, state_value)

            # Actor loss
            actor_loss = - action_log_probs * advantage_term * self.I
            
            # Total loss
            total_loss = actor_loss + critic_loss

            # Update actor and critic networks
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            
            total_loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()

            self.I *= self.gamma


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        if self.policy.alghoritm == "AC":
            normal_dist, _ = self.policy(x)
        else:
            normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.actions.append(action) # added
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

