import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Our code is based on the implementation of TD3: https://github.com/sfujim/TD3

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))
    def feature(self, obs):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))

        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)


    def forward(self, state, action, feature=False):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        f1 = F.relu(self.l2(q1))
        q1 = self.l3(f1)

        if feature:
            return q1, f1,
        else:
            return q1


    def Q1(self, state, action, feature=False):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        f1 = F.relu(self.l2(q1))
        q1 = self.l3(f1)
        if feature:
            return q1, f1
        else:
            return q1


class BEER(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.total_it = 0
        self.batch_size=256
        self.beta = 1e-3

    def select_action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).reshape(1, -1).to(device)
            action = self.actor(state)
            return action.cpu().numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            # Compute the target Q value
            target_Q, target_feature= self.critic_target(next_state, next_action, feature=True)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q, current_feature = self.critic(state, action, feature=True)
        with torch.no_grad():
            _, next_feature1 = self.critic(next_state, self.actor(next_state), feature=True)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        # compute upper bound
        with torch.no_grad():
            parameter = self.critic.l3.get_parameter("weight")
            nn_norm_square = torch.norm(parameter) ** 2  # L2 norm
            upper_bound1 = (0.5 / 0.99) * (torch.norm(current_feature, dim=1, keepdim=True) ** 2 +
                            0.99 ** 2 * torch.norm(next_feature1, dim=1, keepdim=True) ** 2 - reward ** 2 / nn_norm_square)
            upper_bound1 = upper_bound1 / (torch.norm(current_feature, dim=1, keepdim=True) *
                                           torch.norm(target_feature, dim=1, keepdim=True))
        beer_loss = self.beta * (F.relu(F.cosine_similarity(current_feature, target_feature, dim=1, eps=1e-6) - upper_bound1.reshape(1,-1))).mean()
        critic_loss = critic_loss + beer_loss
        Q_function_loss = critic_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        Q_function_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = - self.critic.Q1(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)