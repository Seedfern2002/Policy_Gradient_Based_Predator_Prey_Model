import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

save_dir = './parameter3'
name = 'PG_theta.pth'
fig_dir = './learning_fig3'
torch.manual_seed(666)

'''
MDP model:
    S: the number of individuals in each population
    A: the prey distribution of the population
    P: It is related to the normal distribution of the influence brought by the environment
    R: If the population is stable, get (the number of individuals in this population / 100)
        rewards, else get -100 as rewards
'''


class ANN(nn.Module):
    def __init__(self, n_species):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(n_species, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, n_species)

    def forward(self, input):
        x = torch.relu(self.fc1(input))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        output = torch.sigmoid(self.fc4(x))
        return output


class Agent:
    def __init__(self, num_species, lr=2e-4, batch_size=20, gamma=0.66, learning=True):
        self.lr = lr
        self.batch_size = batch_size
        self.loss = 0.
        self.counter = 0
        self.T = 0  # temperature
        self.gamma = gamma
        self.learning = learning

        self.net = ANN(num_species)
        self.rewards = []
        self.running_average = 0.
        self.log_probs = []

        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)

    def policy(self, state):
        return self.net(state)

    def act(self, discounted_num):
        state = torch.from_numpy(np.float32(discounted_num.T))
        means = 10 * self.policy(state)
        normal = torch.distributions.Normal(means, 4 * np.exp(-self.T * 0.5))
        sample = normal.sample()
        self.log_probs.append((normal.log_prob(sample)))
        return sample.detach().numpy().T

    def learn(self):
        accumulated_rewards = []
        ar = 0.
        for r in self.rewards[::-1]:
            ar = r + self.gamma * ar
            accumulated_rewards.insert(0, ar)
        # calculate the running average(biased estimate)
        self.running_average = (1 - self.lr) * self.running_average + \
                                self.lr * (np.mean(accumulated_rewards) - self.running_average)
        # using normalized rewards
        accumulated_rewards -= self.running_average
        accumulated_rewards = torch.tensor(accumulated_rewards)
        for log_prob, ar in zip(self.log_probs, accumulated_rewards):
            # maximize log_prob*ar equal to minimize -log_prob*ar
            self.loss += torch.sum(-log_prob * ar)
        del self.rewards[:]
        del self.log_probs[:]

        self.counter += 1
        if self.counter == self.batch_size and self.loss != 0.:
            # update parameters per batch_size episodes
            self.counter = 0
            self.T += 1
            self.backward()
        return

    def backward(self):
        self.optimizer.zero_grad()
        self.loss = self.loss / self.batch_size
        self.loss.backward()
        self.optimizer.step()
        self.loss = 0.

    def save(self, dir):
        torch.save(self.net.state_dict(), dir + '/weights.pth')

    def load(self, path):
        print('loading weights from %s' % path)
        self.net.load_state_dict(path)
        print('weights has been loaded')
