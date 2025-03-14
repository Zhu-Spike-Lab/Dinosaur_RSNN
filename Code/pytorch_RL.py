# import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import Ivyer as ea
import pygame


# env = gym.make("CartPole-v1")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# TODO: FEED FORWARD NETWORK BAD
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # Of shape # inputs -> 128
        self.layer1 = nn.Linear(n_observations, 128)
        # 128 -> 128
        self.layer2 = nn.Linear(128, 128)
        # 128 -> # outputs
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # Feed forward network
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DinosaurGame():
    def __init__(self, maximum = None):
        self.time = 0
        self.alive = True
        self.WIDTH, self.HEIGHT = 800, 400

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        # Dinosaur settings
        self.dino_size = 50
        self.dino_x = 80
        self.dino_y = self.HEIGHT - self.dino_size - 40
        self.dino_vel_y = 0
        self.gravity = 2
        self.jumping = False

        # Obstacle settings
        self.obstacle_width = 20
        self.obstacle_height = 50
        self.obstacle_x = self.WIDTH # 800
        self.obstacle_y = self.HEIGHT - self.obstacle_height - 40
        self.obstacle_speed = 10

        # Game settings
        self.running = True
        self.jumping = False
        self.score = 0
        self.scored = 0
        # self.font = pygame.font.Font(None, 36)

        # Calc'd constants
        self.cross_time = self.WIDTH / self.obstacle_speed

        # Game loop
        # self.clock = pygame.time.Clock()


        if maximum:
            self.maximum = maximum
        else:
            self.maximum = False

    # def __len__(self):
    #     return self.cross_time * self.maximum

    # def __getitem__(self, idx):
    #     return self.get_input(idx)
    def get_input(self):

        obs_pos = self.obstacle_x
        if 441 <= obs_pos <= 459:
            observation = 1
        else:
            observation = 0

        info = 0
        # reward = 10 * self.scored if self.scored else 1
        reward = 1 * self.scored if self.alive else -10
        self.scored = 0
        return observation, float(reward), not self.alive, False, info

    def step(self, action):
        
        # Event handling
        # AI output: jump or not (each frame ig)
        if action >= 1 and not self.jumping:
            self.jumping = True
            self.dino_vel_y = -20

        # Dinosaur movement (jumping)
        if self.jumping:
            self.dino_y += self.dino_vel_y
            self.dino_vel_y += self.gravity
            if self.dino_y >= self.HEIGHT - self.dino_size - 40:
                self.dino_y = self.HEIGHT - self.dino_size - 40
                self.jumping = False

        # Obstacle movement: Resets the same obstacle
        # Can give AI input every time this if triggers
        self.obstacle_x -= self.obstacle_speed
        if self.obstacle_x < -self.obstacle_width:
            self.obstacle_x = self.WIDTH
            self.score += 1
            self.scored = 1
            # # Speeding up
            # if self.score % 10 == 0:
            #     self.obstacle_speed += 1

        # Collision detection
        self.alive = (self.dino_x + self.dino_size <= self.obstacle_x or self.obstacle_x + self.obstacle_width <= self.dino_x or self.dino_y + self.dino_size <= self.obstacle_y or self.obstacle_y + self.obstacle_height <= self.dino_y)

        if self.maximum:
            if self.score >= self.maximum:
                self.alive = False

        self.time += 1

        # # Debug
        # if generation >= 30:
        #     time.sleep(0.00001)
        # time.sleep(0.001)
        # print(f'time: {self.time} action: {action}\ndino_y: {self.dino_y}\ndino_vel_y: {self.dino_vel_y}\njumping: {self.jumping}')

    def visualize(self, screen):
        # Draws game state in pygame
        dino_rect = pygame.Rect(self.dino_x, self.dino_y, self.dino_size, self.dino_size)
        obstacle_rect = pygame.Rect(self.obstacle_x, self.obstacle_y, self.obstacle_width, self.obstacle_height)

        # Draw dinosaur and obstacle
        pygame.draw.rect(screen, self.BLACK, dino_rect)
        pygame.draw.rect(screen, self.BLACK, obstacle_rect)

        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, self.BLACK)
        screen.blit(score_text, (10, 10))

#########################################################################################

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 42800
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# # Get number of actions from gym action space
# n_actions = 2
# # Get the number of state observations
# state, info = env.reset()
# n_observations = 1

# policy_net = DQN(n_observations, n_actions).to(device)
# target_net = DQN(n_observations, n_actions).to(device)
policy_net = ea.RSNN2(num_inputs=1, num_hidden=15, num_outputs=1).to(device)
target_net = ea.RSNN2(num_inputs=1, num_hidden=15, num_outputs=1).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0

game = DinosaurGame()

# Chooses whether to use the model's action or a random action
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # return policy_net(state).max(1).indices.view(1, 1)
            return policy_net(state)[0].max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.choice([0, 1])]], device=device, dtype=torch.long)


episode_scores = []

# Visualizer
def plot_scores(show_result=False):
    plt.figure(1)
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores_t.numpy())
    # Take 100 episode averages and plot them too
    if len(scores_t) >= 100:
        means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


####################### Training loop (this func performs 1 step of optimization)
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch)[0].gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states)[0].max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


################################################ Train that bish!
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    # state, info = env.reset()
    state = 0
    state = torch.tensor([state,], dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        game.step(action)
        observation, reward, terminated, truncated, _ = game.get_input()
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
            # Reset game
            game = DinosaurGame()
        else:
            next_state = torch.tensor([observation,], dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state,action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_scores.append(game.score)
            plot_scores()
            break

print('Complete')
plot_scores(show_result=True)
plt.ioff()
plt.show()