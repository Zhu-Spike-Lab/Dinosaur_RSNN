# TODO: Break visualization into parts. Add neural net visualization for any # Neurons

# TODO: Directions:
# Visualize the connections and weights (also add saving of neural nets)
# Start by saving most fit of all time, then maybe go to most fit of each generation. Save spiking activity as well?
# Penalize extra jumps
# Change speed of obstacles and see if model can learn
# CHECK: Make sure weights can decrease
# See how number of neurons affects the training process (scaling laws)
# Specifying that neurons that receive input cannot give output
# Try combining output of multiple neurons to provide final output
# Specifying neuron types
# Integrate with github
# Clean up files
# Vary frequency of obstacles (keep speed constant though) - to make sure model is truly responding to the input
# Use other training models

import numpy as np
import pygame
import time
import itertools
import random
import time


class SpikingNeuron:
    def __init__(self, threshold=0.5, decay=0.9):
        self.potential = 0.0
        self.threshold = threshold
        self.decay = decay

    def receive_input(self, input_value):
        self.potential += input_value

    def step(self):
        # Neuron fires if potential exceeds threshold
        if self.potential >= self.threshold:
            self.potential = 0  # Reset after spike
            return 1  # Spike (output 1)
        else:
            # Decay potential over time
            self.potential *= self.decay
            return 0  # No spike (output 0)

class RecurrentSpikingNeuralNetwork:
    # Each neuron has set threshold and decay
    # What changes are the weights between connections
    # self.neurons needs to be a map (or some sort of list). neuron: [neurons it connects to]
    # each neuron needs a label...
    # need some way to track all the connections, and all the weights
    # Do we have 1 master list of all connections
    # Or each neuron keeps track of its own
    # How does the recurrent bit work..... it's in the connections
    # Wow ChatGPT really did quite an awwful job
    # Ok let's fix it
    def __init__(self, num_inputs, num_neurons):
        self.neurons = [SpikingNeuron() for _ in range(num_neurons)] # List of neurons

        self.input_map = {i: np.floor(np.random.random() * num_neurons) for i in range(num_inputs)}
        self.input_weights = [np.random.random() for _ in range(num_inputs)]

        # List of connections
        # TODO: give connections a direction
        self.connections = []
        for combo in itertools.permutations(range(num_neurons), 2): # Iterate over all possible combinations
            # 20% to become actual connection
            if np.random.random() >= 0.8:
                # combo = list(combo)
                # np.random.shuffle(combo)
                # TODO: ^ Probably not necessary
                self.connections.append(combo)
        
        # Check to make sure each neuron has a connection, otherwise connect it to i-1 (will work for 0 too, methinks)
        # Maybe unhelpful or unneeded
        temp = []
        for i in range(len(self.connections)):
            temp.append(self.connections[i][0])

        for i in range(num_neurons):
            try:
                temp.index(i)

            except ValueError:
                self.connections.append([i, i-1])

        self.connection_weights = np.random.rand(len(self.connections))

        self.spikes = []

    def forward(self, inputs):
        # Every time step:
        # All spikes are calculated & recorded. Place into spike map
        # All neurons decay
        self.spikes = []
        for neuron in self.neurons:
            spike = neuron.step()
            self.spikes.append(spike)

        # All inputs are sent to their proper neuron, multiplied by their proper weight
        for i, input in enumerate(inputs):
            self.neurons[int(self.input_map[i])].receive_input(input * self.input_weights[i])

        # All connections are resolved
        for i, combo in enumerate(self.connections):
            first_neuron = combo[0]
            second_neuron = combo[1]
            weight = self.connection_weights[i]
            self.neurons[second_neuron].receive_input(self.spikes[first_neuron] * weight)

        # What value to return?
        return self.spikes

    def mutate(self, mutation_rate=0.1):
        # Random mutation for evolutionary algorithm
        # TODO: currently weights cannot decrease
        for i in range(self.input_weights.shape[0]):
            if random.random() < mutation_rate:
                self.input_weights[i] += np.random.normal() * 0.1

        for i in range(len(self.connection_weights)):
            if random.random() < mutation_rate:
                self.connection_weights[i] += np.random.normal() * 0.1

    def visualize(self, screen):
        # Need to visualize inputs
        # Need to visualize neurons
        # Need to visualize connections
        BLACK = (0, 0, 0)
        # Draw neural network
        
        num_neurons = len(self.neurons)
        if num_neurons % 2 == 0:
            for i in range(num_neurons//2):
                pygame.draw.circle(screen, BLACK, (720 - (50*i), 50), 15, width=self.spikes[2*i])
                pygame.draw.circle(screen, BLACK, (720 - (50*i), 100), 15, width=self.spikes[2*i + 1])
        
        # TODO: Visualize off number of neurons


        # obs = 1 if self.obstacle_x == 250 else 0
        # pygame.draw.circle(screen, (255, 0, 0), (670, 150), 15, width=obs)

    def save(self, filename):
        if not filename.endswith('.csv'):
            filename = filename + '.csv'
        
        data = [[''],]

        for i in range(len(self.neurons) + 1):
            # Set up connection format
            data[0].append(str(chr(65 + i)))
            data.append([chr(65 + i), *list(0 for _ in range(len(self.neurons) + 1))])


        data = np.array(data)
        # Insert weights according to connections
        for i, (x, y) in enumerate(self.connections):
            # Correct for labels
            x += 1
            y += 1
            data[x, y] = self.connection_weights[i]


        np.savetxt(filename, data, fmt='%s', delimiter=';')

        return filename

class DinosaurGame:
    def __init__(self, maximum = None):
        self.time = 0
        self.alive = True
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        # Dinosaur settings
        self.dino_size = 50
        self.dino_x = 80
        self.dino_y = HEIGHT - self.dino_size - 40
        self.dino_vel_y = 0
        self.gravity = 2
        self.jumping = False

        # Obstacle settings
        self.obstacle_width = 20
        self.obstacle_height = 50
        self.obstacle_x = WIDTH
        self.obstacle_y = HEIGHT - self.obstacle_height - 40
        self.obstacle_speed = 10

        # Game settings
        self.running = True
        self.jumping = False
        self.score = 0
        self.font = pygame.font.Font(None, 36)

        # Game loop
        self.clock = pygame.time.Clock()

        # Initialize Pygame
        # pygame.init()

        # # Screen dimensions
        # if not screen:
        #     self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        # else:
        #     self.screen = screen
        # pygame.display.set_caption("Dino Game")

        if maximum:
            self.maximum = maximum
        else:
            self.maximum = False


    def get_input(self):
        # Return input: 1 if obstacle, 0 if not (obstacles appear every few timesteps)
        return [1 if self.obstacle_x == 350 else 0]

    def step(self, action):
        # Event handling
        # AI output: jump or not (each frame ig)
        if action == 1 and not self.jumping:
            self.jumping = True
            self.dino_vel_y = -20

        # Dinosaur movement (jumping)
        if self.jumping:
            self.dino_y += self.dino_vel_y
            self.dino_vel_y += self.gravity
            if self.dino_y >= HEIGHT - self.dino_size - 40:
                self.dino_y = HEIGHT - self.dino_size - 40
                self.jumping = False

        # Obstacle movement: Resets the same obstacle
        # Can give AI input every time this if triggers
        self.obstacle_x -= self.obstacle_speed
        if self.obstacle_x < -self.obstacle_width:
            self.obstacle_x = WIDTH
            self.score += 1

        # Collision detection
        dino_rect = pygame.Rect(self.dino_x, self.dino_y, self.dino_size, self.dino_size)
        obstacle_rect = pygame.Rect(self.obstacle_x, self.obstacle_y, self.obstacle_width, self.obstacle_height)
        if dino_rect.colliderect(obstacle_rect):
            self.alive = False

        if self.maximum:
            if self.score >= self.maximum:
                self.alive = False

        self.time += 1
        # # Debug
        # if generation >= 30:
        #     time.sleep(0.00001)
        # time.sleep(0.001)
        # print(f'time: {self.time} action: {action}\ndino_y: {self.dino_y}\ndino_vel_y: {self.dino_vel_y}\njumping: {self.jumping}')

    def visualize(self, screen, inputs):
        # Draws game state in pygame

        # Collision detection
        dino_rect = pygame.Rect(self.dino_x, self.dino_y, self.dino_size, self.dino_size)
        obstacle_rect = pygame.Rect(self.obstacle_x, self.obstacle_y, self.obstacle_width, self.obstacle_height)

        # Draw dinosaur and obstacle
        pygame.draw.rect(screen, self.BLACK, dino_rect)
        pygame.draw.rect(screen, self.BLACK, obstacle_rect)

        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, self.BLACK)
        screen.blit(score_text, (10, 10))

class EvolutionaryAlgorithm:
    def __init__(self, population_size=20, mutation_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [RecurrentSpikingNeuralNetwork(num_inputs=1, num_neurons=16) for _ in range(population_size)]
    
    def evaluate_fitness(self, network, screen=None, maximum=False):
        game = DinosaurGame(maximum=maximum)
        total = 0
        while game.alive:
            inputs = game.get_input()
            outputs = network.forward(inputs)
            total += np.sum(outputs)
            game.step(outputs[0])

            # Visualize
            if screen:
                screen.fill((255, 255, 255))
                game.visualize(screen, outputs)
                network.visualize(screen)
                # Update display
                pygame.display.flip()
                time.sleep(0.01)

        # if total > 1:
        #     print(total)
        return game.score

    def run_generation(self, screen=None, maximum=False):
        # Evaluate fitness of each network
        fitness_scores = [self.evaluate_fitness(network, screen=screen, maximum=maximum) for network in self.population]
        
        # Select top-performing networks
        sorted_population = [network for _, network in sorted(zip(fitness_scores, self.population), key=lambda tuple: tuple[0], reverse=True)]
        self.population = sorted_population[:self.population_size // 2]
        
        # Repopulate with mutated copies
        while len(self.population) < self.population_size:
            parent = random.choice(self.population)
            child = RecurrentSpikingNeuralNetwork(num_inputs=1, num_neurons=16)
            ######## TODO: Implement mutation and inheritance
            child.input_weights = np.copy(parent.input_weights)
            child.connection_weights = np.copy(parent.connection_weights)
            child.connections = np.copy(parent.connections)
            child.mutate(self.mutation_rate)
            ########
            self.population.append(child)
        
        return max(fitness_scores)


def main():
    # Initialize Pygame
    pygame.init()
    # Screen dimensions
    global WIDTH, HEIGHT, font
    WIDTH, HEIGHT = 800, 400
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Dino Game")
    font = pygame.font.Font(None, 36)

    # Run evolutionary algorithm
    ea = EvolutionaryAlgorithm(population_size=10, mutation_rate=60)
    global generations
    generations = 10001
    max_score = 11
    for generation in range(generations):
        max_fitness = ea.run_generation(screen=None, maximum=100)
        print(f"Generation {generation}, Max Fitness: {max_fitness}, Max Score: {max_score}")
        if max_fitness > max_score:
            max_score = max_fitness
            screen = pygame.display.set_mode((WIDTH, HEIGHT))
            print(ea.population[0].save(f'network: {time.asctime()}'))
            ea.evaluate_fitness(ea.population[0], screen=screen, maximum=100)
        
        if generation >= 1000:
            main()

main()