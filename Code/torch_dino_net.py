# TODO: Tweak threshold in evaluate_model function a tad more perhaps

# NOT NEEDED: make the game(index, choice) function for CustomLoss. Should calculate the game state at index, determine if choice results in a loss, then return true for a good decision and false for a bad one
# DONE: Fix forward: forward(self, outputs, game, index, criticality, firing_rate, synchrony_fano_factor)
# DONE: Add in a consideration of score into the loss function
# DONE: Just change evaluate model to use the proper game function... :)

import numpy as np
# import pygame
import time
import Ivyer as ea
import torch

# Perhaps turn this into a dataset
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
        # Return input: 1 if obstacle, 0 if not (obstacles appear every few timesteps)
        # TODO: Give position of obstacle just for funsies
        # Could give distance btw obstacle and character
        # Could give its own position & object's position
        # cross_time = self.WIDTH / self.obstacle_speed
        # obs_pos = (time % cross_time) * self.obstacle_speed
        obs_pos = self.obstacle_x
        if 441 <= obs_pos <= 459:
            return 1
        else:
            return 0
        # return [1 if self.time % 45 == 0 else 0]

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
            # # Speeding up
            # if self.score % 10 == 0:
            #     self.obstacle_speed += 1

        # # Collision detection
        # dino_rect = pygame.Rect(self.dino_x, self.dino_y, self.dino_size, self.dino_size)
        # obstacle_rect = pygame.Rect(self.obstacle_x, self.obstacle_y, self.obstacle_width, self.obstacle_height)
        # if dino_rect.colliderect(obstacle_rect):
        #     self.alive = False
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



def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    torch.set_default_device(device)

    # pygame.init()

    # Define the parameters for the evolutionary process
    pop_size = 10
    num_generations = 2500
    n_offspring = 10
    # mutation_rate = 0.05
    mutation_rate = 0.5

    # Create the Evolution object and run the evolution process
    # 
    evolution = ea.Evolution(ea.RSNN2, (), {'num_inputs':1, 'num_hidden':15, 'num_outputs':1})
    # Note: evolve method was altered from Ivyer's OG code so we code Dino-ify it :)
    # done: change evolve, custom loss
    # game_args: maximum=100
    best_model, fitness, final_population = evolution.evolve(pop_size, n_offspring, num_generations, DinosaurGame, (100,), mutation_rate)
    ea.visualize_model(best_model, DinosaurGame, (100,))

    # Save the best model's state dictionary
    torch.save(best_model.state_dict(), 'best_model.pth')

    # Usage example after evolution process
    initial_models = evolution.populate(pop_size)
    best_perf = evolution.decode_population(evolution.encode_population([best_model]), best_model)

    ea.plot_connectivity_changes_heat(initial_models, final_population)


    final_models = evolution.decode_population(evolution.encode_population([best_model]), best_model)
    ea.plot_connectivity_changes_line(initial_models, final_models)


    ea.print_model_performance(best_model, DinosaurGame, (100,))


if __name__ == '__main__':
    main()