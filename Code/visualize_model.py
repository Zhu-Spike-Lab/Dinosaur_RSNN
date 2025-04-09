import torch_multiprocessed as tm
import torch
import pandas as pd
import pygame
import numpy as np



def main():

    # Define the parameters for the evolutionary process
    pop_size = 20
    num_generations = 9
    n_offspring = 20
    mutation_rate = 0.5

    # Create the Evolution object and run the evolution process
    # 
    evolution = tm.Evolution(tm.RSNN2, (), {'num_inputs':1, 'num_hidden':20, 'num_outputs':1})
    # Note: evolve method was altered from Ivyer's OG code so we code Dino-ify it :)
    # done: change evolve, custom loss
    # game_args: maximum=100
    best_model, fitness, final_population = evolution.evolve(pop_size, n_offspring, num_generations, tm.DinosaurGame, (100,), mutation_rate)
    tm.visualize_model(best_model, tm.DinosaurGame, (100,))

    # Save the best model's state dictionary
    torch.save(best_model.state_dict(), 'best_modelXXX.pth')


# if __name__ == '__main__':
#     main()

if __name__ == '__main__':
    filename = "connection_matrix.csv"
    # with open(filename) as f:
    #     ncols = len(f.readline().split(','))
    #     params = {}
    #     for line in f.readlines():
    #         line = line.strip()
    #         line_list = line.split(',')
    #         if line_list[0] not in list(params.keys()):
    #             params[line_list[0]] = {'rows': 0, 'cols': 1 + len(list(filter(None, line_list[1:])))} 

    #         params[line_list[0]]['rows'] += 1
        
    # # input_mat = np.loadtxt(filename, usecols=1, skiprows=1, max_rows=input_rows, delimiter=',', ndmin=2) # This assumes the CSV is semicolon-separated and there's a header row and an index column
    # # conn_mat = np.loadtxt(filename, usecols=range(1,ncols), skiprows=input_rows+1, max_rows=conn_rows, delimiter=',')
    # # output_mat = np.loadtxt(filename, usecols=range(1,ncols), skiprows=input_rows+conn_rows+1, max_rows=output_rows, delimiter=',', ndmin=2)
    # # print(input_mat)
    # # print(pd.DataFrame(conn_mat))
    # # print(output_mat)
    # # state_dict = torch.load('/Users/dfairborn/Documents/Code Documents/Git Repos/Dinosaur_RSNN/best_modelXXX.pth', weights_only=True)
    model = tm.RSNN2(num_inputs=1, num_hidden=20, num_outputs=1)
    # # print(list(params.keys()))
    # rows = 1
    # for name, param in model.named_parameters():
    #     new = np.loadtxt(filename, usecols=range(1, params[name]['cols']), skiprows=rows, max_rows=params[name]['rows'], delimiter=',', ndmin=2)
    #     rows += params[name]['rows']
    #     model.state_dict()[name] = torch.Tensor(new)
    # print(list(model.named_modules()))
    # print(model.l1.weight.data.shape)
    # print(input_mat.shape)
    # print(model.rlif1.recurrent.weight.data.shape)
    # print(conn_mat.shape)
    # print(model.l2.weight.data.shape)
    # print(output_mat.shape)
    # print(list(model.named_modules()))

    # print('debug')
    state_dict = torch.load('best_model Wed Apr  9 00:06:29 2025.pth', weights_only=True)
    print(model.state_dict == state_dict)
    model.load_state_dict(state_dict)
    # print(state_dict)
    pygame.init()
    WIDTH, HEIGHT = 800, 400
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    tm.visualize_model(model, tm.DinosaurGame, (100,))