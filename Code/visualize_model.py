import torch_multiprocessed as tm
import torch
from torch.nn.utils import prune
import pygame
import networkx as nx
import matplotlib.pyplot as plt


def prune_weights(model, pruning_rate=0.5):
    pruner = prune.L1Unstructured(0.5)
    shape = model.rlif1.recurrent.weight.data.shape
    reshaped_weights = model.rlif1.recurrent.weight.reshape((shape[0] * shape[1]))
    pruned = pruner.prune(reshaped_weights)
    model.rlif1.recurrent.weight.data = pruned.reshape(shape)

    return model

def calc_sparsity(model):
    zeros = torch.sum(model.rlif1.recurrent.weight.data == 0)
    total = model.rlif1.recurrent.weight.data.numel()
    sparsity = zeros.item() / total
    return sparsity

def disp_graph(model):
    G = nx.DiGraph(model.rlif1.recurrent.weight.data.numpy().T)
    nx.set_node_attributes(G, 'grey', 'color')
    for u, v, w in G.edges.data('weight'):
        if w > 0:
            G.nodes[u]['color'] = 'blue'
            G[u][v]['color'] = 'blue'
        else:
            G.nodes[u]['color'] = 'red'
            G[u][v]['color'] = 'red'
    G.nodes[0]['color'] = 'green'
    node_colors = [c[1] for c in G.nodes.data('color')]
    edge_colors = [c[2] for c in G.edges.data('color')]
    nx.draw_circular(G, node_color=node_colors, edge_color=edge_colors, with_labels=False)
    plt.show()


if __name__ == '__main__':

    model = tm.RSNN2(num_inputs=1, num_hidden=20, num_outputs=1)
    # 'best_model Wed Apr  9 00:06:29 2025.pth'
    filename = "best_model Wed Apr  9 00:06:29 2025.pth"
    state_dict = torch.load(filename, weights_only=True)
    
    model.load_state_dict(state_dict)

    # print(calc_sparsity(model))
    prune_weights(model, pruning_rate=0.8)
    # print(calc_sparsity(model))
    
    pygame.init()
    WIDTH, HEIGHT = 800, 400
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    tm.visualize_model(model, tm.DinosaurGame, (100,))

    disp_graph(model)


# Old opening of csv file to get the state_dict

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
