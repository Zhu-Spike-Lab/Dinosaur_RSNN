import torch_multiprocessed as tm
import torch
from torch.nn.utils import prune
import pygame
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import little_model_exploration as lme


def prune_weights(model, pruning_rate=0.5):
    pruner = prune.L1Unstructured(pruning_rate)
    shape = model.rlif1.recurrent.weight.data.shape
    reshaped_weights = model.rlif1.recurrent.weight.reshape((shape[0] * shape[1]))
    pruned = pruner.prune(reshaped_weights)
    model.rlif1.recurrent.weight.data = pruned.reshape(shape)

    return pruned.reshape(shape)

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

    return G

def random_walk_normalized_laplacian(graph):
    """
    Computes the random walk normalized Laplacian matrix of a graph.

    Args:
        graph: A NetworkX graph object.

    Returns:
        A NumPy array representing the random walk normalized Laplacian matrix.
    """
    # Compute the adjacency matrix
    adj_matrix = nx.adjacency_matrix(graph)
    adj_matrix = np.array(adj_matrix.todense())

    # Compute the degree matrix
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))

    # Compute the inverse of the degree matrix
    degree_matrix_inv = np.linalg.inv(degree_matrix)

    # Compute the random walk normalized Laplacian
    laplacian_rw = np.eye(graph.number_of_nodes()) - np.dot(degree_matrix_inv, adj_matrix)

    return laplacian_rw


if __name__ == '__main__':

    model = tm.RSNN2(num_inputs=1, num_hidden=40, num_outputs=1)
    # filename = "best_model Wed Apr  9 00:06:29 2025.pth" # Cool behavior where the jumping slows down @ 10
    # filename = "best_model Wed Apr  9 15:58:17 2025.pth" # Nice jumping behavior
    # filename = "best_model Fri Apr 11 11:12:49 2025.pth"
    # filename = "best_model Sat Apr 12 18:30:28 2025.pth" # Most recent: should have everything enforced & no extra input & 128 neurons
    filename = "first_no_bias_sparse_hope.pth" # With new improvements
    state_dict = torch.load(filename, weights_only=True)
    
    model.load_state_dict(state_dict)

    # print(calc_sparsity(model))
    # prune_weights(model, pruning_rate=0.5) 
    print(calc_sparsity(model))
    
    pygame.init()
    WIDTH, HEIGHT = 800, 400
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    lme.visualize_model(model, tm.DinosaurGame, (100,))
    tm.print_model_performance(model, tm.DinosaurGame, (100,))

    G = disp_graph(model)

    # Set all weights to be positive
    for u, v, w in G.edges.data('weight'):
        if w < 0:
            G[u][v]['weight'] = abs(w)

    # Calculate Laplacian
    laplacian = random_walk_normalized_laplacian(G)

    # Calculate eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(laplacian)
    # print(eigenvalues.shape)
    eigenvalues = np.real(eigenvalues) # Ensure eigenvalues are real
    eigenvalues = np.round(eigenvalues, 3) # Round to 3 decimal places for better visualization
    # print(eigenvalues)

    # Graph eigenvector histogram
    sns.set(style="whitegrid")
    sns.kdeplot(eigenvalues)
    plt.title('Eigenvalue Distribution of Random Walk Normalized Laplacian')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()



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


