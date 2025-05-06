import torch_multiprocessed_old as tm
import torch
from torch.nn.utils import prune
import pygame
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import purported_no_bias_exploration as pnbe
import little_model_exploration as vm


def prune_weights(model, pruning_rate=0.5):
    pruner = prune.L1Unstructured(pruning_rate)
    shape = model.rlif1.recurrent.weight.data.shape
    reshaped_weights = model.rlif1.recurrent.weight.reshape((shape[0] * shape[1]))
    pruned = pruner.prune(reshaped_weights)
    # model.rlif1.recurrent.weight.data = pruned.reshape(shape)

    return pruned.reshape(shape)

def calc_sparsity(model):
    zeros = torch.sum(model.rlif1.recurrent.weight.data == 0)
    total = model.rlif1.recurrent.weight.data.numel()
    sparsity = zeros.item() / total
    return sparsity

def disp_graph(model):
    # Currently colors our single edge of interest yellow!
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
    nx.set_edge_attributes(G, 'grey', 'color')
    G[61][7]['color'] = 'yellow'
    node_list = set(list(G.neighbors(61)) + list(G.neighbors(7)))
    node_colors = [c[1] for c in G.nodes.data('color')]
    edge_colors = [c[2] for c in G.edges.data('color')]
    nx.draw_circular(G, node_color=node_colors, nodelist=node_list, edge_color=edge_colors, with_labels=False)
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

    model = tm.RSNN2(num_inputs=1, num_hidden=80, num_outputs=1)
    filename = "big_model.pth"
    state_dict = torch.load(filename, weights_only=True)
    
    model.load_state_dict(state_dict)

    print(calc_sparsity(model))
    # big_prune = prune_weights(model, pruning_rate=0.1615) # fails
    little_prune = prune_weights(model, pruning_rate=0.15) # doesn't fail
    # print(torch.equal(big_prune, little_prune))
    # print(torch.sum(big_prune - little_prune != 0))
    # diffs = torch.where(big_prune - little_prune != 0)
    # print(diffs)
    # print(f'big: {big_prune[diffs]}')
    # print(f'little: {little_prune[7, 61].detach().numpy()}')
    # little_prune[7, 61] = 0
    model.rlif1.recurrent.weight.data = little_prune
    # df = pd.DataFrame(little_prune.detach().numpy())
    # print(df)
    # df.to_csv('little.csv')
    # Why does it fail???
    print(calc_sparsity(model))
    
    pygame.init()
    WIDTH, HEIGHT = 800, 400
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pnbe.visualize_model(model, tm.DinosaurGame, (100,))
    # vm.visualize_model(model, tm.DinosaurGame, (100,))

    G = disp_graph(model)

    # Set all weights to be positive
    for u, v, w in G.edges.data('weight'):
        if w < 0:
            G[u][v]['weight'] = abs(w)

    # Calculate Laplacian
    laplacian = random_walk_normalized_laplacian(G)

    # Calculate eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(laplacian)
    print(eigenvalues.shape)
    eigenvalues = np.real(eigenvalues) # Ensure eigenvalues are real
    eigenvalues = np.round(eigenvalues, 3) # Round to 3 decimal places for better visualization
    print(eigenvalues)

    # Graph eigenvector histogram
    sns.set(style="whitegrid")
    sns.kdeplot(eigenvalues)
    plt.title('Eigenvalue Distribution of Random Walk Normalized Laplacian')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()