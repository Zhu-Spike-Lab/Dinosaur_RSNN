import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def generate_graph(model):
    G = nx.DiGraph(model.rlif1.recurrent.weight.data.numpy().T)
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


def cal_stats(model):
    G = generate_graph(model)

    # Set all weights to be positive
    for u, v, w in G.edges.data('weight'):
        if w < 0:
            G[u][v]['weight'] = abs(w)

    # Calculate Laplacian
    laplacian = random_walk_normalized_laplacian(G)

    # Calculate eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(laplacian)
    eigenvalues = np.real(eigenvalues) # Ensure eigenvalues are real
    eigenvalues = np.round(eigenvalues, 3) # Round to 3 decimal places for better visualization

    return eigenvalues, eigenvectors


def graph_eigen_hist(model):
    eigenvalues, eigenvectors = cal_stats(model)
    
    # Graph eigenvector histogram
    sns.set(style="whitegrid")
    sns.kdeplot(eigenvalues)
    plt.title('Eigenvalue Distribution of Random Walk Normalized Laplacian')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

