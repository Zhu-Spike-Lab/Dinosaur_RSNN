import torch_multiprocessed as tm
import torch
from torch.nn.utils import prune
import pygame
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import imageio
import io
from PIL import Image


fps = 60
video_size = (800, 400)  # Width, Height
pip_size = (400,200)


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

def generate_graph(model):
    G = nx.DiGraph(model.rlif1.recurrent.weight.data.numpy().T)
    nx.set_node_attributes(G, 'black', 'color')
    
    G.nodes[0]['color'] = 'green'

    for u, v, w in G.edges.data('weight'):
        if w > 0:
            G[u][v]['color'] = 'blue'
        else:
            G[u][v]['color'] = 'red'

    return G


def disp_graph(graph, spikes=None):
    G = graph

    
    # for u, v, w in G.edges.data('weight'):
    #         if w > 0:
    #             G[u][v]['color'] = 'blue'
    #         else:
    #             G[u][v]['color'] = 'red'

    # TODO: WHEN SAVING AS A VIDEO, NEED TO CHANGE THIS
    if spikes is not None:
        edgelist = []
        spikes = spikes.flatten()
        
        
        for i, spike in enumerate(spikes):
            if spike > 0:
                G.nodes[i]['color'] = 'orange'
    
    else:
        edgelist = G.edges 
        for u, v, w in G.edges.data('weight'):
            if w > 0:
                G[u][v]['color'] = 'blue'
            else:
                G[u][v]['color'] = 'red'
    

    edgelist = G.edges 

    node_colors = [c[1] for c in G.nodes.data('color')]
    edge_colors = [c[2] for c in G.edges.data('color')]
    with plt.xkcd():
        nx.draw_circular(G, node_color=node_colors, edgelist=edgelist, edge_color=edge_colors, with_labels=False, connectionstyle="arc3,rad=0.1")
    plt.show(block=False)

    # plt.pause(0.001)
    
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

def visualize_model(model, game_class, game_args):
    # Initialize Pygame
    pygame.init()
    # Screen dimensions
    global WIDTH, HEIGHT, font
    WIDTH, HEIGHT = 800, 400
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Dino Game")
    # font = pygame.font.Font(None, 36)

    game = game_class(*game_args)

    model.eval()
    graph = generate_graph(model)
    disp_graph(graph)

    with torch.no_grad():
        with imageio.get_writer("vid.mp4", fps=fps, codec='libx264', format='ffmpeg') as writer:
            # Run the game
            while game.alive:
                start = time.time()

                inputs = torch.tensor([[game.get_input(),]], dtype=torch.float)
                outputs, spikes = model(inputs)


                choice = spikes[0,0]
                game.step(choice)

                disp_graph(graph, spikes=spikes.numpy())

                # Visualize
                screen.fill((255, 255, 255))
                game.visualize(screen)

                # Save frame to video
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                plt.close()
                buf.seek(0)
                pip_img = Image.open(buf).convert("RGB")
                pip_img = pip_img.resize(pip_size)
                pip_array = np.array(pip_img)
                
                # --- Convert pygame screen to numpy array ---
                pg_array = pygame.surfarray.array3d(screen).swapaxes(0, 1)  # Shape: (H, W, 3)
    
                # --- Composite picture-in-picture ---
                frame = pg_array.copy()  # Ensure writable
                frame[0:pip_array.shape[0], 0:pip_array.shape[1], :] = pip_array
    
                # --- Write frame ---
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
                
                writer.append_data(frame)
                print(f'Frame added to video: {game.score / 50 * 100}% complete')

                # # Update display
                # pygame.display.flip()


                # 60 fps (time.time is in nanoseconds)
                # rest = (20000000 - (time.time() - start))/1000000000
                # _ = input('next frame: ')
                # if rest > 0:
                    # time.sleep(rest)
            
            print(f'Score: {game.score}')


if __name__ == '__main__':

    model = tm.RSNN2(num_inputs=1, num_hidden=40, num_outputs=1)
    filename = "first_no_bias_sparse_hope.pth" # With new improvements
    state_dict = torch.load(filename, weights_only=True)
    
    model.load_state_dict(state_dict)

    # print(calc_sparsity(model))
    # prune_weights(model, pruning_rate=0.5) 
    print(calc_sparsity(model))
    
    pygame.init()
    WIDTH, HEIGHT = 800, 400
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    visualize_model(model, tm.DinosaurGame, (100,))

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
