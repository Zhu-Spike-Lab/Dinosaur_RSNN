import torch
import pygame
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import imageio
import io
from PIL import Image


fps = 60
video_size = (800, 400)  # Width, Height
pip_size = (400,200)


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

    nx.set_node_attributes(G, 'black', 'color')
    
    G.nodes[0]['color'] = 'green'

    
    # for u, v, w in G.edges.data('weight'):
    #         if w > 0:
    #             G[u][v]['color'] = 'blue'
    #         else:
    #             G[u][v]['color'] = 'red'

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

    nx.draw_circular(G, node_color=node_colors, edgelist=edgelist, edge_color=edge_colors, with_labels=False, connectionstyle="arc3,rad=0.1")
    plt.show(block=False)

    
    return G


def record_model(model, game_class, game_args):
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
            
            print(f'Score: {game.score}')
