import torch
import pygame
import networkx as nx
import matplotlib.pyplot as plt
import time

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


def display_model(model, game_class, game_args, interactive=False):
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
            # Update display
            pygame.display.flip()


            if interactive:
                _ = input('next frame: ')
            else:
                # 60 fps (time.time is in nanoseconds)
                rest = (20000000 - (time.time() - start))/1000000000
                if rest > 0:
                    time.sleep(rest)
        
        print(f'Score: {game.score}')

