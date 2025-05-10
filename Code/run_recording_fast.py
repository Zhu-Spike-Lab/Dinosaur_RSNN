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
import time
import multiprocessing as mp
import sys

# Printing functions
def line_1(word):
    sys.stdout.write('\033[K')
    sys.stdout.write(word + '\r')
    sys.stdout.flush()

def line_2(word):
    sys.stdout.write(f'\033[{1}B')
    sys.stdout.write('\033[K')
    sys.stdout.write(word + '\r')
    sys.stdout.write(f'\033[{1}A')
    sys.stdout.flush()

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

    # Pre-allocate node colors array for better performance
    node_colors = ['black'] * len(G.nodes)
    node_colors[0] = 'green'  # Set input node color

    if spikes is not None:
        spikes = spikes.flatten()
        # Update node colors for spiking neurons
        for i, spike in enumerate(spikes):
            if spike > 0:
                node_colors[i] = 'orange'
    
    edge_colors = ['blue' if w > 0 else 'red' for _, _, w in G.edges.data('weight')]

    # Create figure with specific size to avoid resizing
    plt.figure(figsize=(8, 8), dpi=100)
    
    # Draw graph with optimized parameters
    nx.draw_circular(
        G,
        node_color=node_colors,
        edge_color=edge_colors,
        with_labels=False,
        connectionstyle="arc3,rad=0.1",
        node_size=100,  # Reduced node size for better performance
        width=1.0,      # Reduced edge width
        alpha=0.8       # Slightly transparent for better visual
    )
    
    # Update
    plt.draw()
    
    return G


# Run game: capture frames & spikes
def run_game(model, game_class, game_args, verbose=False):
    # Initialize Pygame
    pygame.init()
    # Screen dimensions
    global WIDTH, HEIGHT, font
    WIDTH, HEIGHT = 800, 400
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HIDDEN)

    game = game_class(*game_args)

    model.eval()

    all_spikes = []
    game_frames = []

    if verbose:
        with torch.no_grad():
        # Run the game
            while game.alive:
                inputs = torch.tensor([[game.get_input(),]], dtype=torch.float)
                outputs, spikes = model(inputs)

                all_spikes.append(spikes.numpy())

                choice = spikes[0,0]
                game.step(choice)

                # Visualize
                screen.fill((255, 255, 255))
                game.visualize(screen)

                # Convert visualization into numpy array
                pg_array = pygame.surfarray.array3d(screen).swapaxes(0, 1)  # Shape: (H, W, 3)
                game_frames.append(pg_array)

                # time.sleep(0.001)

                print(f'Running game. Score: {game.score}', end='\r')
                # if len(all_spikes) >= 30:
                #     print('debug mode active')
                #     break
            
            print()
                
    else:
        with torch.no_grad():
            # Run the game
            while game.alive:
                inputs = torch.tensor([[game.get_input(),]], dtype=torch.float)
                outputs, spikes = model(inputs)

                all_spikes.append(spikes.numpy())

                choice = spikes[0,0]
                game.step(choice)

                # Visualize
                screen.fill((255, 255, 255))
                game.visualize(screen)

                # Convert visualization into numpy array
                pg_array = pygame.surfarray.array3d(screen).swapaxes(0, 1)  # Shape: (H, W, 3)
                game_frames.append(pg_array)

                # time.sleep(0.001)
            
            print(f'Score: {game.score}')

    
    
    return all_spikes, game_frames

# Render all the graphs, each in their own thread
def render_graphs(model, all_spikes, pip_size, verbose=False):
    G = generate_graph(model)
    graph_frames = []  # Initialize the list before the loop

    if verbose:
        for i, spikes in enumerate(all_spikes):
            disp_graph(G, spikes=spikes)

            # Save frame to video
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            buf.seek(0)
            graph_img = Image.open(buf).convert("RGB")
            graph_img = graph_img.resize(pip_size)
            graph_array = np.array(graph_img)
            
            graph_frames.append(graph_array)

            print(f'Rendering graphs: {(i+1) / len(all_spikes) * 100:.2f}% completed', end='\r')
            
        print()
    
    else:
        for spikes in all_spikes:
            disp_graph(G, spikes=spikes)

            # Save frame to video
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            buf.seek(0)
            graph_img = Image.open(buf).convert("RGB")
            graph_img = graph_img.resize(pip_size)
            graph_array = np.array(graph_img)
            
            graph_frames.append(graph_array)
    
    return graph_frames

# Multiprocessed version
def multi_render_graphs(model, all_spikes, pip_size, verbose=False):
    G = generate_graph(model)

    graph_frames = [[] for _ in range(len(all_spikes))]
    processes = []
    q = mp.Queue()
    status = mp.Value('i', 0)
    total = len(all_spikes)
    max_concurrent = 100  # Maximum number of concurrent processes

    print('Rendering graphs...')
    for i, spikes in enumerate(all_spikes):
        line_1(f'Spinning up: {i / total * 100:.2f}% completed')
        
        # Start new process
        p = mp.Process(target=queue_render_graph, args=(q, i, status, total, G, spikes, pip_size))
        processes.append(p)
        p.start()

        # Check queue and manage processes
        while len(processes) >= max_concurrent:
            try:
                id, graph_array = q.get(timeout=0.1)  # Short timeout to prevent blocking
                graph_frames[id] = graph_array
                # Find and join the corresponding process
                for proc in processes:
                    if not proc.is_alive():
                        proc.join(timeout=0.1)
                        processes.remove(proc)
                        break
            except:
                continue

    # Get remaining items from queue
    while len(processes) > 0:
        try:
            id, graph_array = q.get(timeout=0.1)
            graph_frames[id] = graph_array
            # Clean up completed processes
            for proc in processes[:]:
                if not proc.is_alive():
                    proc.join(timeout=0.1)
                    processes.remove(proc)
        except:
            # Check for any hanging processes
            for proc in processes[:]:
                if not proc.is_alive():
                    proc.join(timeout=0.1)
                    processes.remove(proc)
                elif proc.is_alive() and not q.empty():
                    continue
                else:
                    proc.terminate()  # Force terminate if process is hanging
                    processes.remove(proc)
    
    sys.stdout.write(f'\033[{2}B')
    sys.stdout.flush()
    print()
    
    return graph_frames

# Define a queue function
def queue_render_graph(q, id, status, total, G, spikes, pip_size):
    disp_graph(G, spikes)

    # Save frame to array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    graph_img = Image.open(buf).convert("RGB")
    graph_img = graph_img.resize(pip_size)
    graph_array = np.array(graph_img)

    q.put((id, graph_array))
    with status.get_lock():
        status.value += 1
        line_2(f'Rendering graphs: {status.value / total * 100:.2f}% completed')
    return


# Combine the two to make a final video
def combine(game_frames, graph_frames, verbose=False):
    assert len(game_frames) == len(graph_frames), 'Game video and Graph video have different numbers of frames'

    length = len(game_frames)
    frames = []
    for i in range(length):
        # Combine graph & game frame
        pg_array = game_frames[i]
        pip_array = graph_frames[i]
        
        # --- Composite picture-in-picture ---
        frame = pg_array.copy()  # Ensure writable
        frame[0:pip_array.shape[0], 0:pip_array.shape[1], :] = pip_array

        frames.append(frame)

        if verbose:
            print(f'Combining frames: {(i+1) / length * 100:.2f}% completed', end='\r')
    
    if verbose:
        print()
    
    return frames

# Multiprocessed version
def multi_combine(game_frames, graph_frames, verbose=False):
    assert len(game_frames) == len(graph_frames), 'Game video and Graph video have different numbers of frames'

    num_frames = len(game_frames)
    frames = [[] for _ in range(num_frames)]
    processes = []
    q = mp.Queue()
    status = mp.Value('i', 0)

    # Start the processes
    for i in range(num_frames):
        processes.append(mp.Process(target=queue_combine_frame, args=(q, i, status, num_frames, game_frames[i], graph_frames[i])))
        processes[i].start()

    # Make it go slower to prevent crashes and grab things from queue to prevent too much pileup
        if q.qsize() > 5:
            id, frame = q.get()
            frames[id] = frame
            processes[i].join()
            time.sleep(0.1)

    # Get all the remaining items from the queue
    for i in range(q.qsize()):
        id, frame = q.get()
        frames[id] = frame
        
    for p in processes:
        p.join()

    print()
    
    return frames

# Define a queue function
def queue_combine_frame(q, id, status, total, game_array, pip_array):
    # Combine graph & game frame
    # --- Composite picture-in-picture ---
    combined_frame = game_array.copy()  # Ensure writable
    combined_frame[0:pip_array.shape[0], 0:pip_array.shape[1], :] = pip_array

    q.put((id, combined_frame))
    with status.get_lock():
        status.value += 1
        print(f'Combining frames: {status.value / total * 100:.2f}% completed', end='\r')
    return
# Write frames to a file
def write_frames(frames, filename, fps, verbose=False):
    with imageio.get_writer(filename, fps=fps, codec='libx264', format='ffmpeg') as writer:
        if verbose:
            for i, frame in enumerate(frames):
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
                writer.append_data(frame)
                print(f'Writing frames: {(i+1) / len(frames) * 100:.2f}% completed', end='\r')
            
            print()
        
        else:
            for frame in frames:
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
                writer.append_data(frame)
    
    return frames

# Do the whole thing in a single function
def render_run(model, game_class, game_args, filename='vid.mp4', pip_size=(400,200), fps=60, multi=True, verbose=True):
    all_spikes, game_frames = run_game(model, game_class, game_args, verbose)

    if multi:
        graph_frames = multi_render_graphs(model, all_spikes, pip_size, verbose)
        frames = multi_combine(game_frames, graph_frames, verbose)
    else:
        graph_frames = render_graphs(model, all_spikes, pip_size, verbose)
        frames = combine(game_frames, graph_frames, verbose)

    write_frames(frames, filename, fps, verbose)

    if verbose:
        print(f'Video written to {filename}.')

    return filename