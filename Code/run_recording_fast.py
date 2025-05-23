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
    sys.stdout.write(f'\033[{2}A')
    sys.stdout.write('\033[K')
    sys.stdout.write(word + '\r')
    sys.stdout.write(f'\033[{2}B')
    sys.stdout.flush()

def line_2(word):
    # sys.stdout.write(f'\033[{1}B')
    sys.stdout.write(f'\033[{1}A')
    sys.stdout.write('\033[K')
    sys.stdout.write(word + '\r')
    # sys.stdout.write(f'\033[{1}A')
    sys.stdout.write(f'\033[{1}B')
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
        width=1.0,      # Reduced edge width
        # alpha=0.8       # Slightly transparent for better visual
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

                print(f'Running game. Score: {game.score}', end='\r')
                # if game.score >= 5:
                #     print('debug mode active')
                #     break

            # TODO: FOR BIG MODEL EVALUATION ONLY. BE SURE TO TURN OFF
            print('Running for 600 more frames: 10 more seconds. Giving 0\'s as inputs')

            for i in range(600):
                inputs = torch.tensor([[0,]], dtype=torch.float)
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

                time.sleep(0.001)
            
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
def multi_render_graphs(model, all_spikes, pip_size, max_concurrent, verbose=False):
    G = generate_graph(model)
    total = len(all_spikes)

    graph_frames = [[] for _ in range(total)]
    rendered_frames = np.array([1 for _ in range(total)])
    processes = []
    q = mp.Queue()
    status = mp.Value('i', 0)

    if verbose:
        print('Rendering graphs...\n\n')
    for i, spikes in enumerate(all_spikes):
        if verbose:
            line_1(f'Spinning up: {(i+1) / total * 100:.2f}% started')
        
        # Start new process
        p = mp.Process(target=queue_render_graph, args=(q, i, status, total, G, spikes, pip_size, verbose))
        processes.append(p)
        p.start()

        # Check queue and manage processes
        while len(processes) >= max_concurrent:
            try:
                id, graph_array = q.get(timeout=0.1)  # Short timeout to prevent blocking
                graph_frames[id] = graph_array
                rendered_frames[id] = 0
                # Find and join the corresponding process
                for proc in processes:
                    if not proc.is_alive():
                        proc.join(timeout=0.1)
                        processes.remove(proc)
                        break
            except:
                continue

    # Get remaining items from queue
    now = time.thread_time()
    while len(processes) > 0:
        try:
            id, graph_array = q.get(timeout=1)
            graph_frames[id] = graph_array
            rendered_frames[id] = 0
            # Clean up completed processes
            for proc in processes[:]:
                if not proc.is_alive():
                    proc.join(timeout=1)
                    processes.remove(proc)
        except:
            # Check for any hanging processes
            
            for proc in processes:
                if not proc.is_alive():
                    proc.join(timeout=0.1)
                    processes.remove(proc)
                elif proc.is_alive() and not q.empty():
                    continue
                elif time.thread_time() - now >= 10: # After 10 seconds, start terminating processes
                    proc.terminate()  # Force terminate if process is hanging
                    processes.remove(proc)
    
    if verbose:
        print('Rendering complete')
        print("Checking for failed frames...")
    
    # Check for unrendered frames & fix them
    if np.sum(rendered_frames) != 0:
        if verbose:
            print("Failed frames found")
        # List of unrendered frame ids
        to_render = np.nonzero(rendered_frames) 
        # Turn all_spikes into ndarray for easier indexing
        all_spikes = np.array(all_spikes).squeeze() 
        # Spikes of unrendered frames
        spikes_to_render = all_spikes[to_render] 
        # Recursively render the frames
        if verbose:
            print('Regenerating frames...')
        new_frames = multi_render_graphs(model, spikes_to_render, pip_size, max_concurrent, verbose=verbose)

        # Put the new frames in their proper place
        if verbose:
            print('Fixing frames...')
        for id in to_render:
            graph_frames[id] = new_frames[id]
        
        if verbose:
            print('Frames fixed!')
    else:
        if verbose:
            print("No failed frames found")
    
    return graph_frames

# Define a queue function
def queue_render_graph(q, id, status, total, G, spikes, pip_size, verbose):
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
    if verbose:
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
def multi_combine(game_frames, graph_frames, max_concurrent, verbose=False):
    assert len(game_frames) == len(graph_frames), 'Game video and Graph video have different numbers of frames'

    # print('counting frames')
    num_frames = len(game_frames)
    # print('making frames')
    frames = [[] for _ in range(num_frames)]
    # print('counting frames')
    rendered_frames = np.array([1 for _ in range(num_frames)])
    processes = []
    # print('queueuing')
    q = mp.Queue()
    # print('making values')
    status = mp.Value('i', 0)
    # print('continuing')

    if verbose:
        print('Combining frames...\n\n')
    # Start the processes
    for i in range(num_frames):
        if verbose:
            line_1(f'Spinning up: {i / num_frames * 100:.2f}% started')

        # Start new process
        p = mp.Process(target=queue_combine_frame, args=(q, i, status, num_frames, game_frames[i], graph_frames[i], verbose))
        processes.append(p)
        p.start()

        # Check queue and manage processes
        while len(processes) >= max_concurrent:
            try:
                id, frame = q.get(timeout=0.1)  # Short timeout to prevent blocking
                frames[id] = frame
                rendered_frames[id] = 0
                # Find and join the corresponding process
                for proc in processes:
                    if not proc.is_alive():
                        proc.join(timeout=0.1)
                        processes.remove(proc)
                        break
            except:
                continue

    # Get remaining items from queue
    now = time.thread_time()
    while len(processes) > 0:
        try:
            id, frame = q.get(timeout=1)
            frames[id] = frame
            rendered_frames[id] = 0
            # Clean up completed processes
            for proc in processes[:]:
                if not proc.is_alive():
                    proc.join(timeout=1)
                    processes.remove(proc)
        except:
            # Check for any hanging processes
            for proc in processes:
                if not proc.is_alive():
                    proc.join(timeout=0.1)
                    processes.remove(proc)
                elif proc.is_alive() and not q.empty():
                    continue
                elif time.thread_time() >= 10:
                    proc.terminate()  # Force terminate if process is hanging
                    processes.remove(proc)
        
    if verbose:
        print('Combining complete.')
        print("Checking for failed frames...")
    
    # Check for unrendered frames & fix them
    if np.sum(rendered_frames) != 0:
        if verbose:
            print("Failed frames found.")
        # List of unrendered frame ids
        to_render = np.nonzero(rendered_frames) 
        # Turn graph & game frames into ndarrays for easier indexing
        game_frames = np.array(*game_frames).squeeze() 
        graph_frames = np.array(*graph_frames).squeeze() 
        # Spikes of unrendered frames
        game_frames_to_render = game_frames[to_render] 
        graph_frames_to_render = graph_frames[to_render]
        # Recursively render the frames
        if verbose:
            print('Regenerating frames...')
        new_frames = multi_combine(game_frames_to_render, graph_frames_to_render, max_concurrent, verbose=verbose)

        # Put the new frames in their proper place
        if verbose:
            print('Fixing frames...')
        for id in to_render:
            frames[id] = new_frames[id]
        
        if verbose:
            print('Frames fixed!')
    elif verbose:
        print('No failed frames found.')
    
    return frames

# Define a queue function
def queue_combine_frame(q, id, status, total, game_array, pip_array, verbose):
    # Combine graph & game frame
    # --- Composite picture-in-picture ---
    combined_frame = game_array.copy()  # Ensure writable
    combined_frame[0:pip_array.shape[0], 0:pip_array.shape[1], :] = pip_array

    q.put((id, combined_frame))
    if verbose:
        with status.get_lock():
            status.value += 1
            line_2(f'Combining frames: {status.value / total * 100:.2f}% completed')
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
def render_run(model, game_class, game_args, filename='vid.mp4', pip_size=(400,200), fps=60, multi=True, max_concurrent=100, verbose=True):
    all_spikes, game_frames = run_game(model, game_class, game_args, verbose)

    if multi:
        graph_frames = multi_render_graphs(model, all_spikes, pip_size, max_concurrent, verbose)
        frames = multi_combine(game_frames, graph_frames, max_concurrent, verbose)
    else:
        graph_frames = render_graphs(model, all_spikes, pip_size, verbose)
        frames = combine(game_frames, graph_frames, verbose)

    write_frames(frames, filename, fps, verbose)

    if verbose:
        print(f'Video written to {filename}.')

    return filename