import torch_multiprocessed_old as tm
import torch
from torch.nn.utils import prune
import multiprocessing as mp
import sys
import numpy as np
import time
from copy import deepcopy

# Printing functions
def line_1(word):
    sys.stdout.write(f'\033[{2}A')
    sys.stdout.write('\033[K')
    sys.stdout.write(word + '\r')
    sys.stdout.write(f'\033[{2}B')
    sys.stdout.flush()

def line_2(word):
    sys.stdout.write(f'\033[{1}A')
    sys.stdout.write('\033[K')
    sys.stdout.write(word + '\r')
    sys.stdout.write(f'\033[{1}B')
    sys.stdout.flush()

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

def test(q, id, status, total, model, verbose):
    score = tm.print_model_performance(model, tm.DinosaurGame, (100,), verbose=False)
    # print(id, score)
    q.put((id, score))
    with status.get_lock():
        status.value += 1
        if verbose:
            line_2(f'Combinatoricking: {status.value / total * 100:.2f}% completed')
    return

def little_test(model, combo):
    test_model = deepcopy(model)
    test_model.rlif1.recurrent.weight.data[*combo] = 0
    score = tm.print_model_performance(model, tm.DinosaurGame, (100,), verbose=False)

    return (score >= 50)

def multi_test(model, combos, verbose=True):
    total = len(combos)
    works = torch.zeros(total)
    tested = torch.ones(total)
    processes = []
    q = mp.Queue()
    status = mp.Value('i', 0)
    max_concurrent = 400

    if verbose:
        print('Combinating...\n\n')
    
    # Create a copy of the model's state dict for each process
    state_dict = model.state_dict()
    
    for i, combo in enumerate(combos):
        if verbose:
            line_1(f'Spinning up: {(i+1) / total * 100:.2f}% started, Works: {torch.sum(works)}, Failed: {torch.sum(tested == 0) - torch.sum(works)}')
        # Create a new model instance for each process
        test_model = deepcopy(model)
        test_model.rlif1.recurrent.weight.data[*combo] = 0
        
        p = mp.Process(target=test, args=(q, i, status, total, test_model, verbose)) # Change i to combo
        processes.append(p)
        p.start()

        # Check queue and manage processes
        while len(processes) >= max_concurrent:
            try:
                id, score = q.get(timeout=0.01)  # Short timeout to prevent blocking
                if score >= 50: works[id] = 1
                tested[id] = 0
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
            id, score = q.get(timeout=1)
            if score >= 50: works[id] = 1;
            tested[id] = 0
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
        print("Checking for incomplete combos...")
    
    # Check for unrendered frames & fix them
    if torch.sum(tested) != 0:
        if verbose:
            print("Incomplete combos found")
        # List of unrendered frame ids
        to_render = np.nonzero(tested) 
        # Turn all_spikes into ndarray for easier indexing
        print(to_render)


    else:
        if verbose:
            print("No failed combos found")
    
    return works

def find_minimal(model):
    connections = torch.nonzero(model)
    total = len(connections)

    # Do it in batches of 400
    for i in range(0, total // 400):
        print(i)
        combos = connections[400*i:400*(i+1)]
        works = multi_test(model, combos)

        # Connections that can be removed exist @ works == 1
        removable_connections = combos(torch.nonzero(works))

        if len(removable_connections) == 0:
            print(f'On Run {i}, we found no removable connections and have now quit')
            return model

        model.rlif1.recurrent.weight.data[removable_connections] = 0
    
    i = total // 400
    combos = connections[400*i:]
    works = multi_test(model, combos)

    # Connections that can be removed exist @ works == 1
    removable_connections = combos(torch.nonzero(works))

    if len(removable_connections) == 0:
        print(f'On Run {i}, we found no removable connections and have now quit')
        return model

    model.rlif1.recurrent.weight.data[removable_connections] = 0

    return model
    






if __name__ == '__main__':

    model = tm.RSNN2(num_inputs=1, num_hidden=80, num_outputs=1)
    filename = "big_model.pth"
    state_dict = torch.load(filename, weights_only=True)
    
    model.load_state_dict(state_dict)

    print('Initial sparsity: ', end='')
    print(calc_sparsity(model))
    big_prune = prune_weights(model, pruning_rate=0.1615) # fails
    little_prune = prune_weights(model, pruning_rate=0.15) # doesn't fail

    model.rlif1.recurrent.weight.data = little_prune
    # model.rlif1.recurrent.weight.data[0,0] = 0
    # tm.print_model_performance(model, tm.DinosaurGame, (100,))

    print('Post prune sparsity: ', end='')
    print(calc_sparsity(model))
    

    connections = torch.nonzero(little_prune)
    # works = multi_test(model, connections)

    model = find_minimal(model)
    print(f'Final sparsity:{calc_sparsity(model)}')



    print(little_test(model, [0,0]))

    