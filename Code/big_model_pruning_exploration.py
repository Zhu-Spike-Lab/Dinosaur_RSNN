import torch_multiprocessed_old as tm
import torch
from torch.nn.utils import prune
from copy import deepcopy
from contextlib import redirect_stdout
import multiprocessing as mp



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

def test(q, id, model):
    with redirect_stdout(open('/dev/null', 'w')):
        score = tm.print_model_performance(model, tm.DinosaurGame, (100,))
    
    q.put((id, score))
    return


def multi_test(model, combos):
    works = [False for _ in range(len(combos))]
    processes = []
    q = mp.Queue()
    
    # Create a copy of the model's state dict for each process
    state_dict = model.state_dict()
    
    for i, combo in enumerate(combos):
        # Create a new model instance for each process
        test_model = tm.RSNN2(num_inputs=1, num_hidden=80, num_outputs=1)
        test_model.load_state_dict(state_dict)
        test_model.rlif1.recurrent.weight.data[combo] = 0
        
        p = mp.Process(target=test, args=(q, i, test_model))
        processes.append(p)
        p.start()

    for i in range(len(combos)):
        id, score = q.get()
        works[id] = score >= 50

    for p in processes:
        p.join()

    return works


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

    print('Post prune sparsity: ', end='')
    print(calc_sparsity(model))
    
    print(multi_test(model, [(7,61)]))

    