# Dinosaur_RSNN
Files for the Google Dinosaur Game-inspired temporal task

Python: 3.12.5
Required libraries: numpy, pygame

## Task Explanation
The Google Dinosaur Game offers an incredibly simple temporal task. The only requisite output is either a 1 or a 0: jump or don't jump. The only metric for success is the timing of the jump. Jump at the right time, and the score increases. Jump at the wrong time and the game ends. In other words, the only calculation being done by the model is a calculation of timing, and score is a direct measurement of a model's temporal calculation competency. As an added bonus, this task is incredibly simple (and satisfying) to visualize and understand.

### Decision Points
One of the most important decisions to be made is what exactly the input will look like. 
Another important choice is how, if at all, the game will speed up over time. 

## Current Implementation
File: Code/torch_dino_net.py
### Model
Uses Ivyer's implementation of an Evolutionary Algorithm. It has 3 different inhibitory neuron types and 1 excitatory neuron type. It is built on PyTorch.

### Training
Iver's Evolutionary Algorithm 

### Input
1 when the obstacle reaches a certain point on the screen (selected so that a jump right at this time will fail), otherwise 0

### Output
1 when the output exceeds an arbitrary threshold value (currently set to 0.05). It is set within the evaluate_model function of the Evolution class in Ivyer.py. Yes, it should probably be moved.

### Visualization
Currently, the network is only visualized after training within graphs of neuron statistics that were generated with the evolutionary algorithm code. The model with the best fitness is also visualized playing the game at the end of training. Currently, the graphs are not saved anywhere (should use plt.savefig when we want to save them)

## Next Steps
+ Contained in Code/goals_list.md