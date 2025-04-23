# Dinosaur_RSNN
Files for the Google Dinosaur Game-inspired temporal task

Python: 3.12.5
Required libraries: numpy, pygame

## Task Explanation
The Google Dinosaur Game offers an incredibly simple temporal task. The only requisite output is either a 1 or a 0: jump or don't jump. The only metric for success is the timing of the jump. Jump at the right time, and the score increases. Jump at the wrong time and the game ends. In other words, the only calculation being done by the model is a calculation of timing, and score is a direct measurement of a model's temporal calculation competency. As an added bonus, this task is incredibly simple (and satisfying) to visualize and understand. It allows us to visualize and analyze a neural network's performance on a simple delay task.

### Decision Points
One of the most important decisions to be made is what exactly the input will look like. 
Another important choice is how, if at all, the game will speed up over time. 

## Current Implementation
File: Code/torch_multiprocessed.py
### Model
It has 3 different inhibitory neuron types and 1 excitatory neuron type. It is built on PyTorch.

### Training
Evolutionary Algorithm 

### Input
1 when the obstacle reaches a certain point on the screen (selected so that a jump right at this time will fail), otherwise 0

### Output
1 when the output exceeds an arbitrary threshold value (currently set to 1). It is set within the evaluate_model function of the Evolution class in the implementation file (or one the local library where the training functions are stores). Yes, it could be moved.

### Visualization
visualize_model.py provides an interface that allows a visualization of any save model file. The loaded model can also be probed independently. visualize_model.py shows the agent playing the game as well as a networkx graph of the network. The game can be played frame by frame if desired, and individual neuron spikes can be visualized. When the game flashes red, input is provided to the network. A video of the game and model can be saved for later playback.

## Next Steps
+ Contained in Code/goals_list.md