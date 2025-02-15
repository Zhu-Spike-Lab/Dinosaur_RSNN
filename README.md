# Dinosaur_RSNN
Files for the Google Dinosaur Game-inspired temporal task

Python: 3.12.5
Required libraries: numpy, pygame

Note: This line was added via the Pomona HPC CLI
## Task Explanation
The Google Dinosaur Game offers an incredibly simple temporal task. The only requisite output is either a 1 or a 0: jump or don't jump. The only metric for success is the timing of the jump. Jump at the right time, and the score increases. Jump at the wrong time and the game ends. In other words, the only calculation being done by the model is a calculation of timing, and score is a direct measurement of a model's temporal calculation competency. As an added bonus, this task is incredibly simple (and satisfying) to visualize and understand.

### Decision Points
One of the most important decisions to be made is what exactly the input will look like. 
Another important choice is how, if at all, the game will speed up over time. 

## Current Implementation
File: Code/attempt_dino_net.py
### Model

### Training
Simple Evolutionary Algorithm using pure score as the fitness function

### Input
1 when the obstacle reaches a certain point on the screen (selected so that a jump right at this time will fail), otherwise 0

### Output
Spike status of neuron [0] (1 when neuron at index [0] spikes, otherwise 0)

### Visualization
In game: Neurons are filled in circles. White when spiking, otherwise black. Currently, no connections are visualized.
In Gephi: Neurons are red circles. Connections between neurons are shown. The thickness of the connections represents their relative weights. Currently, no adjustment is made to account for negative weights (and I don't know what Gephi does by default)

## Next Steps
+ Visualize the connections and weights (also add saving of neural nets)
+ Start by saving most fit of all time, then maybe go to most fit of each generation. Save spiking activity as well?
+ Penalize extra jumps
+ Change speed of obstacles and see if model can learn
+ CHECK: Make sure weights can decrease
+ See how number of neurons affects the training process (scaling laws)
+ Specifying that neurons that receive input cannot give output
+ Try combining output of multiple neurons to provide final output
+ Specifying neuron types
+ Integrate with github
+ Clean up files
+ Vary frequency of obstacles (keep speed constant though) - to make sure model is truly responding to the input
+ Use other training models
