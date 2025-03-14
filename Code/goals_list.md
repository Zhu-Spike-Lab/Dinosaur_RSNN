

End goal: get a way to freeze the game, change the model in a custom way, then play the game step by step (like a debug feature)

## TODO: How tf are we gonna feed Ivyer's models our data? Can't do a classic dataloader bc the problem is too open ended... we have to get the model's guess then tell it if it's right...

## Immediately:
Ivyer's code move to gpu :)
Should use one-hot encoding? - yeah try it, but it's spiking so shouldn't really matter
---Need to change the way score is rewarded (give score when there's a good jump)---
---Need to remove references to pygame…---
---Can make it faster by not using pygame rects (I imagine)---
Add a tiny bit more interactive-ness (graphs)
Need to set it up with a plug and play benchmark for multiple types of networks...
Read about LIF neurons for Yuqing
---Integrate Ivyer's code as a library, work through initial issues---
---Potentially change the method of input---

Temporal Ideas

Inputs:
1 while the obstacle is inside a certain range:
	So temporal length of the signal respresents the time left before jumping


1 when the obstacle is at a certain spot:
	Network just has to learn to slowly speed up its timing as training progresses. Potentially impossible without some other speed up signal


1 when hits a spot with a signal to represent if the game has sped up or not
	Similar to the long or short context task that Buonomano used


Position of obstacle
	Speed can be derived from position over time
	Speed changes over time


Distance from obstacle to player
	A little easier than just giving it position of obstacle


Location of obstacle and location of player 
	A little harder than just giving it distance


Osborn - could vary size of the obstacles as well
Distance - can’t just learn place as long as the speeds are varied (will also need to learn “speed” as well)
Current way it’s set up, it learns a delay function
*It has to get input, delay the jump (preserve the activity), then tamp down its activity (prevent epilepsy) (hypothesis???)

Me - there’s a fundamental problem with how space and time are so interrelated….
	What if the obstacle changes speed while moving?
	I supposed what this is is motion over time… at the end of the say
	*What if we only had excitatory neurons? What if we varied the types/quantities of inhibitory neurons? Yooooo this is such an interesting problem actually!
		Also, why isn’t Ivyer’s code learning? (with a big asterisk bc I haven’t tried it very much…)


Precise next steps:
Size of network/scale
Start differentiating neuron types (connectivity probabilities, timescale of activity - need to have distinct neuron time constants, spike threshold)
Look for 3 neuron motifs, I-E/E-I connections
Cannibalize an STDP implementation - Claudia Clopath rule


## Eventually:
Should try letting the synaptic delay vary...
Look into reinforcement learning, STDP, altering the network in real time...? (W a gui? cli? api?)
Game (irl) moves ~twice as fast as network
Visualize the connections and weights (also add saving of neural nets)
Start by saving most fit of all time, then maybe go to most fit of each generation. Save spiking activity as well?
Need to save the input weights as well
Penalize extra jumps
Change speed of obstacles and see if model can learn
See how number of neurons affects the training process (scaling laws)
Specifying that neurons that receive input cannot give output
Try combining output of multiple neurons to provide final output
Vary frequency of obstacles (keep speed constant though) - to make sure model is truly responding to the input
Use other training models

# Meeting 2/28
Computational benefits of certain types of graphs - like go through formats A B C D and say like which is better
Try Jaxly (The Hodgkins-Huxley equation)