<p align="center">
    <h1> Mountain Car DQN </h>
</p>

<br />

<p align="center">
  <h1 align="center">Scott Caputo | Mountain Car </h1>

  <p align="center">
    + Created using Python and Tensorflow 2
  </p>
  <p align="center">
    + Gym AI environment  
  </p>
  <p align="center">
    + Trained Deep Learning Model 
  </p>
  <p align="center">
    + DQN Implementation
  </p>
  <p align="center">
    <a href="https://github.com/scottvcaputo">scottvcaputo.github.io</a>
  </p>
  <p align="center">
    <a href="https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf">Human-level control through deep reinforcement learning</a>
    <br />
    <br />
    <br />
  </p>
</p>

<p align="center">
    <img src="https://media.giphy.com/media/MBHu5zorQTmcxXmUbw/giphy.gif" alt="Demo" />
</p>

<br />
<br />
<hr />

## Frameworks and Dependencies 


[Python](https://www.python.org/)
| [Bootstrap](https://www.tensorflow.org/)
| [Gym](https://gym.openai.com/)
| [Mountain Car](https://gym.openai.com/envs/MountainCar-v0/)
| [NumPy](https://numpy.org/)

## Structure 

- Run
    - Training and Modeling

- Play
    - Model testing and visualization using pre-trained weights


## Inspiration

- After completing several courses and specializations in machine learning and deep learning, I was interested in applying these concepts to better understand all that goes into the implementation of a model. I found much of the information across the web to primarily be done in older framework versions in Tensorflow 1 and Keras, or Pytorch. Being that I wanted to create a future for myself in the field, I was more interested in the most recent and cutting-edge versions of the framework and chose to dedicate my time to Tensorflow 2 and Pytorch.

- This model was written in Python using the Tensorflow 2 framework, with Tensorflow-Keras as the backend API, which was incorporated to build the DQN neural network. 

- The parameters, hyperparameters, and play file structure were determined using several pre-existing Mountain Car repositories as a reference.  


## Parameters and Hyperparameters

- environment = MountainCar-v0
- gamma = 0.99
- epsilon = 1
- epsilon decay = 0.05
- epsilon minumum = 0.01 
- learning rate = 0.001
- episodes = 350
- iterations = 200


Additional Notes: The episode count could be made higher but was found to be effective after roughly 300 episodes. This could be made higher to more effectively train the model, but seems to plateau in effectiveness after roughly 450 epsiodes. 