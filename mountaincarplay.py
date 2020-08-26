import gym 

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

import tensorflow as tf
from tensorflow import keras 

from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



env = gym.make('MountainCar-v0')


#load the pretrained weights into the network 
model = models.load_model('weights_ep398.h5')

for ep_ in range(10):
    currentState = env.reset().reshape(1, 2)

    rewardSum=0
    for t in range(200):
        env.render()
        action = np.argmax(model.predict(currentState)[0])

        state_, reward, done, info = env.step(action)

        state_ = state_.reshape(1, 2)

        currentState = state_

        rewardSum += reward
        if done:
            print("Complete! timesteps: {} reward: {}".format(t+1,rewardSum))
            break