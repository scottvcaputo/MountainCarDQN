import gym

import os
import imageio
import gc

import IPython
from IPython.display import HTML
from IPython.display import clear_output

import PIL.Image
import pyglet

from collections import deque
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

import tensorflow as tf
from tensorflow import keras 

from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# this can be used for personal devices (if the tensorflow-gpu version is installed)
# if done on cloud, connect to GPU using the respective platforms commands / directions

            #config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} ) 

# additionally may need to use these imports to effectively visualize the results 

            #import pyvirtualdisplay
            #from pyvirtualdisplay import Display


# logging, not currently used 
#ep_log = []
#epsilon_log = []
#rewardsum_log = []
#maxpos_log = []

class MountainCarDQN:
    def __init__(self,env):
        # environment 
        self.env = env

        # gamma 
        self.gamma = 0.99

        # control epsilon progression 
        self.epsilon = 1
        self.epsilon_decay = 0.05
        self.epsilon_min = 0.01

        # learning rate 
        self.lr=0.001

        # memory buffer capacity 
        self.replayBuffer = deque(maxlen=20000)

        # define a training network
        self.trainNetwork=self.DQN()

        # longer episode lengths result in more effective learning (until ~ 300 - 350 episodes)
        self.episodes=350
        self.iterations=200

        # interacts with deque buffered memory 
        self.bufferSelect=32

        # define a target network 
        self.targetNetwork=self.DQN()

        # used to sync weights between target and training network 
        self.targetNetwork.set_weights(self.trainNetwork.get_weights())


    # Optimizer = adam      Loss = mean squared error       learning rate = 0.001   
    # activation = [relu, relu, linear]


    # Tensorflow 2 deep Q network using tf.keras api
    def DQN(self):
        model = keras.Sequential()

        state_shape = self.env.observation_space.shape

        model.add(keras.layers.Dense(24, activation='relu', input_shape=state_shape))
        model.add(keras.layers.Dense(48, activation='relu'))
        model.add(keras.layers.Dense(self.env.action_space.n,activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.lr))

        return model




    def getBestAction(self,state):

        # prevent epsilon from going below minimum due to decay 
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            action=np.argmax(self.trainNetwork.predict(state)[0])

        return action


    #####################################################################################
    def trainFromBuffer_Boost(self):
        if len(self.replayBuffer) < self.bufferSelect:
            return
        samples = random.sample(self.replayBuffer,self.bufferSelect)
        npsamples = np.array(samples)
        states_temp, actions_temp, rewards_temp, newstates_temp, dones_temp = np.hsplit(npsamples, 5)
        states = np.concatenate((np.squeeze(states_temp[:])), axis = 0)
        rewards = rewards_temp.reshape(self.bufferSelect,).astype(float)
        targets = self.trainNetwork.predict(states)
        newstates = np.concatenate(np.concatenate(newstates_temp))
        dones = np.concatenate(dones_temp).astype(bool)
        notdones = ~dones
        notdones = notdones.astype(float)
        dones = dones.astype(float)
        Q_futures = self.targetNetwork.predict(newstates).max(axis = 1)
        targets[(np.arange(self.bufferSelect), actions_temp.reshape(self.bufferSelect,).astype(int))] = rewards * dones + (rewards + Q_futures * self.gamma)*notdones
        self.trainNetwork.fit(states, targets, epochs=1, verbose=0)

    #Or self.train() can => self.trainFromBuffer_Boost() (matrix wise version for boosting) 
    #####################################################################################


    def train(self):
        if len(self.replayBuffer) < self.bufferSelect:
            return

        samples = random.sample(self.replayBuffer,self.bufferSelect)

        # declare states and new states as lists to store values
        states = []
        States_ = []

        for sample in samples:
            state, action, reward, state_, done = sample
            states.append(state)
            States_.append(state_)

        # storing values in arrays
        na = np.array(states)
        states = na.reshape(self.bufferSelect, 2)

        na2 = np.array(States_)
        States_ = na2.reshape(self.bufferSelect, 2)

        targets = self.trainNetwork.predict(states)
        state_target=self.targetNetwork.predict(States_)

        # creating an index value 
        i=0

        for sample in samples:
            state, action, reward, state_, done = sample
            target = targets[i]
            if done:
                target[action] = reward
            else:
                Q_future = max(state_target[i])
                target[action] = reward + Q_future * self.gamma
            i+=1

        self.trainNetwork.fit(states, targets, epochs=1, verbose=0)



    def orginalAction(self,currentState,eps):

        rewardSum = 0
        max_pos = -99

        for i in range(self.iterations):
            bestAction = self.getBestAction(currentState)

            # simple mod to display progression over a given interval 
            if eps % 25 == 0:
                env.render()

            state_, reward, done, _ = env.step(bestAction)

            state_ = state_.reshape(1, 2)

            # tracking max position
            if state_[0][0] > max_pos:
                max_pos = state_[0][0]


            # integrating task completion into reward 
            if state_[0][0] >= 0.5:

                reward += 10

            self.replayBuffer.append([currentState, bestAction, reward, state_, done])

            self.train()

            rewardSum += reward

            currentState = state_

            if done:
                break

        if i >= 199:
            print("Failed: episode {}".format(eps))
        else:
            print("Success: episode {}, iterations: {}".format(eps, i))
            self.trainNetwork.save('./weights_ep{}.h5'.format(eps))

        # Sync the target network with the training network (helps the training network catch target network)
        self.targetNetwork.set_weights(self.trainNetwork.get_weights())

        print("epsilon: {}, reward: {} maxPosition: {}".format(max(self.epsilon_min, self.epsilon), rewardSum, max_pos))


        #epis = 0
        #ep_log.append(epis)
        #epsilon_log.append(max(self.epsilon_min, self.epsilon))
        #rewardsum_log.append(rewardSum)
        #maxpos_log.append(max_pos)

        # controlling the epsilon value 
        self.epsilon -= self.epsilon_decay
        #epis += 1


    def start(self):
        for eps in range(self.episodes):
            currentState=env.reset().reshape(1,2)
            self.orginalAction(currentState, eps)


    # incorrectly logs, is not used here 
    def log(self):

        # plot specs
        plt.figure(figsize = (18, 12))


        # maximum position plot 
        plt.subplot(221)
        plt.plot(maxpos_log, ep_log)
        plt.yscale('linear')
        plt.title('maximum position vs episodes')
        plt.grid(True)

        # reward sum plot
        plt.subplot(222)
        plt.plot(rewardsum_log, ep_log)
        plt.yscale('linear')
        plt.title('reward sum vs episodes')
        plt.grid(True)

        # epsilon plot
        plt.subplot(223)
        plt.plot(epsilon_log, ep_log)
        plt.yscale('linear')
        plt.title('epsilon decay vs episodes')
        plt.grid(True)

        # maximum position vs reward sum
        plt.subplot(224)
        plt.plot(maxpos_log, rewardsum_log)
        plt.yscale('linear')
        plt.title('maximum position vs reward sum')
        



####### Starting the Run #######



env = gym.make('MountainCar-v0')
dqn=MountainCarDQN(env=env)
dqn.start()




################################