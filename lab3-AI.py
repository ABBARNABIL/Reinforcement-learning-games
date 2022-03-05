import gym
import imageio as imageio
import numpy as np
from gym.utils.play import play

env = gym.make('Pong-v4')
env.reset()
env.close()


def mycallback(obs_t, obs_tp1, action, rew, done, info):
 #imageio.imwrite("jeu.jpg", obs_t)
 #imageio.imwrite('outfile.png', obs_t[34:194:4, 12:148:2, 1])
 with open('X.txt', 'a') as outfileX:
  np.savetxt(outfileX, delimiter='', X=obs_t[34:194:4, 12:148:2, 1], fmt='%d')
 with open('Y.txt', 'a') as outfileY:
   np.savetxt(outfileY, delimiter='', X=[action], fmt='%d')

 print("action = ", action, " reward = ", rew, "done = ", done)

gym.utils.play.play(env, zoom=3, fps=12, callback=mycallback)


# import necessary modules from keras
from keras.layers import Dense
from keras.models import Sequential

'''
The 80 * 80 input dimension comes from the pre-processing of the raw pixels made by Karpathy (the only important pixels are the balls and the paddle)
Input here represents the difference in pixels betewen one frame and another, giving you direction of agents and ball. Encoded in Karpathy’s own preprocessing functions
'''
model = Sequential()
# hidden layer takes a pre-processed frame as input, and has 200 units. Simple layer architectur of 200 x1, 1x1
model.add(Dense(units=200,input_dim=80*80, activation='relu', kernel_initializer='glorot_uniform'))
# output layer — we use a Sigmoid here, in order to get a 0, or 1 value to represent ACTION UP
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))
# compile the model using traditional Machine Learning losses and optimizers
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())