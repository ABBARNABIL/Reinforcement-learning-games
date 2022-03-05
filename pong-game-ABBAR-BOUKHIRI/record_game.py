import gym
from gym.utils.play import play
import numpy as np


'''
    We play the game and we record the movements of the ball and the paddle as images in X.txt file
    and the actions of the paddle as images in Y.txt file.
'''

env = gym.make('Pong-v4')
env.reset()



def mycallback(obs_t, obs_tp1, action, rew, done, info):
    with open('X.txt', 'a') as outfileX:
        np.savetxt(outfileX,delimiter=',', X=obs_t[34:160:2,::2,1], fmt='%d') # dim : (63,80)
    with open('Y.txt', 'a') as outfileY:
        np.savetxt(outfileY,delimiter='', X=[action], fmt='%d')

play(env, zoom=3, fps=12, callback=mycallback)

env.close() 
