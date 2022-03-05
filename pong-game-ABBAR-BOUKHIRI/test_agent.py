from keras.models import load_model
import numpy as np
import gym
from gym.utils.play import play


model = load_model("test_model.h5")
env = gym.make('Pong-v4')
env.reset()


'''
    Our trained model is used to play the game , however there is an overfitting problem
    that causes the model always predicting the same action (0).
    
    We tried to fix this problem by adding dropout and regularization to the model.but we did not succeed.
    We think that our dataset has a lot of duplicated images which gives a domination of 0 in labels.
    
'''


def mycallback(obs_t, obs_tp1, action, rew, done, info):
    image = obs_t[34:160:2,::2,1]
    x = np.reshape(image,(1,63*80)).astype(np.float32)
    x /= 255
    action_p = model.predict(x)
    print(action_p, action)

play(env, zoom=3, fps=12, callback=mycallback)

'''
while True:
    env.render()
    obs,rew,d,inf=env.step(env.action_space.sample()) # take a random action
        
env.close()
'''
