import gym
from keras.models import load_model

model = load_model('test_model.h5')
env = gym.make('Pong-v4')
env.reset()

'''def mycallback(obs_t, obs_tp1, action, rew, done, info):
 #imageio.imwrite("jeu.jpg", obs_t)
 #imageio.imwrite('outfile.png', obs_t[34:194:4, 12:148:2, 1])

 obs = obs_t[34:160:2,::2,1]
 x = np.reshape(obs, (1, 63*80))
 x = x/255.0
 prediction = model.predict(x)
 print(int(prediction[0][0]))
 env.step(int(prediction[0][0]))


gym.utils.play.play(env, zoom=3, fps=12, callback=mycallback)

'''

while True:
    env.render()
    obs, rew, d, inf = env.step(env.action_space.sample())  # take a random action
    if rew != 0:
        print("reward: ", rew)

env.close()