import tensorflow as tf
import gym_super_mario_bros
import os
import numpy as np
from action_wrapper import mario_action_interpret
from random import randint
from DQNAgent import _build_model
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

ENV_ID = 'SuperMarioBros-v0'
EPISODES = 100
MODEL_FILE_PATH = './model/nn_model.HDF5'
MODEL_DIR = './model/'

def train_dqn(env):
    if os.path.exists(MODEL_FILE_PATH):
        model = tf.keras.models.load_model(MODEL_FILE_PATH)
        print('loaded model: {}'.format(MODEL_FILE_PATH))
    else:
        if os.path.exists(MODEL_DIR):
            print('first time setup.')
            os.mkdir(MODEL_DIR)
        model = _build_model(action_size, state_size)
    
    done = True
    last_state = None
    identity = np.identity(env.action_space.n)

    for e in range(EPISODES):
        for step in range(1000):
            if done:
                state = env.reset()

            action_source = ""
            if randint(0, 10) == 1 or not isinstance(last_state, (np.ndarray, np.generic)):
                action = env.action_space.sample()
                action_source = "random"
            else:
                action = np.argmax(model.predict(np.expand_dims(last_state, axis = 0)))
                action_source = "learn"
            #print('{}: {}'.format(action_source, mario_action_interpret(action)))
            state, reward, done, info = env.step(action)
            last_state = state

            if reward > 0:
                model.train_on_batch(x = np.expand_dims(last_state, axis = 0), y = identity[action: action + 1])

            env.render()

        print('Ep: {}/{} '.format(e, EPISODES))
        model.save(MODEL_FILE_PATH)
                
        
if __name__ == '__main__':
    env = gym_super_mario_bros.make(ENV_ID)
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

    state_size = env.observation_space.shape
    action_size = env.action_space.n

    print('state size: {}'.format(state_size))
    print('action size: {}'.format(action_size))

    train_dqn(env)
