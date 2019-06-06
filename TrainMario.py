import tensorflow as tf
import gym_super_mario_bros
import os
import numpy as np
from random import randint
from nn_model.DQNAgent import DQN_Model
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

ENV_ID = 'SuperMarioBros-v0'
EPISODES = 50
RANGE = 5000
MODEL_FILE_PATH = './saved_model/nn_model.hdf5'
MODEL_DIR = './saved_model/'

def train_dqn(env, state_size, action_size):
    
    if os.path.exists(MODEL_FILE_PATH):
        model = tf.keras.models.load_model(MODEL_FILE_PATH)
        print('loaded model: {}'.format(MODEL_FILE_PATH))
    else:
        if not os.path.exists(MODEL_DIR):
            print('first time setup.')
            os.mkdir(MODEL_DIR)
        model = DQN_Model(action_size, state_size).model
    print(model.summary())

    
    done = True
    last_state = None
    identity = np.identity(env.action_space.n)

    for e in range(EPISODES):
        sum_train_result = 0
        prev_reward = 0
        prev_score = 0
        for step in range(RANGE):
            if done:
                state = env.reset()

            action_source = ""
            if randint(0, 10) == 1 or not isinstance(last_state, (np.ndarray, np.generic)):
                action = env.action_space.sample()
                action_source = "random"
            else:
                action = np.argmax(model.predict(np.expand_dims(last_state, axis = 0)))
                action_source = "learn"
            
            state, reward, done, info = env.step(action)
            last_state = state
            
            score_obtained = prev_score - info['score']
            prev_score = info['score']
            reward += score_obtained * 0.01
            if info['life'] == 0:
                reward = 0
            
            if reward > prev_reward :
                train_result = model.train_on_batch(x = np.expand_dims(last_state, axis = 0),
                                                    y = identity[action: action + 1]) #returns [scalar_mean_loss, prediction]
                print('Reward: {} Life: {} Train Result : {}'.format(reward, info['life'], train_result))
                
                if train_result[0] > 0:
                    max_train_result = train_result
                prev_reward = reward
            env.render()
        print('EP: {}/{}'.format(e, EPISODES))
        model.save(MODEL_FILE_PATH)
    
        
if __name__ == '__main__':
    env = gym_super_mario_bros.make(ENV_ID)
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

    state_size = env.observation_space.shape
    action_size = env.action_space.n

    print('state size: {}'.format(state_size))
    print('action size: {}'.format(action_size))

    train_dqn(env, state_size, action_size)
