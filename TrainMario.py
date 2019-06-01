import gym_super_mario_bros
import cv2
from DQNAgent import DQNAgent
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

EPISODES = 5000

def train_dqn(env_id):
    env = gym_super_mario_bros.make(env_id)
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32
    
    for e in range(EPISODES):
        state = env.reset()
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        env.render()
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _  = env.step(action)
            next_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
            reward = reward if not done else -10

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}. e: {:.2}".format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if e % 10 == 0:
                agent.save("./save/mario-ddqn.h5")
        
if __name__ == '__main__':
    train_dqn('SuperMarioBros-v0')
