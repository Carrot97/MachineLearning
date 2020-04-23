import gym
from PolicyGradient import Params
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 150  # 设置刷新阈值，当reward大于150时开始刷新
RENDER = False

env = gym.make('CartPole-v0')  # 立杆子游戏
env = env.unwrapped
env.seed(1)  # 设置随机种子

pg = Params(env.observation_space.shape[0], env.action_space.n)
print('start')
for e in range(pg.epoch):
    state = env.reset()
    while True:
        if RENDER: env.render()
        action = pg.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        pg.store_state(state, action, reward)
        if done:
            # 刚开始训练不出动画
            round_rewards_sum = sum(pg.round_rewards)
            if 'running_reward' not in globals():
                running_reward = round_rewards_sum
            else:
                running_reward = running_reward * 0.99 + round_rewards_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True

            pg.learn()
            break
        state = next_state