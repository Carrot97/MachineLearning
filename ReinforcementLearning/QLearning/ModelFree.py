from Maze import Maze
import numpy as np
import pandas as pd

"""
游戏规则：
冒险家（红）从地图左上角出发，每次向上下左右移动一格，直到找到宝藏（黄）或坠入深渊（黑）。
找到宝藏加一分，坠入深渊坠入深渊减一分
"""

class QParam:
    def __init__(self, actions, learningRate=0.01, rewardDecay=0.9, eGreedy=0.1, epoch=100):
        self.actions = actions
        self.lr = learningRate
        self.gamma = rewardDecay
        self.epsilon = eGreedy
        self.epoch = epoch
        self.QTable = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def chooseAction(self, state):
        """
        根据e贪婪策略选择一个行动
        """
        self.checkState(state)                       # 无模型强化学习需要动态构建Q表
        if np.random.uniform() < self.epsilon:       # epsilon贪婪
            action = np.random.choice(self.actions)  # 小于e时随机选取一个动作
        else:
            stateAction = self.QTable.loc[state, :]
            action = np.random.choice(stateAction[stateAction == np.max(stateAction)].index)  # 大于e时随机选取一个Q值最大的动作
        return action

    def learn(self, state, newState, action, reward):
        """
        按照公式更新Q表
        """
        self.checkState(newState)
        QPredict = self.QTable.loc[state, action]
        if newState != 'terminal':
            QReal = reward + self.gamma * self.QTable.loc[newState, :].max()
        else:
            QReal = reward
        self.QTable.loc[state, action] += self.lr * (QReal - QPredict)

    def checkState(self, state):
        """
        若Q表中不存在该状态则将其添加进Q表，Q值初始化为0
        """
        if state not in self.QTable.index:
            self.QTable = self.QTable.append(pd.Series([0]*len(self.actions), index=self.actions, name=state))

def update():
    for e in range(para.epoch):
        state = maze.reset()  # 初始化
        while True:
            maze.fresh()                                  # 刷新环境（相当于显示）
            action = para.chooseAction(str(state))        # 基于策略选择一个行动
            newState, reward, isDone = maze.step(action)  # 获得新棋盘状态，奖励和游戏是否结束
            para.learn(str(state), str(newState), action, reward)  # 以字符串形式传递与tkinter有关，暂时还没了解
            if isDone:
                break
            state = newState
    maze.destory()

if __name__ == "__main__":
    maze = Maze()  # 建立迷宫环境
    para = QParam(actions=list(range(maze.n_actions)))  # 初始化QL参数缺省值参数采用默认值
    maze.after(para.epoch, update)  # 循环100次
    maze.mainloop()