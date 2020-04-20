from Maze import Maze
import numpy as np
import pandas as pd

"""
游戏规则：
冒险家（红）从地图左上角出发，每次向上下左右移动一格，直到找到宝藏（黄）或坠入深渊（黑）。
找到宝藏加一分，坠入深渊减一分
"""

class QParam:
    def __init__(self, actions, learningRate=0.01, rewardDecay=0.9, eGreedy=0.1, epoch=100, traceDecay=0.9):
        self.actions = actions
        self.lr = learningRate
        self.gamma = rewardDecay
        self.epsilon = eGreedy
        self.epoch = epoch
        self.QTable = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.lambda_ = traceDecay
        self.ETable = self.QTable.copy()

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

    def learn(self, state, newState, action, newAction, reward):
        """
        按照公式更新Q表
        """
        self.checkState(newState)
        QPredict = self.QTable.loc[state, action]
        if newState != 'terminal':
            QReal = reward + self.gamma * self.QTable.loc[newState, newAction]
        else:
            QReal = reward
        error = QReal - QPredict

        # 更新该步E表中的值
        self.ETable.loc[state, :] *= 0
        self.ETable.loc[state, action] = 1

        # 全局更新
        self.QTable += self.lr * error * self.ETable
        self.ETable *= self.gamma * self.lambda_

    def checkState(self, state):
        """
        若Q表中不存在该状态则将其添加进Q表和E表，值初始化为0
        """
        if state not in self.QTable.index:
            self.QTable = self.QTable.append(pd.Series([0]*len(self.actions), index=self.actions, name=state))
            self.ETable = self.ETable.append(pd.Series([0]*len(self.actions), index=self.actions, name=state))

def update():
    for e in range(para.epoch):
        state = maze.reset()  # 初始化
        para.ETable *= 0      # 数组清零方法
        action = para.chooseAction(str(state))
        while True:
            maze.fresh()                                        # 刷新环境（相当于显示）
            newState, reward, isDone = maze.step(action)        # 获得新棋盘状态，奖励和游戏是否结束
            newAction = para.chooseAction(str(newState))        # 基于策略选择一个行动
            para.learn(str(state), str(newState), action, newAction, reward)  # 以字符串形式传递与tkinter有关，暂时还没了解
            if isDone:
                break
            state = newState
            action = newAction
    maze.destory()

if __name__ == "__main__":
    maze = Maze()  # 建立迷宫环境
    para = QParam(actions=list(range(maze.n_actions)))  # 初始化QL参数缺省值参数采用默认值
    maze.after(para.epoch, update)  # 循环100次
    maze.mainloop()

    """
    可加快收敛速度，同时更易过拟合，并且会出现不收敛的情况（由于Sarsa“胆小”的特性）
    """