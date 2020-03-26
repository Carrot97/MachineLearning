import numpy as np
import time

"""
寻宝游戏
（-o---T）
“-”代表路，“o”代表探险者，“T”代表宝藏
"""

# np.random.seed(2)  # 每次使用相同随机种子，使每次运行随机数相同

class Parameter(object):
    def __init__(self, stateNum, action, epsilon, alpha, gamma, epoch):
        self.stateNum = stateNum    # 迷宫长度
        self.s = 0                  # 初始状态为0
        self.actionName = action    # 可选取动作
        self.epsilon = epsilon
        self.alpha = alpha          # 学习率
        self.gamma = gamma          # 衰减率
        self.epoch = epoch
        self.QTable = np.zeros((stateNum, len(action)))  # q表

def chooseAction(Param):
    stateActions = Param.QTable[Param.s, :]
    if (np.random.uniform() > Param.epsilon) or ((stateActions == 0).all()):  # 全零状态即初始状态
        actionNum = np.random.choice(np.arange(len(Param.actionName)))
    else:
        actionNum = np.argmax(stateActions)
    return actionNum

def getFeedback(Param, action):
    if action == 1:  # 0向左，1向右
        if Param.s == Param.stateNum - 2:
            newState = 'terminal'
            reward = 1  # 只有成功到达终点才有奖励值
        else:
            newState = Param.s + 1
            reward = 0
    else:
        reward = 0
        if Param.s == 0:
            newState = Param.s
        else:
            newState = Param.s - 1
    return newState, reward

def updateEnv(Param, epoch, stepCount):
    drawEnv = ['-']*(Param.stateNum-1) + ['T']
    if Param.s == 'terminal':  # 当成功到达终点时，输出该epoch所用步数
        interaction = 'Episode %s: total_steps = %s' % (epoch+1, stepCount)
        print('\r{}'.format(interaction), end='')  # \r相当于清屏，{}为占位符（在此位置输出format()中的变量）
        time.sleep(2)
        print('\r                                ', end='')
    else:
        drawEnv[Param.s] = 'o'
        interaction = ''.join(drawEnv)
        print('\r{}'.format(interaction), end='')
        time.sleep(0.3)

def QLeaning():
    Param = Parameter(6, ['left', 'right'], 0.9, 0.1, 0.9, 10)
    for epoch in range(Param.epoch):
        stepCount = 0
        Param.s = 0  # 初始状态为0
        isTerminal = False
        updateEnv(Param, epoch, stepCount)
        while not isTerminal:
            action = chooseAction(Param)  # 根据epsilon贪婪策略选取行动
            newState, reward = getFeedback(Param, action)
            QPredict = Param.QTable[Param.s, action]  # 行动前对行动价值的估计
            if newState != 'terminal':
                QReal = reward + Param.gamma * Param.QTable[newState, :].max()  # 真实的行动价值
            else:
                QReal = reward  # 到达终点后没有下一个状态
                isTerminal = True
            Param.QTable[Param.s, action] += Param.alpha * (QReal - QPredict)  # 更新Q表
            Param.s = newState
            stepCount += 1
            updateEnv(Param, epoch, stepCount)

if __name__ == "__main__":
    QLeaning()