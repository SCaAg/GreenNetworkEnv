# 通信网络环境强化学习项目

## 项目概述

本项目实现了一个通信网络环境的强化学习模型。它模拟了一个包含宏基站、微基站和用户设备的通信网络系统,并使用近端策略优化(PPO)算法来优化基站的能量分配和开关状态,以最小化电力消耗和切换惩罚。

## 技术细节

### 环境 (CommunicationEnv)

- 使用 Gymnasium 框架实现
- 包含宏基站、微基站和用户设备
- 状态空间:
  - 微基站的能量消耗
  - 微基站的能量需求
  - 微基站的电池电量
  - 微基站的开关状态
- 动作空间:
  - 能量分配 (连续)
  - 基站开关状态 (离散)
- 奖励:
  - 负的电力消耗成本
  - 负的基站切换惩罚

### 强化学习算法 (PPO)

- 使用 PyTorch 实现
- 网络结构:
  - 共享层: 两个全连接层 (128 和 64 个神经元)
  - 策略头: 输出动作概率
  - 价值头: 输出状态价值估计
- 优化器: Adam
- 损失函数:
  - 策略损失: PPO-Clip 目标
  - 价值损失: 均方误差
  - 熵损失: 鼓励探索

### 训练过程

1. 环境重置,获取初始状态
2. 智能体选择动作
3. 环境执行动作,返回下一状态、奖励和是否结束
4. 存储转换 (状态、动作、奖励等)
5. 计算优势估计和回报
6. 多个 epoch 更新策略和价值网络
7. 重复步骤 2-6 直到达到最大回合数

### 关键参数

- 折扣因子 (GAMMA): 0.99
- GAE lambda: 0.95
- PPO epsilon: 0.2
- 批次大小: 64
- PPO epochs: 10
- 最大梯度范数: 0.5

## 使用说明

1. 确保安装了所有必要的依赖 (PyTorch, Gymnasium 等)
2. 运行 `main.py` 文件开始训练
3. 训练结束后,会显示奖励随回合数变化的图表

## 注意事项

- 环境参数 (如地图大小、时间槽等) 可在 `com_env.py` 中调整
- 训练超参数可在 `main.py` 中修改
- 当前设置为 1000 个训练回合,可根据需要调整
