import gymnasium as gym

# 确保已经注册了环境
from test import CommunicationEnv

# 创建环境实例
env = gym.make('CommunicationEnv-v0')

# 重置环境
observation = env.reset()

# 运行一些简单的测试步骤
for _ in range(1000):
    action = env.action_space.sample()  # 随机选择一个动作
    observation, reward, done, info = env.step(action)
    
   # print(f"观察: {observation}")
   # print(f"奖励: {reward}")
    #print(f"完成: {done}")
    #print(f"信息: {info}")
    
    if done:
        observation = env.reset()
        print(f"完成: {done}")

env.close()