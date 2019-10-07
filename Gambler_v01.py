# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:45:20 2019

@author: Tomomasa Takahashi
"""
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
target = 100
prob = 0.2
win = 3
epsilon_change=1e-5
epsilon_reward = 1e-4 # be conservative if reward is almost same
change_max = epsilon_change*100

reward = [[0]*target for i in range(target+win)]
for i in range(target, target+win):
    reward[i] = [1]*target

k=0 # track iteration
while change_max > epsilon_change:
    reward_cp = cp.deepcopy(reward)
    change_max = 0
    for i in range(1,target): # current money
        for j in range(1,min(i,(target-i-1)//win+1)+1): # bet
            temp = reward_cp[i][j]
            reward[i][j] = prob * np.max(reward_cp[i+win*j]) + (1-prob) * np.max(reward_cp[i-j])
            change = abs(temp-reward[i][j])
            if change_max < change:
                change_max=change
    if k>200:
        break
    k=k+1
    print(k, change_max)

optimal = []
for i in range(1,target):# current money
    max_reward=0
    for j in range(1,min(i,(target-i-1)//win+1)+1):# bet
        if max_reward==0 or reward[i][j] > (max_reward+epsilon_reward):
            max_reward = reward[i][j]
            optimal_bet = j
    optimal.append(optimal_bet)

plt.bar(range(1, target), optimal)
