# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:22:10 2019

@author: Tomomasa Takahashi
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sum_range_gb = 21
A_range_gb = 2
upcard_range_gb = 12
firstact_range_gb = 2
naction_range_gb = 3

def card_range_correcter(card): # 引いたカードはここ通す
    if card > 10: # 絵札排除
        card = 10
    return card

def draw_card():
    return min(np.random.randint(1, 14),10)

def calc_card_sum_dealer(L):
    total = 0
    for card in L:
        if card== 1:
            card_adj = 11
        else:
            card_adj = card
        total += card_adj
    return total

def calc_card_sum(L):
    total = 0
    for card in L:
        if card== 1:
            card_adj = 11
        else:
            card_adj = card
        total += card_adj
    if total > 21 and 1 in L:
        total -= 10
    return total

def with_A(L):
    if 1 in L:
        return 1
    else:
        return 0

def UCB_policy_update(i, history, reward, n_action, policy):
    for state in history:
        UCB=[0]*naction_range_gb
        for k in range(naction_range_gb):
            UCB[k] = reward[state[0]][state[1]][state[2]][k]
            UCB[k] += np.sqrt(2*np.log10(i+1)/(n_action[state[0]][state[1]][state[2]][k]+1))
        if state[3] == 0: #first action, DD allowed
            policy[state[0]][state[1]][state[2]][state[3]] = np.argmax(UCB)
        else: #second- action, DD not allowed
            policy[state[0]][state[1]][state[2]][state[3]] = np.argmax(UCB[:(naction_range_gb-1)])
    return policy

class Dealer():
    def __init__(self):
        while True: #Avoid natural BJ
            self.cards = []
            self.cards.append(draw_card())
            self.cards.append(draw_card())
            if calc_card_sum(self.cards) != 21:
                break
    
    def play(self):
        while calc_card_sum_dealer(self.cards) < 17:
            self.cards.append(draw_card())
        return calc_card_sum(self.cards)
    

class Player():
    def __init__(self, policy):
        while True: #Avoid natural BJ
            self.cards = []
            self.cards.append(draw_card())
            self.cards.append(draw_card())
            self.bet = 1 #for DD
            if calc_card_sum(self.cards) != 21:
                break
        self.policy = policy
    
    def hit(self):
        self.cards.append(draw_card())
        
    def play(self, dealer, reward):
        have_A = with_A(self.cards)
        sum_cards = calc_card_sum(self.cards)
        self.history=[]
        while sum_cards < 21:
            if len(self.history)==0:
                first = 0
            else:
                first = 1
            action = self.policy[sum_cards][have_A][dealer.cards[0]][first]
            self.history.append([sum_cards, have_A, dealer.cards[0], first, action])
            #print(sum_cards,have_A,dealer.cards[0],first,action)
            if action == 0: #stay
                break
            elif action == 1: #hit
                self.hit()
                sum_cards = calc_card_sum(self.cards)
            else: #Double down
                self.hit()
                self.bet = 2
                break
        sum_cards = calc_card_sum(self.cards)
        return sum_cards

class BJ():
    def __init__(self, policy):
        self.reward = [[[[0]*naction_range_gb for i in range(upcard_range_gb)]for i in range(A_range_gb)] for j in range(sum_range_gb)] #sum:4-20, with_A:0-1, dealer's upcard:1-10 policy:0-2
        self.policy = policy
        self.n_action = [[[[0]*naction_range_gb for i in range(upcard_range_gb)]for i in range(A_range_gb)] for j in range(sum_range_gb)] #same as reward
        
    def play(self,times):
        gamma = 1.0
        print_freq = 1e5
        subtotal_result = 0
        self.avg_result_hist = []
        for i in range(times):                
            #create player and dealer
            self.player = Player(self.policy)
            self.dealer = Dealer()
            
            #play BJ game
            p_hand = self.player.play(self.dealer,self.reward)
            d_hand = self.dealer.play()
            if p_hand > 21: #player bust
                result = -self.player.bet
            else: #player non bust
                if d_hand>21:
                    result = self.player.bet
                elif d_hand>p_hand:
                    result = -self.player.bet
                elif d_hand==p_hand:
                    result = 0
                else:
                    result = self.player.bet
            #print(self.player.history,result)
            subtotal_result += result
            
            #update reward
            for j in range(len(self.player.history)):
                state = self.player.history[-(j+1)]
                self.n_action[state[0]][state[1]][state[2]][state[4]] += 1
                n_visit = self.n_action[state[0]][state[1]][state[2]][state[4]]
                old_reward = self.reward[state[0]][state[1]][state[2]][state[4]]
                self.reward[state[0]][state[1]][state[2]][state[4]] = (old_reward*(n_visit-1) + result * (gamma**j))/n_visit #discount by gamma
                
            #update policy, note that only action taken state should be updated
            self.policy = UCB_policy_update(i, self.player.history, self.reward, self.n_action, self.policy)
            #simple update
            """
            for j in range(len(self.player.history)):
                state = self.player.history[j]
                if state[3] == 0: #first action, DD allowed
                    self.policy[state[0]][state[1]][state[2]][state[3]] = np.argmax(self.reward[state[0]][state[1]][state[2]])
                else: #second- action, DD not allowed
                    self.policy[state[0]][state[1]][state[2]][state[3]] = np.argmax(self.reward[state[0]][state[1]][state[2]][:naction_range_gb])
            """
            if (i+1)%print_freq == 0:
                avg_result = subtotal_result / print_freq
                print(i+1,"times done. avg result:", avg_result)
                subtotal_result = 0
                self.avg_result_hist.append(avg_result)
            
def main():
    #Set initial policy
    policy = [[[[1]*firstact_range_gb for i in range(upcard_range_gb)] for i in range(A_range_gb)] for j in range(sum_range_gb)]
    for i0 in range(18,sum_range_gb):
        for i1 in range(A_range_gb):
            for i2 in range(upcard_range_gb):
                for i3 in range(firstact_range_gb):
                    policy[i0][i1][i2][i3]=0
    times = int(1e6)
    #Learn
    bj = BJ(policy)
    bj.play(times)
    plt.plot(range(len(bj.avg_result_hist)),bj.avg_result_hist)
    
