# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# UCB
N = 10000
d = 10
ads_selected = []
number_of_selections = [0] * d
sum_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_ucb = 0
    for i in range(0, d):
        if (number_of_selections[i] > 0):
            avg_reward = sum_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / number_of_selections[i])
            ucb = avg_reward + delta_i
        else:
            ucb = 1e400
        if (ucb > max_ucb): 
            max_ucb = ucb
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_reward = total_reward + reward
        
        
        
        


## random select
#import random
#N = 10000
#d = 10
#ads_selected = []
#total_reward = 0
#for n in range(0, N):
#    ad = random.randrange(d)
#    ads_selected.append(ad)
#    reward = dataset.values[n, ad]
#    total_reward = total_reward + reward





# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()