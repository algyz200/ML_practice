import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


#Thomson sampling algorithm
import random
N = 10000
d = 10
ads_selected = []
numbers_of_rewards = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N): #rows (e.g. dif users)
    max_random = 0
    ad = 0
    for i in range(0, d): #ads
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
    
#visualize
plt.hist(ads_selected, rwidth = 0.9)
plt.title('Thomson sampling results')
plt.xlabel('Ads')
plt.ylabel('Each add selected #')
plt.show()