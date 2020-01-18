import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

import random
N = 10000
d = 10
ads_selected = []
total_reward = 0

for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
    
#visualize
plt.hist(ads_selected, rwidth = 0.9)
plt.title('Random Selection')
plt.xlabel('Ads')
plt.ylabel('Each add selected #')
plt.show()