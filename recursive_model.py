# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
df = pd.read_csv('data/MatchWinner.csv')


# %%
# drop NaN and incomplete ADH
df = df.dropna(axis='index', ignore_index=True)
ADH_is_complete = df.groupby(by='game_i')['game_j'].transform(lambda b: b.count() == 3)
df = df[ADH_is_complete].reset_index(drop=True)


# %%
# BM = bookmakers
BM_names = ['Bwin', 'NordicBet', '10Bet', 'WilliamHill', 'Bet365', 'Marathonbet', 'Unibet', 'Betfair', 'Betsson', '188Bet', 'Pinnacle', 'SBO', '1xBet', 'Sportingbet', 'Betway', 'Tipico', 'Betcris', '888Sport', 'Dafabet', 'TW']

# %%
b = df[BM_names].to_numpy().reshape(-1 ,3, 20)
bA = b[:, 0, -1]
bD = b[:, 1, -1]
bH = b[:, 2, -1]

plt.plot(1/bA, 1/bD, 'r.')
plt.plot(1/bH, 1/bD, 'g.')
plt.plot(1/bH, 1/bA, 'b.')
plt.show()

plt.plot(1/bA, 1/bD+1/bH, 'r.')
plt.plot(1/bD, 1/bH+1/bA, 'g.')
plt.plot(1/bH, 1/bA+1/bD, 'b.')
plt.plot([0, 1], [1, 0], 'k')
plt.ylim(0, 1.25)
plt.xlim(0, 1.25)
plt.show()



# %%



# BM odd
b = df[BM_names].to_numpy().reshape(-1 ,3, 20)
r0 = 1 - 1 / (1 / b).sum(axis=1, keepdims=True).repeat(repeats=3, axis=1)
n_recur = 10
r = r0
for i in range(n_recur):
    p = ((1 - r) / b).mean(axis=2, keepdims=1)
    p /= p.sum(axis=1, keepdims=True)
    r = 1 - p * b
# %%
# TW odd
b_TW = b[:, :, 19:20]

EV = p * b_TW - 1
play = EV > 0
a = df['pass'].to_numpy().reshape(-1, 3, 1)
rwd = (a * b_TW - 1)[play]
rwd_sum = rwd.sum()
rwd_mean = rwd.mean()
rwd_std = rwd.std()

n_game = b_TW.shape[0]
n_bet = b_TW.shape[0] * b_TW.shape[1]
n_play = play.sum()

print(f'Num of game: {n_game}')
print(f'Num of bet: {n_bet}')
print(f'Num of play: {n_play}')
print(f'Total reward: {round(rwd_sum, 2)}')
print(f'Played ratio: {round(n_play/n_bet*100, 2)}%')
print(f'ROI: {round(rwd_mean*100, 2)}%')
print(f'sROI: {round(rwd_std*100, 2)}%')

# %%
