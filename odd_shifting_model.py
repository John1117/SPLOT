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
n_his = 0
his_idx = n_his * 3
BM_names = ['Bwin', 'NordicBet', '10Bet', 'WilliamHill', 'Bet365', 'Marathonbet', 'Unibet', 'Betfair', 'Betsson', '188Bet', 'Pinnacle', 'SBO', '1xBet', 'Sportingbet', 'Betway', 'Tipico', 'Betcris', '888Sport', 'Dafabet', 'TW']

# fn that calculates BM rake from BM odd of ADH
r_fn = lambda b: 1 - 1 / (1 / b).sum()

# BM odd
b = df[BM_names].to_numpy()

# BM rake
r = df.groupby(by='game_i')[BM_names].transform(r_fn).to_numpy()

# var of BM rake accoss diff game
v = df[df.index < his_idx].groupby(by='game_i')[BM_names].apply(r_fn).var().to_numpy()

# predicted prob from inverse-variance weighted sum of BM prob = 1 - rake / odd
#p = ((1 - r) / b / v).sum(axis=1) / (1 / v).sum()
p = ((1 - r) / b).mean(axis=1)

# TW odd
b_TW = b[:, -1]

# bet only when EV > 0
EV = p * b_TW - 1
bet = (EV > 0) & (df.index >= his_idx)
a = df['pass'].to_numpy() # pass or not
rwd = (a * b_TW - 1)[bet]
rwd_sum = rwd.sum()
rwd_mean = rwd.mean()
rwd_std = rwd.std()

n_game = (df.index >= his_idx).sum() // 3
n_bet = bet.sum()

#print(f'Num of data: {n_his}')
print(f'Num of bet: {n_game*3}')
print(f'Num of play: {n_bet}')
print(f'Total reward: {round(rwd_sum, 2)}')
print(f'Bet ratio: {round(n_bet/n_game/3*100, 2)}%')
print(f'ROI: {round(rwd_mean*100, 2)}%')
print(f'sROI: {round(rwd_std*100, 2)}%')


df[bet][['game_i', 'date', 'bet_name', 'TW', 'pass']]
# %%
m = df.groupby(by='game_i')[BM_names].apply(r_fn).mean().to_numpy()
i = np.argsort(p)
plt.plot(p[i], 1/p[i], 'k--')
plt.plot(p[i], 1/p[i]*(1-m[-1]), 'b--')
plt.plot(p[i], b_TW[i], 'b.')
plt.show()

# %%
r = df.groupby(by='game_i')[BM_names].apply(r_fn).to_numpy()
# %%
sns.violinplot(r)
plt.ylim(-0.015, 0.25)
plt.show()


# %%
