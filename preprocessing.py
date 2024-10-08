# %%
import pandas as pd


# %%
away_df = pd.read_csv('data/Match_Winner_Away.csv', parse_dates=['日期'])
draw_df = pd.read_csv('data/Match_Winner_Draw.csv', parse_dates=['日期'])
home_df = pd.read_csv('data/Match_Winner_Home.csv', parse_dates=['日期'])

# %%
df = pd.concat([away_df, draw_df, home_df], axis=0, join='outer', ignore_index=True)
df = df.drop(labels=['Unnamed: 0.1', 'Unnamed: 0', '注名'], axis='columns')
df = df.rename({'日期': 'date', '場數': 'game_j', '子注名': 'bet_name', 'odds_t': 'TW', 'pass or not': 'pass'}, axis='columns')
df['game_j'] = df['game_j'].map(lambda x: int(x[-1]))
df['bet_name'] = df['bet_name'].map(lambda x: x[0])
df = df.sort_values(by=['date', 'game_j'])
df = df.reset_index(drop=True)
game_i = df.groupby(by=['date', 'game_j']).ngroup()
df.insert(loc=0, column='game_i', value=game_i)

# %%
