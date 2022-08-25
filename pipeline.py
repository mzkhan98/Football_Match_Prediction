import pandas as pd 
import glob 
import seaborn as sns

def clean_link(x):
    new_link = x.split('.com')[1]
    new_link_year = new_link.split('/')[-1][0:4]
    new_link_complete = '/'.join(new_link.split('/')[:-1]) + '/' + new_link_year
    return new_link_complete


def get_home_result(x):
    home = x.split('-')[0]
    return home

def get_away_result(x):
    away = x.split('-')[1]
    return away

def drop_duplicates(df):
    df = df.drop_duplicates(subset='Link')

def format_capacity(df):
    final_df = final_df.drop_duplicates(subset='Link')
    df['Capacity'] = pd.to_numeric(df['Capacity'], errors = 'coerce')
    df['Capacity'].fillna(df['Capacity'].median(), inplace=True)
    df['Capacity'] = df['Capacity'].astype('int64')

def change_date(df):
    df['Date'] = df['Date_New'].apply(lambda x: x[x.find(',') + 2:x.rfind(',')])
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
    #print(set(df['Date']))

    df['Day'] = df['Date_New'].apply(lambda x: (x[:x.find(',')]))
    df['Day'] = df['Date'].dt.day_of_week
    print(set(df['Day']))

    df['Time'] = df['Date_New'].apply(lambda x: x[x.rfind(',') + 2:])
    print(set(df['Time']))
    df.head()

    df = df.drop(['Date_New'], axis=1)

def drop_irrelevant(df):
    df = df.drop(['Link', 'Round', 'Referee', 'City', 'Stadium', 'Pitch', 'Time', 'Home_Yellow', 'Home_Red', 'Away_Yellow', 'Away_Red'], axis=1)

def fix_goals(df):
    df['Home_Goals'] = pd.to_numeric(df['Home_Goals'], errors = 'coerce')
    df['Away_Goals'] = pd.to_numeric(df['Away_Goals'], errors = 'coerce')

def get_streak(df, team):
    team_result_series = df['Outcome'].where(df['Home_Team'] == team, -df['Outcome'])
    team_streak_series = team_result_series.groupby((team_result_series != team_result_series.shift()).cumsum()).cumsum()
    return team_streak_series.where(team_streak_series > 0, 0)

def get_form(df, team):
    period = 5
    team_result_series = df['Outcome'].where(df['Home_Team'] == team, -df['Outcome'])
    return team_result_series.rolling(period).sum()

def get_goals(df, team, home, away):
    period = 10
    home_goal_series = df['Home_Goals'].where(df['Home_Team'] == team, 0).rolling(period).sum() / period
    away_goal_series = df['Away_Goals'].where(df['Away_Team'] == team, 0).rolling(period).sum() / period
    return home * home_goal_series + away * away_goal_series

def form(df,f_mask,h_mask,a_mast, ):
    for team in set(df['Home_Team'].to_list()):
        f_mask = (df['Home_Team'] == team) | (df['Away_Team'] == team)
        h_mask = (df['Home_Team'] == team)
        a_mask = (df['Away_Team'] == team)

        df.loc[f_mask, 'temp'] = get_streak(df[f_mask], team).shift()
        df.loc[h_mask, 'Home_Team_Streak'] = df.loc[h_mask, 'temp']
        df.loc[a_mask, 'Away_Team_Streak'] = df.loc[a_mask, 'temp']

        df.loc[h_mask, 'Home_Team_Home_Streak'] = get_streak(df[h_mask], team).shift()
        df.loc[a_mask, 'Away_Team_Away_Streak'] = get_streak(df[a_mask], team).shift()

        df.loc[f_mask, 'temp'] = get_form(df[f_mask], team).shift()
        df.loc[h_mask, 'Home_Team_Form'] = df.loc[h_mask, 'temp']
        df.loc[a_mask, 'Away_Team_Form'] = df.loc[a_mask, 'temp']

        df.loc[h_mask, 'Home_Team_Home_Form'] = get_form(df[h_mask], team).shift()
        df.loc[a_mask, 'Away_Team_Away_Form'] = get_form(df[a_mask], team).shift()

        df.loc[f_mask, 'temp'] = get_goals(df[f_mask], team, home=1, away=1).shift()
        df.loc[h_mask, 'Home_Team_Goals'] = df.loc[h_mask, 'temp']
        df.loc[a_mask, 'Away_Team_Goals'] = df.loc[a_mask, 'temp']

        df.loc[h_mask, 'Home_Team_Home_Goals'] = get_goals(df[h_mask], team, home=1, away=0).shift()
        df.loc[a_mask, 'Away_Team_Away_Goals'] = get_goals(df[a_mask], team, home=0, away=1).shift()
        df = df.drop(['temp'], axis=1)
        df = df.dropna(axis=0, subset=[
        'Home_Team_Streak',
        'Away_Team_Streak',
        'Home_Team_Home_Streak',
        'Away_Team_Away_Streak',
        'Home_Team_Form',
        'Away_Team_Form',
        'Home_Team_Home_Form',
        'Away_Team_Away_Form',
        'Home_Team_Goals',
        'Away_Team_Goals',
        'Home_Team_Home_Goals',
        'Away_Team_Away_Goals'])
        df.info()

def drop_notused(df):
    df = df.drop(['Home_Team', 'Away_Team', 'Result', 'League', 'Country', 'Home_Goals', 'Away_Goals'], axis=1)

def save_data(df):
    df.to_json('clean_data.json')
    df.to_csv('clean_data.csv')