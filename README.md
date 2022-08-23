## Football Outcome Predictions

### Overview and Business Problem

In this project a large dataset containing information on Football matches from various different leagues since 1990, is cleaned, analysed and processed to create a ML model to predict the outcomes of matches. 

The business application of such a project is widespread, since betting companies can use ML Models to dynamically change the odds of Football matches in order to maximise profits. 

### Data Cleaning
The data set has 31 csv files for each league, and we have been given data on 14 leagues, therefore there are 434 csv files, that need to be concatenated.  

The files are iteratively added converted into one large pandas data frame using the following for loop. 

```

files = glob.glob("./Football-Dataset/*/*.csv")
pd_list = []

for file in files:
    temp_csv = pd.read_csv(file)
    pd_list.append(temp_csv)

df = pd.concat(pd_list)

```

A dictionary containing ELO's for each match, needs to converted into a Dataframe and joined to the data frame containing match data. Once this is done various steps are taken to clean the data, such as: 
- Removal of null and duplicated values
- Removal of the leagues 'eerste_divisie' and 'segundu_liga' due to lack of values
- Convesion of columns with object or string values to float or integer. 

## Exploratory Data Analysis 

Combining the rows for Home and Away goals, we can analyse how the average amount of goals scored has changed over the past 31 years. 

The figure below show how the average number of goals scored varies by season and league.Accross the 31 seasons goals scored per game remains fairly constant, at around 2.5 goals per game per season. 

<p align='center'>
  <img 
    width='400'
    src='images/average_goals_by_league.png'
  >
</p>

There is more variation in the average amount of goals scored when analysing by League, with 'eredivisie' and 'Premier League' having a higher amount of goals scored. 

<p align='center'>
  <img 
    width='400'
    src='images/average_goals_year.png'
  >
</p>

Exploring how the average number of fouls have varied by season, we can observe that the average number of fouls were much lower from 1990-1995, and have risen over the years. This is perhaps due to stringent refeering as well innovations such as VAR. 

<p align='center'>
  <img 
    width='400'
    src='images/average_fouls_season.png'
  >
</p>

Additionally. the average number of fouls varies significantly by League, as shown in the figure below. Segunda Division has the highest a an average of 5 fouls per game where as the Premier leagues average is less than three. 

<p align='center'>
  <img 
    width='400'
    src='images/average_fouls_by_league.png'
  >
</p>

An important aspect to investigate is if playing at home corraleted with winning. A new column is created for the Result of a match, using the following code. 

```
for idx, match in new_df.iterrows():
    if match['Home_Score'] > match['Away_Score']:
        res = 1
    elif match['Home_Score'] < match['Away_Score']:
        res = -1
    else:
        res = 0
    new_df.loc[ idx, 'Result_Num'] = res

```

The Result_Num is 1 when the Home team wins, 0 when the match is a draw and -1 when the away team wins. Therefore, if the Result_Num is high for a given season this means the Home team won more often than the away team. 

As seen in the figure below, From 1900 till 2016 the home team would will a lot more games than the away team, however this Home Team Advantage has decreases sharply after 2016. 

<p align='center'>
  <img 
    width='400'
    src='images/home_team_adv.png'
  >
</p>