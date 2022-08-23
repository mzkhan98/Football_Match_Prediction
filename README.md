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

The two figures below show how the average number of goals scored varies by season and league.Accross the 31 seasons goals scored per game remains fairly constant, at around 2.5 goals per game per season. 
There is more variation in the average amount of goals scored when analysing by League, with 'eredivisie' and 'Premier League' having a higher amount of goals scored. 
![Average Goals By Season](images/average_goals_by_league.png)
![Average Goals By Year](images/average_goals_year.png)

Exploring how the average number of fouls have varied by season and leag