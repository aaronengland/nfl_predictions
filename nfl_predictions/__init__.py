import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from game_predictions import game_predictions

# define function for upcoming week predictions
def nfl_pickem(year, weighted_mean=False, n_simulations=1000):
    # get url
    r = requests.get('https://www.pro-football-reference.com/years/{0}/games.htm'.format(year))
    # get content of page
    soup = BeautifulSoup(r.content, 'html.parser')
    # get all table rows
    table_rows = soup.find_all('tr')
    
    # instantiate empty lists
    list_week = []
    list_winning_team = []
    list_game_location = []
    list_losing_team = []
    list_winning_team_points = []
    list_losing_team_points = []
    # for each row
    for i in range(1, len(table_rows)):
        # get a row
        row = soup.find_all('tr')[i]
        # get all td elements
        td = row.find_all('td')
        # if td is not an empty list
        if td:
            # get week
            week = int(row.find('th').text)
            list_week.append(week)
            # get winning team
            winning_team = td[3].find('a').text
            list_winning_team.append(winning_team)
            # get game location
            game_location = td[4].text
            list_game_location.append(game_location)
            # get losing team
            losing_team = td[5].find('a').text
            list_losing_team.append(losing_team)
            # get winning team points
            winning_team_points = td[7].text
            list_winning_team_points.append(winning_team_points)
            # get losing team points
            losing_team_points = td[8].text
            list_losing_team_points.append(losing_team_points)
    
    # put into df
    df = pd.DataFrame({'week': list_week,
                       'winning_team': list_winning_team,
                       'game_loc': list_game_location,
                       'losing_team': list_losing_team,
                       'winning_team_points': list_winning_team_points,
                       'losing_team_points': list_losing_team_points})
    
    # convert week and points to integer
    df['winning_team_points'] = df.apply(lambda x: int(x['winning_team_points']) if x['winning_team_points'] != '' else np.nan, axis=1)
    df['losing_team_points'] = df.apply(lambda x: int(x['losing_team_points']) if x['losing_team_points'] != '' else np.nan, axis=1)
    
    # get the data into a form we can do something with
    
    # get home and away teams
    list_home_team = []
    list_away_team = []
    # get home and away scores
    list_home_points = []
    list_away_points = []
    # iterate through all rows
    for i in range(df.shape[0]):
        if df['game_loc'].iloc[i] == '@':
            # get teams
            home_team = df['losing_team'].iloc[i]
            away_team = df['winning_team'].iloc[i]
            # get points
            home_points = df['losing_team_points'].iloc[i]
            away_points = df['winning_team_points'].iloc[i]
        else:
            # get teams
            home_team = df['winning_team'].iloc[i]
            away_team = df['losing_team'].iloc[i]
            # get points
            home_points = df['winning_team_points'].iloc[i]
            away_points = df['losing_team_points'].iloc[i]
        # append to lists
        # teams
        list_home_team.append(home_team)
        list_away_team.append(away_team)
        # points
        list_home_points.append(home_points)
        list_away_points.append(away_points)
    
    # put into df
    df = pd.DataFrame({'week': df['week'],
                       'home_team': list_home_team,
                       'away_team': list_away_team,
                       'home_points': list_home_points,
                       'away_points': list_away_points})  
    
    # get games for the upcoming week
    upcoming_week = np.min(df[df.isnull().any(axis=1)]['week'])  
    
    # get the matchups for the upcoming week
    df_upcoming_week = df[df['week'] == upcoming_week]
    
    # drop rows with missing values
    df = df.dropna(subset=['home_points'])
    
    # instantiate lists
    list_home_score = []
    list_away_score = []
    list_home_win_prob = []
    for i in range(df_upcoming_week.shape[0]):
        # get home_team
        home_team = df_upcoming_week['home_team'].iloc[i]
        # get away_team
        away_team = df_upcoming_week['away_team'].iloc[i]
    
        # check to make sure each team is in the respective lists
        if home_team in list(df['home_team']) and away_team in list(df['away_team']):
            # simulate game
            simulated_game = game_predictions(home_team_array=df['home_team'], 
                                              home_score_array=df['home_points'], 
                                              away_team_array=df['away_team'], 
                                              away_score_array=df['away_points'], 
                                              home_team=home_team, 
                                              away_team=away_team,
                                              n_simulations=n_simulations,
                                              weighted_mean=weighted_mean)
            # get the predicted home score
            home_score = simulated_game.mean_home_score
            # get the predicted away score
            away_score = simulated_game.mean_away_score
            # get the predicted win probability
            home_win_prob = simulated_game.prop_home_win
        else:
            home_score = 'NA'
            away_score = 'NA'
            home_win_prob = 'NA'
        # append to lists
        list_home_score.append(home_score)
        list_away_score.append(away_score)
        list_home_win_prob.append(home_win_prob)
        
    # put into df
    df_upcoming_week['home_points'] = list_home_score
    df_upcoming_week['away_points'] = list_away_score
    df_upcoming_week['home_win_prob'] = list_home_win_prob
    
    # choose the winning team
    df_upcoming_week['winning_team'] = df_upcoming_week.apply(lambda x: x['home_team'] if x['home_points'] >= x['away_points'] else x['away_team'], axis=1)
    
    # return df_upcoming_week
    return df_upcoming_week


# define function for nfl_season_simulation
def nfl_season_simulation(year, weighted_mean=False, n_simulations=1000):
    # get url
    r = requests.get('https://www.pro-football-reference.com/years/2019/games.htm'.format(year))
    # get content of page
    soup = BeautifulSoup(r.content, 'html.parser')
    # get all table rows
    table_rows = soup.find_all('tr')

    # instantiate empty lists
    list_week = []
    list_winning_team = []
    list_game_location = []
    list_losing_team = []
    list_winning_team_points = []
    list_losing_team_points = []
    # for each row
    for i in range(1, len(table_rows)):
        # get a row
        row = soup.find_all('tr')[i]
        # get all td elements
        td = row.find_all('td')
        # if td is not an empty list
        if td:
            # get week
            week = int(row.find('th').text)
            list_week.append(week)
            # get winning team
            winning_team = td[3].find('a').text
            list_winning_team.append(winning_team)
            # get game location
            game_location = td[4].text
            list_game_location.append(game_location)
            # get losing team
            losing_team = td[5].find('a').text
            list_losing_team.append(losing_team)
            # get winning team points
            winning_team_points = td[7].text
            list_winning_team_points.append(winning_team_points)
            # get losing team points
            losing_team_points = td[8].text
            list_losing_team_points.append(losing_team_points)

    # put into df
    df = pd.DataFrame({'week': list_week,
                       'winning_team': list_winning_team,
                       'game_loc': list_game_location,
                       'losing_team': list_losing_team,
                       'winning_team_points': list_winning_team_points,
                       'losing_team_points': list_losing_team_points})

    # convert week and points to integer
    df['winning_team_points'] = df.apply(lambda x: int(x['winning_team_points']) if x['winning_team_points'] != '' else np.nan, axis=1)
    df['losing_team_points'] = df.apply(lambda x: int(x['losing_team_points']) if x['losing_team_points'] != '' else np.nan, axis=1)

    # get the data into a format we can do something with
    
    # get home and away teams
    list_home_team = []
    list_away_team = []
    # get home and away scores
    list_home_points = []
    list_away_points = []
    # iterate through all rows
    for i in range(df.shape[0]):
        if df['game_loc'].iloc[i] == '@':
            # get teams
            home_team = df['losing_team'].iloc[i]
            away_team = df['winning_team'].iloc[i]
            # get points
            home_points = df['losing_team_points'].iloc[i]
            away_points = df['winning_team_points'].iloc[i]
        else:
            # get teams
            home_team = df['winning_team'].iloc[i]
            away_team = df['losing_team'].iloc[i]
            # get points
            home_points = df['winning_team_points'].iloc[i]
            away_points = df['losing_team_points'].iloc[i]
        # append to lists
        # teams
        list_home_team.append(home_team)
        list_away_team.append(away_team)
        # points
        list_home_points.append(home_points)
        list_away_points.append(away_points)

    # put into df
    df = pd.DataFrame({'week': df['week'],
                       'home_team': list_home_team,
                       'away_team': list_away_team,
                       'home_points': list_home_points,
                       'away_points': list_away_points})  

    # get the played games
    df_played_games = df.dropna(subset=['home_points'])    
        
    # get the unplayed games
    df_unplayed_games = df[df.isnull().any(axis=1)]

    # instantiate lists
    list_home_score = []
    list_away_score = []
    list_home_win_prob = []
    for i in range(df_unplayed_games.shape[0]):
        # get home_team
        home_team = df_unplayed_games['home_team'].iloc[i]
        # get away_team
        away_team = df_unplayed_games['away_team'].iloc[i]
    
        # check to make sure each team is in the respective lists
        if home_team in list(df_played_games['home_team']) and away_team in list(df_played_games['away_team']):
            # simulate game
            simulated_game = game_predictions(home_team_array=df_played_games['home_team'], 
                                              home_score_array=df_played_games['home_points'], 
                                              away_team_array=df_played_games['away_team'], 
                                              away_score_array=df_played_games['away_points'], 
                                              home_team=home_team, 
                                              away_team=away_team,
                                              n_simulations=n_simulations,
                                              weighted_mean=weighted_mean)
            # get the predicted home score
            home_score = simulated_game.mean_home_score
            # get the predicted away score
            away_score = simulated_game.mean_away_score
            # get the predicted win probability
            home_win_prob = simulated_game.prop_home_win
        else:
            home_score = 'NA'
            away_score = 'NA'
            home_win_prob = 'NA'
        # append to lists
        list_home_score.append(home_score)
        list_away_score.append(away_score)
        list_home_win_prob.append(home_win_prob)
    
    # put into df_unplayed_games
    df_unplayed_games['home_points'] = list_home_score
    df_unplayed_games['away_points'] = list_away_score
    df_unplayed_games['home_win_prob'] = list_home_win_prob

    # choose the winning team
    df_unplayed_games['winning_team'] = df_unplayed_games.apply(lambda x: x['home_team'] if x['home_points'] >= x['away_points'] else x['away_team'], axis=1)

    # make columns in df_played_games match with df_unplayed_games
    df_played_games['home_win_prob'] = df_played_games.apply(lambda x: 1.0 if x['home_points'] > x['away_points'] else 0.0, axis=1)
    df_played_games['winning_team'] = df_played_games.apply(lambda x: x['home_team'] if x['home_points'] >= x['away_points'] else x['away_team'], axis=1)

    # append df_unplayed_games to df_played_games
    df_simulated_season = df_played_games.append(df_unplayed_games)

    # get number of wins for each team

    # get unique teams
    df_unique_teams = pd.DataFrame(pd.unique(df['home_team']))

    # get the wins for each team
    list_unique_winning_teams = list(pd.value_counts(df_simulated_season['winning_team']).index)
    list_n_wins = list(pd.value_counts(df_simulated_season['winning_team']))

    # put into a df
    df_predicted_wins = pd.DataFrame({'team': list_unique_winning_teams,
                                      'wins': list_n_wins})
    
    # left join df_unique_teams and df_predicted_wins
    df_final_win_predictions = pd.merge(left=df_unique_teams, right=df_predicted_wins,
                                        left_on=df_unique_teams[0], right_on='team',
                                        how='left').fillna(0)

    # drop the col we dont want
    df_final_win_predictions.drop([0], axis=1, inplace=True)    

    # get predicted losses
    df_final_win_predictions['losses'] = 16 - df_final_win_predictions['wins']

    # sort by wins
    df_final_win_predictions = df_final_win_predictions.sort_values(by=['wins'], ascending=False)

    # get the conference for each team
    df_nfl_teams_conferences = pd.read_csv('https://raw.githubusercontent.com/aaronengland/data/master/nfl_teams_conferences.csv')

    # left join df_final_win_predictions and df_nfl_teams_conferences on team name
    df_final_win_predictions_conf = pd.merge(left=df_final_win_predictions, right=df_nfl_teams_conferences, 
                                             left_on='team', right_on='Name',
                                             how='left')

    # drop the cols we don't need
    df_final_win_predictions_conf.drop(['Unnamed: 0', 'ID', 'Name'], axis=1, inplace=True)

    # separate into AFC and NFC
    df_NFC = df_final_win_predictions_conf[df_final_win_predictions_conf['Conference'] == 'NFC']
    df_AFC = df_final_win_predictions_conf[df_final_win_predictions_conf['Conference'] == 'AFC']

    # group by division
    df_NFC_div = df_NFC.sort_values(by=['Division', 'wins'], ascending=False)
    df_AFC_div = df_AFC.sort_values(by=['Division','wins'], ascending=False)

    # define attributes class
    class attributes:
        def __init__(self, df_simulated_season, df_final_win_predictions_conf, df_NFC, df_AFC, df_NFC_div, df_AFC_div):
            self.df_simulated_season = df_simulated_season
            self.df_final_win_predictions_conf = df_final_win_predictions_conf
            self.df_NFC = df_NFC
            self.df_AFC = df_AFC
            self.df_NFC_div = df_NFC_div
            self.df_AFC_div = df_AFC_div
    # save as returnable object
    x = attributes(df_simulated_season, df_final_win_predictions_conf, df_NFC, df_AFC, df_NFC_div, df_AFC_div)
    return x








