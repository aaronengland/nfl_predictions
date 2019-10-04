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
    # suppress the SettingWithCopyWarning
    pd.options.mode.chained_assignment = None
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

# define function for postseason probabilities
def nfl_postseason_probabilities(year, n_simulations, weighted_mean=False, weighted_mean_super_bowl=True):
    # suppress the SettingWithCopyWarning
    pd.options.mode.chained_assignment = None
    # get the unique teams
    list_teams = list(pd.read_csv('https://raw.githubusercontent.com/aaronengland/data/master/nfl_teams_conferences.csv')['Name'])
    # make data frame with just teams for postseason
    df_teams_postseason = pd.DataFrame({'team': list_teams})
    # make data frame with just teams for conference champs
    df_teams_conference_champs = pd.DataFrame({'team': list_teams})
    # make data frame with just teams for superbowl champs
    df_teams_superbowl_champs = pd.DataFrame({'team': list_teams})
    
    # user defined number of simulations
    for x in range(n_simulations):
        # simulate season
        simulated_season = nfl_season_simulation(year=year, weighted_mean=False, n_simulations=1)
        # get the simulated season to help with playoff predictions later
        df_entire_season = simulated_season.df_simulated_season
        
        #######################################################################
        # get division standings for each conference
        # nfc
        df_nfc_division_standings = simulated_season.df_NFC_div
        # afc
        df_afc_division_standings = simulated_season.df_AFC_div
        
        # put both into a list so we can loop through them
        list_df_conference_division_standings = [df_nfc_division_standings, df_afc_division_standings]
        
        #######################################################################
        # iterate through eacch df in list_conference_division_standings
        # instantiate empty lists
        list_list_playoff_teams = []
        list_conference_champs = []
        for df_conference_division_standings in list_df_conference_division_standings:
            # instantiate list of divisions
            list_divisions = ['North','East','South','West']
        
            # get division champs
            list_divsion_champs = []
            list_division_champ_wins = []
            for division in list_divisions:
                # get division champ
                division_champ = df_conference_division_standings[df_conference_division_standings['Division'] == division]['team'].iloc[0]
                # append to list
                list_divsion_champs.append(division_champ)
                # get wins
                division_champ_wins = df_conference_division_standings[df_conference_division_standings['Division'] == division]['wins'].iloc[0]
                # append to list
                list_division_champ_wins.append(division_champ_wins)
        
            # put into df
            df_postseason_teams = pd.DataFrame({'team': list_divsion_champs,
                                                'wins': list_division_champ_wins})
          
            # sort by wins so we can get seeds later
            df_postseason_teams = df_postseason_teams.sort_values(by=['wins'], ascending=False)
            
            # put series into lists
            list_playoff_teams = list(df_postseason_teams['team'])
            list_playoff_teams_wins = list(df_postseason_teams['wins'])
        
            # get the wildcard teams
        
            # sort df_NFC_division_standings
            df_conference_division_standings = df_conference_division_standings.sort_values(by=['wins'], ascending=False)
        
            # remove teams in list_divsion_champs
            df_conference_division_standings_no_champs = df_conference_division_standings[~df_conference_division_standings['team'].isin(list_divsion_champs)]
        
            # get the teams and wins for teams with the top 2 records
            for i in range(0,2):
                wildcard_team = df_conference_division_standings_no_champs['team'].iloc[i]
                # append to list
                list_playoff_teams.append(wildcard_team)
                # get wins
                wildcard_team_wins = df_conference_division_standings_no_champs['wins'].iloc[i]
                # append to list
                list_playoff_teams_wins.append(wildcard_team_wins)
        
            # put into df
            df_postseason_teams = pd.DataFrame({'team': list_playoff_teams,
                                                    'wins': list_playoff_teams_wins})
            # append list_playoff_teams to list_list_playoff_teams
            list_list_playoff_teams.extend(list_playoff_teams)
            
            ###################################################################
            # begin game simulations
            # define helper function to pick winners faster
            def who_won(predicted_home_score, predicted_away_score, home_team, away_team):
                # get winning team
                if predicted_home_score >= predicted_away_score:
                    winning_team = home_team
                else:
                    winning_team = away_team
                return winning_team
            
            # wildcard game 1
            # 3 vs 6
            home_team = df_postseason_teams['team'].iloc[2]
            away_team = df_postseason_teams['team'].iloc[5]
            # simulate game
            game_simulation = game_predictions(home_team_array=df_entire_season['home_team'],
                                               home_score_array=df_entire_season['home_points'],
                                               away_team_array=df_entire_season['away_team'],
                                               away_score_array=df_entire_season['away_points'],
                                               home_team=home_team,
                                               away_team=away_team,
                                               n_simulations=1,
                                               weighted_mean=weighted_mean)
        
            # get winner
            winning_team_1 = who_won(predicted_home_score=game_simulation.mean_home_score, 
                                     predicted_away_score=game_simulation.mean_away_score, 
                                     home_team=home_team, away_team=home_team)
        
            # 4 vs 5
            home_team = df_postseason_teams['team'].iloc[3]
            away_team = df_postseason_teams['team'].iloc[4]
                # simulate game
            game_simulation = game_predictions(home_team_array=df_entire_season['home_team'],
                                               home_score_array=df_entire_season['home_points'],
                                               away_team_array=df_entire_season['away_team'],
                                               away_score_array=df_entire_season['away_points'],
                                               home_team=home_team,
                                               away_team=away_team,
                                               n_simulations=1,
                                               weighted_mean=weighted_mean)
        
            # get winner
            winning_team_2 = who_won(predicted_home_score=game_simulation.mean_home_score, 
                                     predicted_away_score=game_simulation.mean_away_score, 
                                     home_team=home_team, away_team=home_team)
        
            # 1 vs winning_team_2
            home_team = df_postseason_teams['team'].iloc[0]
            away_team = winning_team_2
            # simulate game
            game_simulation = game_predictions(home_team_array=df_entire_season['home_team'],
                                               home_score_array=df_entire_season['home_points'],
                                               away_team_array=df_entire_season['away_team'],
                                               away_score_array=df_entire_season['away_points'],
                                               home_team=home_team,
                                               away_team=away_team,
                                               n_simulations=1,
                                               weighted_mean=weighted_mean)
            
            # get winner
            winning_team_3 = who_won(predicted_home_score=game_simulation.mean_home_score, 
                                     predicted_away_score=game_simulation.mean_away_score, 
                                     home_team=home_team, away_team=home_team)
        
        
            # 2 vs winning_team_1
            home_team = df_postseason_teams['team'].iloc[1]
            away_team = winning_team_1
            # simulate game
            game_simulation = game_predictions(home_team_array=df_entire_season['home_team'],
                                               home_score_array=df_entire_season['home_points'],
                                               away_team_array=df_entire_season['away_team'],
                                               away_score_array=df_entire_season['away_points'],
                                               home_team=home_team,
                                               away_team=away_team,
                                               n_simulations=1,
                                               weighted_mean=weighted_mean)
            
            # get winner
            winning_team_4 = who_won(predicted_home_score=game_simulation.mean_home_score, 
                                     predicted_away_score=game_simulation.mean_away_score, 
                                     home_team=home_team, away_team=home_team)
        
            # winning_team_3 vs winning_team_4
            # find index of winning_team_3
            index_winning_team_3 = list(df_postseason_teams['team']).index(winning_team_3)
            # find index of winning_team_4
            index_winning_team_4 = list(df_postseason_teams['team']).index(winning_team_4)
            # decide who is home/away based on seed
            if index_winning_team_3 < index_winning_team_4:
                home_team = winning_team_3
                away_team = winning_team_4
            else:
                home_team = winning_team_4
                away_team = winning_team_3
            # simulate game
            game_simulation = game_predictions(home_team_array=df_entire_season['home_team'],
                                               home_score_array=df_entire_season['home_points'],
                                               away_team_array=df_entire_season['away_team'],
                                               away_score_array=df_entire_season['away_points'],
                                               home_team=home_team,
                                               away_team=away_team,
                                               n_simulations=1,
                                               weighted_mean=weighted_mean)
            
            # get winner
            conference_champs = who_won(predicted_home_score=game_simulation.mean_home_score, 
                                        predicted_away_score=game_simulation.mean_away_score, 
                                        home_team=home_team, away_team=home_team)
            
            # append conference_champs to list_conference_champs
            list_conference_champs.append(conference_champs)
            
        ############################### super bowl ####################################
        # for super bowl we dont want to separate games by home/away so we will change the function a little
        # make sure final_afc_points and final_nfc_points are equal
        final_afc_points = 0
        final_nfc_points = 0
        while final_afc_points == final_nfc_points:
            # instantiate lists
            list_pred_points_target = []
            list_pred_points_opp_target = []
            # iterate through home and away teams
            for conference_champs in list_conference_champs:
                # subset to all games involving the conference_champs
                df_target_scores = df_entire_season[(df_entire_season['home_team'] == conference_champs) | (df_entire_season['away_team'] == conference_champs)]
                # get home scores only
                df_target_scores['target_points_only'] = df_target_scores.apply(lambda x: x['home_points'] if x['home_team'] == conference_champs else x['away_points'], axis=1)
                # get home opponents scores only
                df_target_scores['target_opp_points_only'] = df_target_scores.apply(lambda x: x['away_points'] if x['home_team'] == conference_champs else x['home_points'], axis=1)
                # calculate mean points scored by home team
                if weighted_mean_super_bowl==True:
                    # get mean points scored by heom_team
                    mean_points_target = np.average(df_target_scores['target_points_only'], weights=[x for x in range(1, df_target_scores.shape[0]+1)])
                else:
                    mean_points_target = np.mean(df_target_scores['target_points_only'])
                # generate a random number from a poisson distribution with this number as the lambda
                pred_points_target = np.random.poisson(mean_points_target, 1)[0]
                # append to list_pred_points_target
                list_pred_points_target.append(pred_points_target)
                
                # calculate mean points allowed by home team
                if weighted_mean_super_bowl==True:
                    # get mean points scored by heom_team
                    mean_opp_points_target = np.average(df_target_scores['target_opp_points_only'], weights=[x for x in range(1, df_target_scores.shape[0]+1)])
                else:
                    mean_opp_points_target = np.mean(df_target_scores['target_opp_points_only'])
                # generate a random number from a poisson distribution with this number as the lambda
                pred_points_opp_target = np.random.poisson(mean_opp_points_target, 1)[0]
                # append to list_pred_points_opp_target
                list_pred_points_opp_target.append(pred_points_opp_target)
            
            # calculate predicted points scored for each team in list_conference_champs
            final_afc_points = (list_pred_points_target[0] + list_pred_points_opp_target[1]) / 2
            final_nfc_points = (list_pred_points_target[1] + list_pred_points_opp_target[0]) / 2
        
        # get super bowl champ
        if final_afc_points > final_nfc_points:
            superbowl_champs = list_conference_champs[0]
        else:
            superbowl_champs = list_conference_champs[1]
                
        ###############################################################################
        # mark if a team made postseason
        df_teams_postseason['sim_{0}'.format(x)] = df_teams_postseason.apply(lambda x: 1 if x['team'] in list_list_playoff_teams else 0, axis=1)
        # mark if team was a conference champion
        df_teams_conference_champs['sim_{0}'.format(x)] = df_teams_conference_champs.apply(lambda x: 1 if x['team'] in list_conference_champs else 0, axis=1)
        # mark if team won superbowl
        df_teams_superbowl_champs['sim_{0}'.format(x)] = df_teams_superbowl_champs.apply(lambda x: 1 if x['team'] == superbowl_champs else 0, axis=1)
    
    # get the probability of each
    # put all probability cols into the same df
    df = pd.DataFrame({'team': list_teams,
                       'prob_postseason': df_teams_postseason.mean(axis=1),
                       'prob_conf_champ': df_teams_conference_champs.mean(axis=1),
                       'prob_superbowl_champ': df_teams_superbowl_champs.mean(axis=1)})
    return df






