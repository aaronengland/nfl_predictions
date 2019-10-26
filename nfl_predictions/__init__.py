import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from game_predictions import game_predictions
import datetime

# define function for scraping nfl schedule/results
def scrape_schedule(year):
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
        
    # get winning team
    df['winning_team'] = df.apply(lambda x: x['home_team'] if x['home_points'] > x['away_points'] else x['away_team'], axis=1)
    # return df
    return df

# define function to tune model parameters
def tune_hyperparameters(df, week_to_simulate, list_outer_weighted_mean, list_distributions, list_inner_weighted_mean, list_weight_home, list_weight_away, n_simulations=1000):
    # we will tune our model on one week before week_to_simulate
    week_to_simulate_train = week_to_simulate - 1

    # drop everything after week X
    df_data = df[df['week'] < week_to_simulate_train]
    # get the games to simulate
    df_predictions = df[df['week'] == week_to_simulate_train]
    
    # time to tune model
    time_start = datetime.datetime.now()
    # instantiate empty list
    list_sum_correct = []
    list_sum_error = []
    list_dict_outcomes = []
    for outer_weighted_mean in list_outer_weighted_mean:
        # if all_games_weighted
        if outer_weighted_mean == 'all_games_weighted':
            for distribution in list_distributions:
                for weight_home in list_weight_home:
                    for weight_away in list_weight_away:
                        # we only want equal weights when both equal 1
                        if (weight_home + weight_away == 2) or (weight_home != weight_away):
                            for inner_weighted_mean in list_inner_weighted_mean:
                                # predict every game in df_predictions
                                df_predictions['pred_outcome'] = df_predictions.apply(lambda x: game_predictions(home_team_array=df_data['home_team'], 
                                                                                                                 home_score_array=df_data['home_points'], 
                                                                                                                 away_team_array=df_data['away_team'], 
                                                                                                                 away_score_array=df_data['away_points'], 
                                                                                                                 home_team=x['home_team'], 
                                                                                                                 away_team=x['away_team'], 
                                                                                                                 outer_weighted_mean=outer_weighted_mean, 
                                                                                                                 inner_weighted_mean=inner_weighted_mean, 
                                                                                                                 weight_home=weight_home,
                                                                                                                 weight_away=weight_away,
                                                                                                                 n_simulations=n_simulations), axis=1)
                
                                # get winning team
                                df_predictions['pred_winning_team'] = df_predictions.apply(lambda x: (x['pred_outcome']).get('winning_team'), axis=1)
                                # get number right
                                sum_correct = np.sum(df_predictions.apply(lambda x: 1 if x['winning_team'] == x['pred_winning_team'] else 0, axis=1))
                                # append to list
                                list_sum_correct.append(sum_correct)
                                
                                # get the total spread difference so we can sort by that as well
                                # get predicted home points
                                df_predictions['pred_home_points'] = df_predictions.apply(lambda x: (x['pred_outcome']).get('mean_home_pts'), axis=1)
                                # get predicted away points
                                df_predictions['pred_away_points'] = df_predictions.apply(lambda x: (x['pred_outcome']).get('mean_away_pts'), axis=1)
                                # get absolute difference between home_points and pred_home_points
                                df_predictions['pred_home_points_error'] = df_predictions.apply(lambda x: np.abs(x['home_points'] - x['pred_home_points']), axis=1)
                                # get absolute difference between away_points and pred_away_points
                                df_predictions['pred_away_points_error'] = df_predictions.apply(lambda x: np.abs(x['away_points'] - x['pred_away_points']), axis=1)
                                # sum pred_home_points_error and pred_away_points_error
                                sum_error = np.sum(df_predictions['pred_home_points_error']) + np.sum(df_predictions['pred_away_points_error'])
                                # append to list
                                list_sum_error.append(sum_error)
                                
                                # create dictionary
                                dict_outcomes = {'outer_weighted_mean': outer_weighted_mean,
                                                 'distribution': distribution,
                                                 'weight_home': weight_home,
                                                 'weight_away': weight_away,
                                                 'inner_weighted_mean': inner_weighted_mean}
                                # append to list
                                list_dict_outcomes.append(dict_outcomes)
        # else (i.e., outer_weighted_mean != 'all_games_weighted')
        else:
            for distribution in list_distributions:
                # save weight home and weight away for the dictionary
                weight_home = None
                weight_away = None
                for inner_weighted_mean in list_inner_weighted_mean:
                    # predict every game in df_predictions
                    df_predictions['pred_outcome'] = df_predictions.apply(lambda x: game_predictions(home_team_array=df_data['home_team'], 
                                                                                                     home_score_array=df_data['home_points'], 
                                                                                                     away_team_array=df_data['away_team'], 
                                                                                                     away_score_array=df_data['away_points'], 
                                                                                                     home_team=x['home_team'], 
                                                                                                     away_team=x['away_team'], 
                                                                                                     outer_weighted_mean=outer_weighted_mean, 
                                                                                                     inner_weighted_mean=inner_weighted_mean, 
                                                                                                     n_simulations=n_simulations), axis=1)
            
                    # get winning team
                    df_predictions['pred_winning_team'] = df_predictions.apply(lambda x: (x['pred_outcome']).get('winning_team'), axis=1)
                    # get number right
                    sum_correct = np.sum(df_predictions.apply(lambda x: 1 if x['winning_team'] == x['pred_winning_team'] else 0, axis=1))
                    # append to list
                    list_sum_correct.append(sum_correct)
                    
                    # get the total spread difference so we can sort by that as well
                    # get predicted home points
                    df_predictions['pred_home_points'] = df_predictions.apply(lambda x: (x['pred_outcome']).get('mean_home_pts'), axis=1)
                    # get predicted away points
                    df_predictions['pred_away_points'] = df_predictions.apply(lambda x: (x['pred_outcome']).get('mean_away_pts'), axis=1)
                    # get absolute difference between home_points and pred_home_points
                    df_predictions['pred_home_points_error'] = df_predictions.apply(lambda x: np.abs(x['home_points'] - x['pred_home_points']), axis=1)
                    # get absolute difference between away_points and pred_away_points
                    df_predictions['pred_away_points_error'] = df_predictions.apply(lambda x: np.abs(x['away_points'] - x['pred_away_points']), axis=1)
                    # sum pred_home_points_error and pred_away_points_error
                    sum_error = np.sum(df_predictions['pred_home_points_error']) + np.sum(df_predictions['pred_away_points_error'])
                    # append to list
                    list_sum_error.append(sum_error)
                    
                    # create dictionary
                    dict_outcomes = {'outer_weighted_mean': outer_weighted_mean,
                                     'distribution': distribution,
                                     'weight_home': weight_home,
                                     'weight_away': weight_away,
                                     'inner_weighted_mean': inner_weighted_mean}
                    # append to list
                    list_dict_outcomes.append(dict_outcomes)
    # get elapsed time
    elapsed_time = (datetime.datetime.now() - time_start).seconds
    # print message
    print('Time to tune the model: {0} min'.format(elapsed_time/60))
    
    # put outcome lists into a df
    df_outcomes = pd.DataFrame({'hyperparameters': list_dict_outcomes,
                                'n_correct': list_sum_correct,
                                'error': list_sum_error})
    
    # sort values descending
    df_outcomes_sorted = df_outcomes.sort_values(by=['n_correct','error'], ascending=[False, True])
    
    # get the best set of hyperparameters
    dict_best_hyperparameters = df_outcomes_sorted['hyperparameters'].iloc[0]
    
    # make a dictionary with output
    dict_results = {'df_outcomes_sorted': df_outcomes_sorted,
                    'dict_best_hyperparameters': dict_best_hyperparameters}
    
    # return dict_results
    return dict_results

# define function to simulate current week's games
def simulate_current_week(df, week_to_simulate, dict_best_hyperparameters, n_simulations):
    # drop everything after week X
    df_data = df[df['week'] < week_to_simulate]
    # get the games to simulate
    df_predictions = df[df['week'] == week_to_simulate]
    
    # generate my predictions
    df_predictions['pred_outcome'] = df_predictions.apply(lambda x: game_predictions(home_team_array=df_data['home_team'], 
                                                                                     home_score_array=df_data['home_points'], 
                                                                                     away_team_array=df_data['away_team'], 
                                                                                     away_score_array=df_data['away_points'], 
                                                                                     home_team=x['home_team'], 
                                                                                     away_team=x['away_team'], 
                                                                                     outer_weighted_mean=dict_best_hyperparameters.get('outer_weighted_mean'), 
                                                                                     distribution=dict_best_hyperparameters.get('distribution'),
                                                                                     inner_weighted_mean=dict_best_hyperparameters.get('inner_weighted_mean'), 
                                                                                     weight_home=dict_best_hyperparameters.get('weight_home'),
                                                                                     weight_away=dict_best_hyperparameters.get('weight_away'),
                                                                                     n_simulations=n_simulations), axis=1)
    # drop the winning_team column
    df_predictions.drop(columns=['winning_team'], inplace=True, axis=1)
    
    # make a table we can copy and paste into our index.html
    for key in df_predictions['pred_outcome'].iloc[0]:
        df_predictions[key] = df_predictions.apply(lambda x: x['pred_outcome'].get(key), axis=1)
    
    # drop cols
    df_predictions.drop(['home_points','away_points','pred_outcome'], inplace=True, axis=1)
    
    # rename the cols
    df_predictions.columns = ['Week','Home','Away','Home Points','Away Points','Home Win Probability','Winning Team']
    
    # return df_predictions
    return df_predictions

# define function to simulate a season
def nfl_season_simulation(df, dict_best_hyperparameters, n_simulations=1000): 
    # get the played games
    df_played_games = df.dropna(subset=['home_points'])    
    # get the unplayed games
    df_unplayed_games = df[df.isnull().any(axis=1)]
    
    # apply game_predictions to df_unplayed_games
    df_unplayed_games['predictions'] = df_unplayed_games.apply(lambda x: game_predictions(home_team_array=df_played_games['home_team'],
                                                                                          home_score_array=df_played_games['home_points'],
                                                                                          away_team_array=df_played_games['away_team'],
                                                                                          away_score_array=df_played_games['away_points'],
                                                                                          home_team=x['home_team'],
                                                                                          away_team=x['away_team'],
                                                                                          distribution=dict_best_hyperparameters.get('distribution'),
                                                                                          outer_weighted_mean=dict_best_hyperparameters.get('outer_weighted_mean'),
                                                                                          inner_weighted_mean=dict_best_hyperparameters.get('inner_weighted_mean'),
                                                                                          weight_home=dict_best_hyperparameters.get('weight_home'),
                                                                                          weight_away=dict_best_hyperparameters.get('weight_away'),
                                                                                          n_simulations=n_simulations), axis=1)
    
    # put into df_unplayed_games
    df_unplayed_games['home_points'] = df_unplayed_games.apply(lambda x: x['predictions'].get('mean_home_pts'), axis=1)
    df_unplayed_games['away_points'] = df_unplayed_games.apply(lambda x: x['predictions'].get('mean_away_pts'), axis=1)
    df_unplayed_games['winning_team'] = df_unplayed_games.apply(lambda x: x['predictions'].get('winning_team'), axis=1)
    
    # drop predictions col
    df_unplayed_games.drop(['predictions'], axis=1, inplace=True)

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

    # create a dictionary with output
    results = {'simulated_season': df_simulated_season,
               'final_win_predictions_conf': df_final_win_predictions_conf,
               'nfc': df_NFC,
               'afc': df_AFC,
               'nfc_div': df_NFC_div,
               'afc_div': df_AFC_div}

    # return results
    return results

# define function for postseason probabilities
def nfl_postseason_probabilities(df, dict_best_hyperparameters, n_simulations):
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
        simulated_season = nfl_season_simulation(df=df, 
                                                 dict_best_hyperparameters=dict_best_hyperparameters,
                                                 n_simulations=1)
        # get the simulated season to help with playoff predictions later
        #df_entire_season = simulated_season.get('final_win_predictions_conf')
        
        # get division standings for each conference
        # nfc
        df_nfc_division_standings = simulated_season.get('nfc_div')
        # afc
        df_afc_division_standings = simulated_season.get('afc_div')
        
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
            # wildcard game 1
            # 3 vs 6
            home_team = df_postseason_teams['team'].iloc[2]
            away_team = df_postseason_teams['team'].iloc[5]
            
            # get the games thathave been played
            df_played = df.dropna(subset=['home_points'])
            
            # simulate game
            game_simulation = game_predictions(home_team_array=df_played['home_team'],
                                               home_score_array=df_played['home_points'],
                                               away_team_array=df_played['away_team'],
                                               away_score_array=df_played['away_points'],
                                               home_team=home_team,
                                               away_team=away_team,
                                               outer_weighted_mean=dict_best_hyperparameters.get('outer_weighted_mean'),
                                               inner_weighted_mean=dict_best_hyperparameters.get('inner_weighted_mean'),
                                               weight_home=dict_best_hyperparameters.get('weight_home'),
                                               weight_away=dict_best_hyperparameters.get('weight_away'),
                                               n_simulations=1)
        
            # get winner
            winning_team_1 = game_simulation.get('winning_team')
        
            # 4 vs 5
            home_team = df_postseason_teams['team'].iloc[3]
            away_team = df_postseason_teams['team'].iloc[4]
                # simulate game
            game_simulation = game_predictions(home_team_array=df_played['home_team'],
                                               home_score_array=df_played['home_points'],
                                               away_team_array=df_played['away_team'],
                                               away_score_array=df_played['away_points'],
                                               home_team=home_team,
                                               away_team=away_team,
                                               outer_weighted_mean=dict_best_hyperparameters.get('outer_weighted_mean'),
                                               inner_weighted_mean=dict_best_hyperparameters.get('inner_weighted_mean'),
                                               weight_home=dict_best_hyperparameters.get('weight_home'),
                                               weight_away=dict_best_hyperparameters.get('weight_away'),
                                               n_simulations=1)
        
            # get winner
            winning_team_2 = game_simulation.get('winning_team')
        
            # 1 vs winning_team_2
            home_team = df_postseason_teams['team'].iloc[0]
            away_team = winning_team_2
            # simulate game
            game_simulation = game_predictions(home_team_array=df_played['home_team'],
                                               home_score_array=df_played['home_points'],
                                               away_team_array=df_played['away_team'],
                                               away_score_array=df_played['away_points'],
                                               home_team=home_team,
                                               away_team=away_team,
                                               outer_weighted_mean=dict_best_hyperparameters.get('outer_weighted_mean'),
                                               inner_weighted_mean=dict_best_hyperparameters.get('inner_weighted_mean'),
                                               weight_home=dict_best_hyperparameters.get('weight_home'),
                                               weight_away=dict_best_hyperparameters.get('weight_away'),
                                               n_simulations=1)
            
            # get winner
            winning_team_3 = game_simulation.get('winning_team')
        
            # 2 vs winning_team_1
            home_team = df_postseason_teams['team'].iloc[1]
            away_team = winning_team_1
            # simulate game
            game_simulation = game_predictions(home_team_array=df_played['home_team'],
                                               home_score_array=df_played['home_points'],
                                               away_team_array=df_played['away_team'],
                                               away_score_array=df_played['away_points'],
                                               home_team=home_team,
                                               away_team=away_team,
                                               outer_weighted_mean=dict_best_hyperparameters.get('outer_weighted_mean'),
                                               inner_weighted_mean=dict_best_hyperparameters.get('inner_weighted_mean'),
                                               weight_home=dict_best_hyperparameters.get('weight_home'),
                                               weight_away=dict_best_hyperparameters.get('weight_away'),
                                               n_simulations=1)
            
            # get winner
            winning_team_4 = game_simulation.get('winning_team')
        
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
            game_simulation = game_predictions(home_team_array=df_played['home_team'],
                                               home_score_array=df_played['home_points'],
                                               away_team_array=df_played['away_team'],
                                               away_score_array=df_played['away_points'],
                                               home_team=home_team,
                                               away_team=away_team,
                                               outer_weighted_mean=dict_best_hyperparameters.get('outer_weighted_mean'),
                                               inner_weighted_mean=dict_best_hyperparameters.get('inner_weighted_mean'),
                                               weight_home=dict_best_hyperparameters.get('weight_home'),
                                               weight_away=dict_best_hyperparameters.get('weight_away'),
                                               n_simulations=n_simulations)
            
            # get winner
            conference_champs = game_simulation.get('winning_team')
            
            # append conference_champs to list_conference_champs
            list_conference_champs.append(conference_champs)
            import random
        ########################### super bowl ####################################
        # generate random number so home_team and away_team are randomly chosen
        rand_number_1 = random.randint(0, 1)
        # assign to home_team/away_team
        if rand_number_1 == 0:
            rand_number_2 = 1
        else:
            rand_number_2 = 0
        
        # get home and away teams
        home_team = list_conference_champs[rand_number_1]
        away_team = list_conference_champs[rand_number_2]
        
        # simulate game
        game_simulation = game_predictions(home_team_array=df_played['home_team'],
                                           home_score_array=df_played['home_points'],
                                           away_team_array=df_played['away_team'],
                                           away_score_array=df_played['away_points'],
                                           home_team=home_team,
                                           away_team=away_team,
                                           outer_weighted_mean=dict_best_hyperparameters.get('outer_weighted_mean'),
                                           inner_weighted_mean=dict_best_hyperparameters.get('inner_weighted_mean'),
                                           weight_home=dict_best_hyperparameters.get('weight_home'),
                                           weight_away=dict_best_hyperparameters.get('weight_away'),
                                           n_simulations=n_simulations)
            
        # get winning team
        superbowl_champs = game_simulation.get('winning_team')
        
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






