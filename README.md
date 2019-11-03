# nfl_predictions

A package for scraping the user-defined NFL season's schedule/results from [here](https://www.pro-football-reference.com/) (`scrape_schedule`), tune the model's hyperparameters on the previous week (`tune_hyperparameters`), simulate the current week (`simulate_current_week`) and remainder of season (`nfl_season_simulation`), and generate postseason probabilities (`nfl_postseason_probabilities`) using the [`game_predictions`](https://github.com/aaronengland/game_predictions/blob/master/README.md) algorithm.

To install, use: `pip install git+https://github.com/aaronengland/nfl_predictions.git`

---

## scrape_nfl_schedule

Arguments:
- `year`: season to scrape.

Returns a data frame with the columns: `week`, `home_team`, `away_team`, `home_points`, `away_points`, and `winning_team`.

---

## tune_nfl_hyperparameters

Arguments:
- `df`: data frame produced from the `scrape_schedule` function.
- `week_to_simulate`: the week in which to simulate.
- `list_outer_weighted_mean`: list of options for the outer weighted mean (see `game_predictions`).
- `list_distributions`: list of distributions to use for generating predictions (see `game_predictions`). 
- `list_inner_weighted_mean`: list of weights for the inner weighted mean (see `game_predictions`).
- `list_weight_home`: list of weights to apply to games where the home team is home (can only be used when `outer_weighted_mean` is `all_games_weighted`; see `game_predictions`).
- `list_weight_away`: list of weights to apply to games where the away team is away (can only be used when `outer_weighted_mean` is `al_games_weighted`; see `game_predictions`).
- `n_simulations`: number of simulations for each contest.

Returns a dictionary containing a data frame of hyperparameter combinations and the corresponding correctly predicted wins and total error in predicted vs. actual points scored.

---

## simulate_current_nfl_week

Arguments:
- `df`: data frame produced from the `scrape_schedule` function.
- `week_to_simulate`: the week in which to simulate.
- `dict_best_hyperparameters`: dictionary containing the best hyperparameters (returned from `tune_hyperparameters`).
- `n_simulations`: number of simulations for each contest.

Returns a data frame containing: `Week`, `Home`, `Away`, `Home Points`, `Away Points`, `Home Win Probability`, and `Winning Team`.

---

## simulate_nfl_season

Arguments:
- `df`: data frame produced from the `scrape_schedule` function.
- `dict_best_hyperparameters`: dictionary containing the best hyperparameters (returned from `tune_hyperparameters`).
- `n_simulations`: number of seasons to simulate.

Returns a data frame of the entire season's matchups with the actual scores of the games that have been played and the predicted scores of the games that have not been played.

---

## nfl_postseason_probabilities

Arguments:
- `df`: data frame produced from the `scrape_schedule` function.
- `dict_best_hyperparameters`: dictionary containing the best hyperparameters (returned from `tune_hyperparameters`).
- `n_simulations`: number of simulations for each contest.

Returns a data frame containing `team`, `prob_postseason` (probability of postseason), `prob_conf_champs` (probability of winning the conference), and `prob_superbowl_champ` (probability of winning the superbowl).

---

Example:

```
# dependencies
from nfl_predictions import scrape_nfl_schedule, tune_nfl_hyperparameters, simulate_current_nfl_week, simulate_nfl_season, nfl_postseason_probabilities

# save arguments
# year
year = 2019
# week to simulate
week_to_simulate = 9
# number of simulations
n_simulations = 1000

# scrape schedulde
df = scrape_nfl_schedule(year=2019)

# tune hyperparameters
hyperparams_tuned = tune_nfl_hyperparameters(df=df, 
                                             week_to_simulate=week_to_simulate, 
                                             list_central_tendency=['mean','median'], 
                                             list_distributions=['normal','poisson'], 
                                             list_inner_weighted_mean=['none','win_pct'], 
                                             list_weight_home=[1,2], 
                                             list_weight_away=[1,2], 
                                             n_simulations=n_simulations)

# get the best hyperparameters
dict_best_hyperparameters = hyperparams_tuned.get('dict_best_hyperparameters')

# simulate current week's games
df_predictions = simulate_current_nfl_week(df=df, 
                                           week_to_simulate=week_to_simulate, 
                                           dict_best_hyperparameters=dict_best_hyperparameters, 
                                           n_simulations=n_simulations)

# simulate season
season_simulation = simulate_nfl_season(df=df, 
                                        dict_best_hyperparameters=dict_best_hyperparameters, 
                                        n_simulations=n_simulations)

# get the final win totals
win_totals = season_simulation.get('final_win_predictions_conf')

# get postseason probabilities
postseason_prob = nfl_postseason_probabilities(df=df, 
                                               dict_best_hyperparameters=dict_best_hyperparameters,
                                               n_simulations=n_simulations)
```



