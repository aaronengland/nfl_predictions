# nfl_predictions

A package for scraping the user-defined NFL season's schedule/results and simulating the upcoming week's games (`nfl_pickem`) and the unplayed games for the entire season (`nfl_season_simulation`) using the `game_predictions` algorithm.

To install, use: `pip install git+https://github.com/aaronengland/nfl_predictions.git`

## nfl_pickem

Arguments:
- `year`: season to simulate.
- `weighted_mean`: use of weighted mean in simulation (boolean; default=False; False is recommended for early in the season while True is recommended for later games).
- `n_simulations`: number of simulations for each game (default=1000).

Returns a data frame with predicted results for the upcoming week's games.

Example:

```
from nfl_predictions import nfl_pickem

# simulate upcoming week
upcoming_week_simulation = nfl_pickem(year=2019, 
                                      weighted_mean=False, 
                                      n_simulations=1000)

# view results
upcoming_week_simulation
```

---

## nfl_season_simulation

Arguments:
- `year`: season to simulate.
- `weighted_mean`: use of weighted mean in simulation (boolean; default=False; False is recommended for early in the season while True is recommended for later games).
- `n_simulations`: number of simulations for each game (default=1000).

Attributes:
- `df_simulated_season`: data frame of all played and simulated games in season.
- `df_final_win_predictions_conf`: data frame of predicted wins.
- `df_NFC`: data frame of predicted wins (NFC only).
- `df_AFC`: data frame of predicted wins (AFC only).
- `df_NFC_div`: data frame of predicted wins in NFC sortted by division.
- `df_AFC_div`: data frame of predicted wins in AFC sortted by division

Example:

```
from nfl_predictions import nfl_season_simulation

# simulate season
simulated_season = nfl_season_simulation(year=2019, 
                                         weighted_mean=False, 
                                         n_simulations=1000)

# get simulated season
df_entire_season = simulated_season.df_simulated_season

# get final win predictions
df_standings = simulated_season.df_final_win_predictions_conf

# get NFC
df_NFC_standings = simulated_season.df_NFC

# get AFC
df_AFC_standings = simulated_season.df_AFC

# get NFC sorted by division and wins
df_NFC_division_standings = simulated_season.df_NFC_div

# get AFC sorted by division and wins
df_AFC_division_standings = simulated_season.df_AFC_div
```

---

## nfl_postseason_probabilities

Arguments:
- `year`: season to simulate.
- `n_simulations`: number of seasons to simulate.
- `weighted_mean`: use of weighted mean in simulation (boolean; default=False; False is recommended for early in the season while True is recommended for later games).
- `weighted_mean_super_bowl`: use of weighted mean in simulation of super bowl (boolean; default=True).

Returns a data frame with columns for probability of postseason, conference champion, and super bowl champion.

Example:

```
from nfl_predictions import nfl_postseason_probabilities

# get probabilities
df_postseason_probabilities = nfl_postseason_probabilities(year=2019, 
                                                           n_simulations=100, 
                                                           weighted_mean=False,
                                                           weighted_mean_super_bowl=True)
```

---
