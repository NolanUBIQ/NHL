
import pandas as pd

goalie = pd.read_csv("goalie-gsax-output.csv")
prices = pd.read_csv("price-export-2024.csv")

goalie = goalie.sort_values(by='toi', ascending=False).drop_duplicates(subset=["gid", "Goalie Team"], keep='first')

goalie["isHome"] = goalie.apply(lambda x: 1 if x["Home Team"] == x["Goalie Team"] else 0, axis=1)

homeGoalie = goalie[goalie["isHome"] == 1][["gid", "Goalie Team", "playerName"]]
awayGoalie = goalie[goalie["isHome"] == 0][["gid", "Goalie Team", "playerName"]]

prices = prices.merge(homeGoalie, left_on=["gid_nhl", "Home Team"], right_on=["gid","Goalie Team"], how="left").rename({"playerName":"home_goalie"}, axis=1)

prices = prices.merge(awayGoalie, left_on=["gid_nhl", "Away Team"], right_on=["gid","Goalie Team"], how="left").rename({"playerName":"away_goalie"}, axis=1)

prices["gametime"] = "7:00:00"

prices=prices[["Date", "gid_nhl", "Home Team", "Away Team", "gametime", 'close_away_ml', 'close_home_ml', 'close_total_line', 'close_total_over', 'close_total_under', 'home_goalie', 'away_goalie']]

prices.columns = ['date', 'game-ID', 'home_team_abv', 'away_team_abv', 'gametime',
       'close_away_ml', 'close_home_ml', 'close_total_line',
       'close_total_over', 'close_total_under', 'home_goalie', 'away_goalie']

prices.to_csv("market-implied-data-file.csv", index=False)

