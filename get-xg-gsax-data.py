from time import sleep
import pandas as pd
import requests
from datetime import datetime, timedelta
import pytz
from io import StringIO
import os
pd.options.mode.chained_assignment = None
from scrapeGoalSituations import get_goals

import sys

# Add script's directory to the system path
sys.path.append(os.path.dirname(__file__))




def get_goalie_stats(df, gid):
    df = df[df["position"] == "G"][["gameId","playerId","playerName","OnIce_A_goals","team", "OnIce_A_xGoals", "situation", "I_A_iceTime"]]
    df = df[df["situation"] == "all"]
    df["toi"] = ((df["I_A_iceTime"]//60).astype(int)).astype(str) + ":" + ((df["I_A_iceTime"]%60).astype(int)).astype(str)
    df["map"] = df["gameId"].astype(str) + df["playerName"].astype(str)
    df["Goals Saved Above Expected"] = df["OnIce_A_xGoals"] - df["OnIce_A_goals"]
    df['proj_gsaX'] = (df["Goals Saved Above Expected"].astype(float)*0.9138)  - 0.1685
    df["date"] = ''
    df["gsaX"] = df["Goals Saved Above Expected"]
    df.reset_index(drop=True, inplace=True)
    home = df[df["team"] == df.at[0,"team"]]
    home["isHome"] = True
    away = df[df["team"] != df.at[0,"team"]]
    away["isHome"] = False
    df = pd.concat([home,away],axis=0)
    df = df[["date", "gameId", 'playerId', 'playerName', 'OnIce_A_goals', 'toi', 'isHome', 'team', 'gsaX', 'map', 'Goals Saved Above Expected', 'proj_gsaX']]
    return df
    
def get_game_data_for_date(date):
    date = date.strftime("%Y-%m-%d")
    url = f"https://api-web.nhle.com/v1/schedule/{date}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data for {date}: Status code {response.status_code}")
        return None

def request_game_data(start_date, end_date):
    all_games_data = []
    existing_game_ids = set()  # Using a set for faster look-up times

    # Now convert the string to a datetime object
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    current_date = start_date
    while current_date <= end_date:
        day_data = get_game_data_for_date(current_date)
        if day_data:
            # Extract the games for the current date
            game_weeks = day_data.get('gameWeek', [])
            for game_week in game_weeks:
                games = game_week.get('games', [])
                for game in games:
                    game_id = game.get('id')
                    if game_id and game_id not in existing_game_ids:
                        all_games_data.append(game)
                        existing_game_ids.add(game_id)
        current_date = current_date + timedelta(days=1)

    return all_games_data

def parse_game_info(game):
    # Parse the startTimeUTC string into a datetime object
    utc_time = datetime.strptime(game['startTimeUTC'], '%Y-%m-%dT%H:%M:%SZ')
    
    # Convert UTC time to Eastern Time
    utc_zone = pytz.utc
    eastern_zone = pytz.timezone('US/Eastern')
    
    utc_time = utc_zone.localize(utc_time)  # Localize the time as UTC
    eastern_time = utc_time.astimezone(eastern_zone)  # Convert to Eastern Time
    
    # Format the Eastern Time as "YYYY-MM-DD"
    formatted_date = eastern_time.strftime('%Y-%m-%d')
    
    return [
        game['id'],
        game['gameType'],
        game['neutralSite'],
        formatted_date,  # Use the formatted Eastern Time date string
        game['awayTeam']['abbrev'],
        game['awayTeam']['score'],
        game['homeTeam']['abbrev'],
        game['homeTeam']['score']
    ]
    
def get_xg_stats(df, gid):
        fourOnFive = df[(df["situation"] == "4on5") & (~df["position"].isin(["pairing", "line", "Team Level", "G"]))].groupby("team").agg({'I_F_flurryAdjustedxGoals':"sum"}).reset_index().rename({"I_F_flurryAdjustedxGoals":"4on5 Flurry Xgoals"}, axis=1)
        fiveOnFive = df[(df["situation"] == "5on5") & (~df["position"].isin(["pairing", "line", "Team Level", "G"]))].groupby("team").agg({'I_F_flurryAdjustedxGoals':"sum"}).reset_index().rename({"I_F_flurryAdjustedxGoals":"5on5 Flurry Xgoals"}, axis=1)
        fiveOnFour = df[(df["situation"] == "5on4") & (~df["position"].isin(["pairing", "line", "Team Level", "G"]))].groupby("team").agg({'I_F_flurryAdjustedxGoals':"sum"}).reset_index().rename({"I_F_flurryAdjustedxGoals":"5on4 Flurry Xgoals"}, axis=1)
        fiveOnFiveAg = df[df["situation"] == "5on5"].groupby("team").agg({"I_F_goals":"sum"}).reset_index().rename({"I_F_goals":"5on5 goals"}, axis=1)
        fourOnFiveAg = df[df["situation"] == "4on5"].groupby("team").agg({"I_F_goals":"sum"}).reset_index().rename({"I_F_goals":"4on5 goals"}, axis=1)
        fiveOnFourAg = df[df["situation"] == "5on4"].groupby("team").agg({"I_F_goals":"sum"}).reset_index().rename({"I_F_goals":"5on4 goals"}, axis=1)
        teamA = fiveOnFive.merge(fiveOnFour, how="left", on="team").merge(fourOnFive, how="left", on="team").head(1).reset_index(drop=True)
        teamAopp = fiveOnFive.merge(fiveOnFour, how="left", on="team").merge(fourOnFive, how="left", on="team").tail(1).rename({'team':"opp_team", '5on5 Flurry Xgoals':"opp_5on5 Flurry Xgoals", '5on4 Flurry Xgoals':"opp_5on4 Flurry Xgoals",'4on5 Flurry Xgoals':"opp_4on5 Flurry Xgoals"}, axis=1).reset_index(drop=True)
        teamB = fiveOnFive.merge(fiveOnFour, how="left", on="team").merge(fourOnFive, how="left", on="team").tail(1).reset_index(drop=True)
        teamBopp = fiveOnFive.merge(fiveOnFour, how="left", on="team").merge(fourOnFive, how="left", on="team").head(1).rename({'team':"opp_team", '5on5 Flurry Xgoals':"opp_5on5 Flurry Xgoals", '5on4 Flurry Xgoals':"opp_5on4 Flurry Xgoals",'4on5 Flurry Xgoals':"opp_4on5 Flurry Xgoals"}, axis=1).reset_index(drop=True)
        teamAag = fiveOnFiveAg.merge(fiveOnFourAg, how="left", on="team").merge(fourOnFiveAg, how="left", on="team").head(1).reset_index(drop=True).drop("team", axis=1)
        teamAoppAg = fiveOnFiveAg.merge(fiveOnFourAg, how="left", on="team").merge(fourOnFiveAg, how="left", on="team").tail(1).rename({'5on5 goals':"opp_5on5 goals", '4on5 goals':"opp_4on5 goals",'5on4 goals':"opp_5on4 goals"}, axis=1).reset_index(drop=True).drop("team", axis=1)
        teamBag = fiveOnFiveAg.merge(fiveOnFourAg, how="left", on="team").merge(fourOnFiveAg, how="left", on="team").tail(1).reset_index(drop=True).drop("team", axis=1)
        teamBoppAg = fiveOnFiveAg.merge(fiveOnFourAg, how="left", on="team").merge(fourOnFiveAg, how="left", on="team").head(1).rename({'5on5 goals':"opp_5on5 goals", '4on5 goals':"opp_4on5 goals",'5on4 goals':"opp_5on4 goals"}, axis=1).reset_index(drop=True).drop("team", axis=1)
        teamAgame = pd.concat([teamA, teamAopp, teamAag, teamAoppAg], axis=1)
        teamBgame = pd.concat([teamB, teamBopp, teamBag, teamBoppAg], axis=1)
        teamBgame = teamBgame[['team', 'opp_team', '5on5 Flurry Xgoals',"5on5 goals", '5on4 Flurry Xgoals',"5on4 goals",
                '4on5 Flurry Xgoals',"4on5 goals", 'opp_5on5 Flurry Xgoals',"opp_5on5 goals",
                'opp_5on4 Flurry Xgoals',"opp_5on4 goals", 'opp_4on5 Flurry Xgoals',"opp_4on5 goals"]]
        teamAgame = teamAgame[['team', 'opp_team', '5on5 Flurry Xgoals',"5on5 goals", '5on4 Flurry Xgoals',"5on4 goals",
                '4on5 Flurry Xgoals',"4on5 goals", 'opp_5on5 Flurry Xgoals',"opp_5on5 goals",
                'opp_5on4 Flurry Xgoals',"opp_5on4 goals", 'opp_4on5 Flurry Xgoals',"opp_4on5 goals"]]
        done = pd.concat([teamAgame, teamBgame], axis=0).reset_index(drop=True)
        done["gid"] = gid
        return done

team_mapping_NewAPI = {
    'TBL': 'Tampa Bay Lightning',
    'PIT': 'Pittsburgh Penguins',
    'VGK': 'Vegas Golden Knights',
    'CAR': 'Carolina Hurricanes',
    'TOR': 'Toronto Maple Leafs',
    'BOS': 'Boston Bruins',
    'CGY': 'Calgary Flames',
    'LAK': 'Los Angeles Kings',
    'VAN': 'Vancouver Canucks',
    'BUF': 'Buffalo Sabres',
    'CBJ': 'Columbus Blue Jackets',
    'NJD': 'New Jersey Devils',
    'DAL': 'Dallas Stars',
    'MIN': 'Minnesota Wild',
    'NSH': 'Nashville Predators',
    'SJS': 'San Jose Sharks',
    'WSH': 'Washington Capitals',
    'OTT': 'Ottawa Senators',
    'WPG': 'Winnipeg Jets',
    'DET': 'Detroit Red Wings',
    'MTL': 'Montréal Canadiens',
    'NYI': 'New York Islanders',
    'STL': 'St. Louis Blues',
    'EDM': 'Edmonton Oilers',
    'ANA': 'Anaheim Ducks',
    'NYR': 'New York Rangers',
    'PHI': 'Philadelphia Flyers',
    'SEA': 'Seattle Kraken',
    'FLA': 'Florida Panthers',
    'COL': 'Colorado Avalanche',
    'ARI': 'Arizona Coyotes',
    'CHI': 'Chicago Blackhawks',
    "UTA": 'Arizona Coyotes',
}

# Get today and tommorows dates
today = datetime.now().date()
yday = today + timedelta(days=-1)
# Format the dates as 'YYYY-MM-DD'
formatted_today = today.strftime('%Y-%m-%d')
formatted_yday = yday.strftime('%Y-%m-%d')
gameData = request_game_data(str(yday), str(today))
#gameData = request_game_data("2024-10-03", str(today))
rowData = []
for i in gameData:
    if i["gameState"] in ['FINAL', 'OFF']:
        rowData.append(parse_game_info(i))
games = pd.DataFrame(rowData, columns=["gid", "Season Type", "Neutral", "Date", "Away Team", "Away Score", "Home Team", "Home Score"])
games = games[games["Season Type"].isin([2,3])]
goalieStats = pd.DataFrame(columns=['date', 'gameId', 'playerId', 'playerName', 'OnIce_A_goals', 'toi',
       'isHome', 'team', 'gsaX', 'map', 'Goals Saved Above Expected',
       'proj_gsaX'])
xgStatsRaw = pd.DataFrame(columns=['team', 'opp_team', '5on5 Flurry Xgoals', '5on5 goals',
       '5on4 Flurry Xgoals', '5on4 goals', '4on5 Flurry Xgoals', '4on5 goals',
       'opp_5on5 Flurry Xgoals', 'opp_5on5 goals', 'opp_5on4 Flurry Xgoals',
       'opp_5on4 goals', 'opp_4on5 Flurry Xgoals', 'opp_4on5 goals', 'gid'])
nhlRef = pd.DataFrame(columns=['EV_home_goals', 'PP_home_goals', 'SH_home_goals', 'gid',
       'EV_away_goals', 'PP_away_goals', 'SH_away_goals'])

count = 0

folder_path = r"C:\Users\NolanNicholls\Documents\NHL\2024\scott_scripts"

if len(games) > 0:

    for i in games["gid"].unique():
        try:
            url = "https://moneypuck.com/moneypuck/playerData/games/20242025/{}.csv".format(i)
            response = requests.request("GET", url)
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            tempGoalie = get_goalie_stats(df, i)
            tempXg = get_xg_stats(df, i)
            goalieStats = pd.concat([goalieStats, tempGoalie], axis=0)
            xgStatsRaw = pd.concat([xgStatsRaw, tempXg], axis=0)

            tempGames = games[games["gid"] == i].head(1)
            homeTeam = tempGames["Home Team"].unique()[0]
            awayTeam = tempGames["Away Team"].unique()[0]
            if homeTeam == "VGK":
                homeTeam = "VEG"
            if awayTeam == "VGK":
                awayTeam = "VEG"
            tempGoals = get_goals(homeTeam, awayTeam,tempGames["Date"].unique()[0], i)
            nhlRef = pd.concat([nhlRef, tempGoals], axis=0)
            sleep(2)
            count += 1
        except Exception as e:
            print(e)
            print("error on gid ", i)


            
    goalieGsax = goalieStats.merge(games, left_on="gameId", right_on="gid", how="left")
    goalieGsax = goalieGsax[['gid', 'Season Type', 'Neutral', 'Date', 'Away Team','Away Score', 'Home Team', 'Home Score', 'playerId', 'playerName', 'toi', 'team','gsaX','proj_gsaX']]
    goalieGsax["isHome"] = goalieGsax.apply(lambda x: True if x["team"] == x["Home Team"] else False, axis=1)
    goalieGsax["Home Team Long"] = goalieGsax["Home Team"].apply(lambda x: team_mapping_NewAPI[x])
    goalieGsax["Away Team Long"] = goalieGsax["Away Team"].apply(lambda x: team_mapping_NewAPI[x])
    goalieGsax["Goalie Team Long"] = goalieGsax["team"].apply(lambda x: team_mapping_NewAPI[x])
    goalieGsax = goalieGsax.rename({"team":"Goalie Team"}, axis=1)
    goalieGsax = goalieGsax[['gid', 'Season Type', 'Neutral', 'Date', 'Away Team',"Away Team Long", 'Home Team',"Home Team Long",'Away Score', 'Home Score',"Goalie Team","Goalie Team Long",'playerId', 'playerName', 'toi','gsaX','proj_gsaX']]
    xgStats = xgStatsRaw.merge(games, on="gid", how="left")
    xgStats = xgStats.rename({"team":"Team"}, axis=1)
    xgStats["isHome"] = xgStats.apply(lambda x: True if x["Team"] == x["Home Team"] else False, axis=1)
    xgStats["Home Team Long"] = xgStats["Home Team"].apply(lambda x: team_mapping_NewAPI[x])
    xgStats["Away Team Long"] = xgStats["Away Team"].apply(lambda x: team_mapping_NewAPI[x])
    xgStats["Team Long"] = xgStats["Team"].apply(lambda x: team_mapping_NewAPI[x])

    xgStats = xgStats[['gid','Season Type', 'Neutral', 'Date','Team','Team Long', 'isHome', 'Away Team', 'Away Team Long',
            'Home Team', 'Home Team Long', 'Away Score', 'Home Score', '5on5 Flurry Xgoals',"5on5 goals", '5on4 Flurry Xgoals',"5on4 goals",
                    '4on5 Flurry Xgoals',"4on5 goals", 'opp_5on5 Flurry Xgoals',"opp_5on5 goals",
                    'opp_5on4 Flurry Xgoals',"opp_5on4 goals", 'opp_4on5 Flurry Xgoals',"opp_4on5 goals"
        ]]

    xgStats = xgStats[xgStats["isHome"] == True]

    xgStats = xgStats.merge(nhlRef, on="gid", how="left")


    goalie_gsax_file = os.path.join(folder_path, "goalie-gsax-output.csv")
    xg_stats_file = os.path.join(folder_path, "xg-stats-output.csv")

    if not os.path.exists(goalie_gsax_file):
        goalieGsax.to_csv(goalie_gsax_file, index=False)
    else:
        prevGoalieGsax = pd.read_csv(goalie_gsax_file)
        goalieGsax = pd.concat([prevGoalieGsax, goalieGsax], axis =0)
        goalieGsax = goalieGsax.drop_duplicates(subset=["gid", "Date", "playerName"])
        goalieGsax.to_csv(goalie_gsax_file, index=False)

    if not os.path.exists(xg_stats_file):
        xgStats.to_csv(xg_stats_file, index=False)
    else:
        prevXgStats = pd.read_csv(xg_stats_file)
        xgStats = pd.concat([prevXgStats, xgStats], axis =0)
        xgStats = xgStats.drop_duplicates(subset=["gid", "Date", "Team"])
        xgStats.to_csv(xg_stats_file, index=False)
else:
    pass    
