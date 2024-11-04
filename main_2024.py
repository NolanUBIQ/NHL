import datetime
from datetime import datetime, timedelta
import pymc as pm

from collections import defaultdict
import requests
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe
import ast

import math
from math import sqrt
from math import factorial
from scipy.integrate import dblquad
from math import pi
from itertools import combinations
from scipy.stats import norm
import pytz
from dateutil import parser
import os
import requests
import json
import numpy as np
import math
import pickle


# Your Airtable details
API_KEY = 'pat86JThLoARQ9dQs.4293315060988ade2101ee60d327d6642f426a5e7c7e206c12bea6654955d7d6'
BASE_ID = 'appAtWR2HGzOAeklT'
TABLE_NAME = 'team_goalies'
ENDPOINT = f'https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}'

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Make the API request
response = requests.get(ENDPOINT, headers=headers)
dict_starting_goalies = {}

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    records = data['records']

    # Process the records to build the desired dictionary
    for record in records:
        fields = record.get('fields', {})
        team = fields.get('Team')
        goalie = fields.get('Player Name')

        # Ensure both team and goalie are present before processing
        if team and goalie:
            if team in dict_starting_goalies:
                dict_starting_goalies[team].append(goalie)
            else:
                dict_starting_goalies[team] = goalie

    print(dict_starting_goalies)
else:
    print("Failed to fetch data from Airtable:", response.content)



# Set up the connection to Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name("C:/Users/NolanNicholls/Documents/NHL/2024/credentials/smooth-topic-379501-521289118ec5.json", scope)
client = gspread.authorize(creds)

# Open the Google Sheet using its name
sheet = client.open("NHL_Dashboard_2024").worksheet("goalie-depth")





df_goalies = pd.DataFrame.from_dict(dict_starting_goalies, orient='index')
df_goalies.columns=["goalie_1", "goalie_2", "goalie_3"]
# Update the Google Sheet with the DataFrame
set_with_dataframe(sheet, df_goalies, row=2, include_column_header=False,include_index=True)

neutral_games = ["2024020001", "2024020002", "2024020168", "2024020174"]

team_mapping = {
    'PIT': 'Pittsburgh Penguins',
    'T.B': 'Tampa Bay Lightning',
    'SEA': 'Seattle Kraken',
    'VGK': 'Vegas Golden Knights',
    'MTL': 'Montréal Canadiens',
    'TOR': 'Toronto Maple Leafs',
    'NYR': 'New York Rangers',
    'WSH': 'Washington Capitals',
    'CHI': 'Chicago Blackhawks',
    'COL': 'Colorado Avalanche',
    'EDM': 'Edmonton Oilers',
    'VAN': 'Vancouver Canucks',
    'BUF': 'Buffalo Sabres',
    'OTT': 'Ottawa Senators',
    'DET': 'Detroit Red Wings',
    'FLA': 'Florida Panthers',
    'DAL': 'Dallas Stars',
    'CAR': 'Carolina Hurricanes',
    'NYI': 'New York Islanders',
    'UTA': 'Utah Hockey Club',
    'CBJ': 'Columbus Blue Jackets',
    'NSH': 'Nashville Predators',
    'ANA': 'Anaheim Ducks',
    'WPG': 'Winnipeg Jets',
    'L.A': 'Los Angeles Kings',
    'N.J': 'New Jersey Devils',
    'PHI': 'Philadelphia Flyers',
    'MIN': 'Minnesota Wild',
    'BOS': 'Boston Bruins',
    'STL': 'St. Louis Blues',
    'CGY': 'Calgary Flames',
    'S.J': 'San Jose Sharks'
}
team_dict_rev = {
    'Anaheim Ducks': 'ANA',
    'Utah Hockey Club': 'UTA',
    'Boston Bruins': 'BOS',
    'Buffalo Sabres': 'BUF',
    'Calgary Flames': 'CGY',
    'Carolina Hurricanes': 'CAR',
    'Chicago Blackhawks': 'CHI',
    'Colorado Avalanche': 'COL',
    'Columbus Blue Jackets': 'CBJ',
    'Dallas Stars': 'DAL',
    'Detroit Red Wings': 'DET',
    'Edmonton Oilers': 'EDM',
    'Florida Panthers': 'FLA',
    'Los Angeles Kings': 'L.A',
    'Minnesota Wild': 'MIN',
    'Montréal Canadiens': 'MTL',
    'Nashville Predators': 'NSH',
    'New Jersey Devils': 'N.J',
    'New York Islanders': 'NYI',
    'New York Rangers': 'NYR',
    'Ottawa Senators': 'OTT',
    'Philadelphia Flyers': 'PHI',
    'Pittsburgh Penguins': 'PIT',
    'San Jose Sharks': 'S.J',
    'Seattle Kraken': 'SEA',
    'St. Louis Blues': 'STL',
    'Tampa Bay Lightning': 'T.B',
    'Toronto Maple Leafs': 'TOR',
    'Vancouver Canucks': 'VAN',
    'Vegas Golden Knights': 'VGK',
    'Washington Capitals': 'WSH',
    'Winnipeg Jets': 'WPG'
}


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
    'UTA': 'Utah Hockey Club',
    'CHI': 'Chicago Blackhawks'
}

nhl_timezones = {
    "NSH": "CT",
    "SJS": "PT",
    "NYR": "ET",
    "LAK": "PT",
    "WSH": "ET",
    "CAR": "ET",
    "MTL": "ET",
    "COL": "MT",
    "ANA": "PT",
    "EDM": "MT",
    "PIT": "ET",
    "PHI": "ET",
    "BUF": "ET",
    "NYI": "ET",
    "TOR": "ET",
    "MIN": "CT",
    "CAL": "MT",
    "VGK": "PT",
    "CBJ": "ET",
    "DET": "ET",
    "WPG": "CT",
    "BOS": "ET",
    "NJD": "ET",
    "STL": "CT",
    "DAL": "CT",
    "SEA": "PT",
    "OTT": "ET",
    "TAM": "ET",
    "FLA": "ET",
    "CHI": "CT",
    "VAN": "PT",
    "UTA": "MT"
}

team_dict_rev_fortimezone = {
    'Anaheim Ducks': 'ANA',
    'Utah Hockey Club': 'UTA',
    'Boston Bruins': 'BOS',
    'Buffalo Sabres': 'BUF',
    'Calgary Flames': 'CAL',
    'Carolina Hurricanes': 'CAR',
    'Chicago Blackhawks': 'CHI',
    'Colorado Avalanche': 'COL',
    'Columbus Blue Jackets': 'CBJ',
    'Dallas Stars': 'DAL',
    'Detroit Red Wings': 'DET',
    'Edmonton Oilers': 'EDM',
    'Florida Panthers': 'FLA',
    'Los Angeles Kings': 'LAK',
    'Minnesota Wild': 'MIN',
    'Montréal Canadiens': 'MTL',
    'Nashville Predators': 'NSH',
    'New Jersey Devils': 'NJD',
    'New York Islanders': 'NYI',
    'New York Rangers': 'NYR',
    'Ottawa Senators': 'OTT',
    'Philadelphia Flyers': 'PHI',
    'Pittsburgh Penguins': 'PIT',
    'San Jose Sharks': 'SJS',
    'Seattle Kraken': 'SEA',
    'St. Louis Blues': 'STL',
    'Tampa Bay Lightning': 'TAM',
    'Toronto Maple Leafs': 'TOR',
    'Vancouver Canucks': 'VAN',
    'Vegas Golden Knights': 'VGK',
    'Washington Capitals': 'WSH',
    'Winnipeg Jets': 'WPG'
}

data = {
    'team': [
        'Anaheim Ducks', 'Utah Hockey Club', 'Boston Bruins', 'Buffalo Sabres',
        'Calgary Flames', 'Carolina Hurricanes', 'Chicago Blackhawks', 'Colorado Avalanche',
        'Columbus Blue Jackets', 'Dallas Stars', 'Detroit Red Wings', 'Edmonton Oilers',
        'Florida Panthers', 'Los Angeles Kings', 'Minnesota Wild', 'Montréal Canadiens',
        'Nashville Predators', 'New Jersey Devils', 'New York Islanders', 'New York Rangers',
        'Ottawa Senators', 'Philadelphia Flyers', 'Pittsburgh Penguins', 'San Jose Sharks',
        'Seattle Kraken', 'St. Louis Blues', 'Tampa Bay Lightning', 'Toronto Maple Leafs',
        'Vancouver Canucks', 'Vegas Golden Knights', 'Washington Capitals', 'Winnipeg Jets'
    ],
    'i': list(range(32))
}

num_teams=32

df = pd.DataFrame(data)

def get_game_data_for_date(date):
    url = f"https://api-web.nhle.com/v1/schedule/{date}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data for {date}: Status code {response.status_code}")
        return None

def increment_date_by_one_day(date_string):
    date_format = "%Y-%m-%d"
    current_date = datetime.strptime(date_string, date_format)
    next_date = current_date + timedelta(days=1)
    return next_date.strftime(date_format)

from datetime import timezone, timedelta

# Helper function to convert UTC time to Eastern Time
def convert_utc_to_eastern(start_time_utc):
    # Define the UTC and Eastern Time offsets
    utc_offset = timedelta(hours=0)
    eastern_offset_standard_time = timedelta(hours=-5)  # EST
    eastern_offset_daylight_saving_time = timedelta(hours=-4)  # EDT
    
    # Parse the start time as a datetime object
    start_time_obj = datetime.strptime(start_time_utc, "%Y-%m-%dT%H:%M:%SZ")
    
    # Assuming daylight saving changes are the same every year
    # This needs to be updated based on actual daylight saving dates
    daylight_saving_start = datetime(start_time_obj.year, 3, 14)  # Second Sunday of March
    daylight_saving_end = datetime(start_time_obj.year, 11, 7)    # First Sunday of November
    
    # Determine if the date is in daylight saving time
    is_daylight_saving = daylight_saving_start <= start_time_obj.replace(tzinfo=None) < daylight_saving_end
    
    # Convert the UTC start time to Eastern Time
    if is_daylight_saving:
        return (start_time_obj.replace(tzinfo=timezone(utc_offset)) + eastern_offset_daylight_saving_time).strftime("%Y-%m-%d")
    else:
        return (start_time_obj.replace(tzinfo=timezone(utc_offset)) + eastern_offset_standard_time).strftime("%Y-%m-%d")


def request_game_data(start_date, end_date):


    def get_game_date_from_start_time(start_time):
        return start_time.split("T")[0]  # Assuming the start time is always in a consistent format
    
    all_games_data = []
    existing_game_ids = set()  # Using a set for faster look-up times
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    # Loop over each day from the season start date to the end date
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
                    game_date = convert_utc_to_eastern(game['startTimeUTC'])

                    # Convert the game date string to a datetime object for comparison
                    game_date_obj = datetime.strptime(game_date, "%Y-%m-%d")
                    # Check if the game ID is new and the game date is on or before the end date
                    if game_id not in existing_game_ids and game_date_obj <= end_date_obj:
                        all_games_data.append(game)
                        existing_game_ids.add(game_id)  # Add the game ID to the set of known game IDs
        current_date = increment_date_by_one_day(current_date)

    
    return all_games_data


def extract_outcome_details(game_outcome):
    if isinstance(game_outcome, dict) and 'lastPeriodType' in game_outcome:
        last_period_type = game_outcome['lastPeriodType']
        went_to_overtime = last_period_type == 'OT'
        went_to_shoot_out = last_period_type == 'SO'
    else:
        went_to_overtime = False  # Default to False if the entry is not a dictionary
        went_to_shoot_out = False  # Default to False if the entry is not a dictionary
    return went_to_overtime, went_to_shoot_out

# A function to extract the relevant data from the schedule
# and return it as a pandas dataframe
def extract_game_data(all_games_data):

    df_games = pd.DataFrame(all_games_data)
    df_games = df_games.join(pd.json_normalize(df_games.pop('homeTeam')).add_prefix('homeTeam_'))
    df_games = df_games.join(pd.json_normalize(df_games.pop('awayTeam')).add_prefix('awayTeam_'))

    df_games[['went_to_overtime', 'went_to_shoot_out']] = pd.DataFrame(df_games['gameOutcome'].apply(extract_outcome_details).tolist(), index=df_games.index)

    df_selected = df_games.rename(columns={
    'id': 'game-ID',
    'gameType': 'game_type',
    'homeTeam_abbrev': 'home_team',
    'homeTeam_score': 'home_team_reg_score',  # Assuming this is the regular time score
    'awayTeam_abbrev': 'away_team',
    'awayTeam_score': 'away_team_reg_score',  # Assuming this is the regular time score
    'startTimeUTC': 'game_time'})

    eastern = pytz.timezone('US/Eastern')

    df_selected['game_time'] = pd.to_datetime(df_selected['game_time'])

    df_selected['game_time'] = df_selected['game_time'].apply(lambda x: x.astimezone(eastern))

    # If you want to remove the timezone information after conversion, use .tz_localize(None)
    df_selected['game_time'] = df_selected['game_time'].dt.tz_localize(None)

    # If 'home_team_fin_score' and 'away_team_fin_score' are not explicitly provided, we can assume they are the same as the regular score for now
    df_selected['home_team_fin_score'] = df_selected['home_team_reg_score']
    df_selected['away_team_fin_score'] = df_selected['away_team_reg_score']

    # Convert 'game_time' to just the date if it's currently a timestamp



    df_selected['date'] = pd.to_datetime(df_selected['game_time']).dt.date

    final_columns = [
    'date',
    'game-ID',
    'season',
    'game_type',
    'home_team',
    'home_team_reg_score',
    'home_team_fin_score',
    'away_team',
    'away_team_reg_score',
    'away_team_fin_score',
    'went_to_overtime',  # Optional: Remove this line if you do not want this column
    'went_to_shoot_out',
    'game_time']


    df_final = df_selected.reindex(columns=final_columns)
    df_final['winner'] = df_final.apply(lambda row: 'home' if row['home_team_fin_score'] > row['away_team_fin_score'] else 'away', axis=1)
    
    def adjust_score(row):
        if row['went_to_shoot_out'] or row['went_to_overtime']:
            if row['winner'] == 'home':
                row['home_team_reg_score'] -= 1
            else:
                row['away_team_reg_score'] -= 1
        return row

    df_final = df_final.apply(adjust_score, axis=1)

    df_final["game_type"]=df_final["game_type"].replace({2:"R", 3:"P"})

    df_final['gametime'] = pd.to_datetime(df_final['game_time']).dt.strftime('%H:%M')
    
    df_final["home_team"] = df_final["home_team"].map(team_mapping_NewAPI)
    df_final["away_team"] = df_final["away_team"].map(team_mapping_NewAPI)
    df_final["home_team_reg_score"] = df_final["home_team_reg_score"].astype('Int64').fillna(0)
    df_final["away_team_reg_score"] = df_final["away_team_reg_score"].astype('Int64').fillna(0)
    df_final["home_team_fin_score"] = df_final["home_team_fin_score"].astype('Int64').fillna(0)
    df_final["away_team_fin_score"] = df_final["away_team_fin_score"].astype('Int64').fillna(0)




    completed_games = df_final[['date', 'game-ID', 'season', 'game_type', 'home_team',
       'home_team_reg_score', 'home_team_fin_score', 'away_team',
       'away_team_reg_score', 'away_team_fin_score', 'went_to_shoot_out',
       'game_time', 'gametime']]

    return completed_games


def game_outcome_to_bernoulli_data(row):
    if row['home_team_fin_score'] > row['away_team_fin_score']:
        #   return row['home_team'] == team_pairs_heads_dict[(row['home_team'], row['away_team'])]
        return True

    #     return row['away_team'] == team_pairs_heads_dict[(row['home_team'], row['away_team'])]
    return False




# Modify the data to include team and pair integer labels
def add_team_data_labels(game_data):
    game_data = game_data.merge(teams, left_on='home_team', right_on='team', how='left')
    game_data = game_data.rename(columns={'i': 'i_home'}).drop('team', axis=1)
    game_data = game_data.merge(teams, left_on='away_team', right_on='team', how='left')
    game_data = game_data.rename(columns={'i': 'i_away'}).drop('team', axis=1)
    game_data['i_pair'] = game_data.apply(lambda row: team_pairs_dict[(row['home_team'], row['away_team'])], axis=1)
    game_data['i_pair_winner'] = game_data.apply(game_outcome_to_bernoulli_data, axis=1)

    return game_data


def define_time_advantages(game_df, timezones):
    ## helper to apply overrides ##
   
    ## copy frame ##
    temp_df = game_df.copy()
    peak_time = '17:00'
    ## add time zones ##
    temp_df['home_tz'] = temp_df['home_team'].replace(team_dict_rev_fortimezone).replace(timezones).fillna('ET')
    temp_df['away_tz'] = temp_df['away_team'].replace(team_dict_rev_fortimezone).replace(timezones).fillna('ET')
    ## apply overrides ##
   
    ## define optimals in ET ##
    temp_df['home_optimal_in_et'] = pd.Timestamp(peak_time)
    temp_df['away_optimal_in_et'] = pd.Timestamp(peak_time)
    ## home ##
    temp_df['home_optimal_in_et'] = np.where(
        temp_df['home_tz'] == 'ET',
        temp_df['home_optimal_in_et'].dt.time,
        np.where(
            temp_df['home_tz'] == 'CT',
            (temp_df['home_optimal_in_et'] + pd.Timedelta(hours=1)).dt.time,
            np.where(
                temp_df['home_tz'] == 'MT',
                (temp_df['home_optimal_in_et'] + pd.Timedelta(hours=2)).dt.time,
                np.where(
                    temp_df['home_tz'] == 'PT',
                    (temp_df['home_optimal_in_et'] + pd.Timedelta(hours=3)).dt.time,
                    temp_df['home_optimal_in_et'].dt.time
                )
            )
        )
    )
    ## away ##
    temp_df['away_optimal_in_et'] = np.where(
        temp_df['away_tz'] == 'ET',
        temp_df['away_optimal_in_et'].dt.time,
        np.where(
            temp_df['away_tz'] == 'CT',
            (temp_df['away_optimal_in_et'] + pd.Timedelta(hours=1)).dt.time,
            np.where(
                temp_df['away_tz'] == 'MT',
                (temp_df['away_optimal_in_et'] + pd.Timedelta(hours=2)).dt.time,
                np.where(
                    temp_df['away_tz'] == 'PT',
                    (temp_df['away_optimal_in_et'] + pd.Timedelta(hours=3)).dt.time,
                    temp_df['away_optimal_in_et'].dt.time
                )
            )
        )
    )
    ## get kickoff ##
    temp_df['gametimestamp'] = pd.to_datetime(temp_df['gametime']).dt.time

    ## define advantage ##
    temp_df['home_time_advantage'] = np.round(
        np.absolute(
            (
                pd.to_datetime(temp_df['gametimestamp'], format='%H:%M:%S') -
                pd.to_datetime(temp_df['away_optimal_in_et'], format='%H:%M:%S')
            ) / np.timedelta64(1, 'h')
        ) -
        np.absolute(
            (
                pd.to_datetime(temp_df['gametimestamp'], format='%H:%M:%S') -
                pd.to_datetime(temp_df['home_optimal_in_et'], format='%H:%M:%S')
            ) / np.timedelta64(1, 'h')
        )
    )
    return temp_df['home_time_advantage'].fillna(0)

def poisson_gamma(mu,  n):
   
    poisson_samples = np.random.poisson(lam=mu, size=n)
    
    return poisson_samples


def poisson_draws(mu_spec, n):
    """
    Generate n draws from a Poisson distribution with the specified mean (mu_spec).
    
    Parameters:
    mu_spec (float): The mean of the Poisson distribution.
    n (int): The number of draws.
    
    Returns:
    np.ndarray: An array of n Poisson-distributed samples.
    """
    # Draw n samples from the Poisson distribution with the specified mean (mu_spec)
    poisson_samples = np.random.poisson(lam=mu_spec, size=n)
    
    return poisson_samples


today = datetime.today().date()+timedelta(days=0)
formatted_today = today.strftime('%Y-%m-%d')

tomorrow = today + timedelta(days=1)
formatted_tomorrow = tomorrow.strftime('%Y-%m-%d')
yesterday = today + timedelta(days=-1)
formatted_yesterday = yesterday.strftime('%Y-%m-%d')

completed_game_data = request_game_data('2024-10-03', formatted_today)
completed_games = extract_game_data(completed_game_data)
completed_games['gametime'] = pd.to_datetime(completed_games['game_time']).dt.strftime('%H:%M')



completed_games['date'] = pd.to_datetime(completed_games['date'])

completed_games = completed_games[(completed_games["game_type"]=="R")|(completed_games["game_type"]=="P")]
# Step 1: Compute the last game date for each team

# Create a temporary dataframe with dates for each team, regardless of home or away
all_games = completed_games.melt(id_vars='date', value_name='team', value_vars=['home_team', 'away_team'])
all_games_sorted = all_games.sort_values(['team', 'date'])

# Compute the date of the last game for each team
all_games_sorted['last_game_date'] = all_games_sorted.groupby('team')['date'].shift(1)

# Merge this information back into the completed_games dataframe
completed_games = pd.merge(completed_games, all_games_sorted[['date', 'team', 'last_game_date']], left_on=['date', 'home_team'], right_on=['date', 'team'], how='left').rename(columns={'last_game_date': 'home_last_game_date'}).drop('team', axis=1)
completed_games = pd.merge(completed_games, all_games_sorted[['date', 'team', 'last_game_date']], left_on=['date', 'away_team'], right_on=['date', 'team'], how='left').rename(columns={'last_game_date': 'away_last_game_date'}).drop('team', axis=1)

# Step 2: Compute the rest days based on this new information
completed_games['home_rest_days'] = (completed_games['date'] - completed_games['home_last_game_date']).dt.days - 1
completed_games['away_rest_days'] = (completed_games['date'] - completed_games['away_last_game_date']).dt.days - 1

completed_games["home_back2back"] = np.where(completed_games["home_rest_days"]==0,1,0)
completed_games["away_back2back"] = np.where(completed_games["away_rest_days"]==0,1,0)




completed_games["home_time_adv"]=define_time_advantages(completed_games,nhl_timezones)




team_data_archive_df = pd.read_csv("C:/Users/NolanNicholls/Documents/NHL/2024/scott_scripts/xg-stats-output.csv")[["Date","gid", "Neutral","Team Long", "Away Team Long", "5on5 Flurry Xgoals",
"opp_5on5 Flurry Xgoals", "PP_home_goals", "SH_home_goals", "PP_away_goals", "SH_away_goals"]]
# team_data_increment_df = pd.read_csv("C:/Users/NolanNicholls/Documents/NHL/2024/team_data_increment.csv")
# team_data_archive_df=pd.concat([team_data_archive_df,team_data_increment_df], axis=0)
# team_data_archive_df.to_csv("C:/Users/NolanNicholls/Documents/NHL/2024/team_data_archive.csv")


team_data_archive_df["home_spec_g"] = team_data_archive_df["PP_home_goals"]+team_data_archive_df["SH_home_goals"]
team_data_archive_df["away_spec_g"] = team_data_archive_df["PP_away_goals"]+team_data_archive_df["SH_away_goals"]

team_data_archive_df=team_data_archive_df.rename(columns={"gid":"game-ID", "5on5 Flurry Xgoals":"home_5v5_xg", "opp_5on5 Flurry Xgoals":"away_5v5_xg"
    
    })



completed_games = pd.merge(left=completed_games, right=team_data_archive_df, how='left', left_on="game-ID", right_on="game-ID")




completed_games = completed_games[['date','gametime','game-ID','home_team', 'away_team',
                                   'home_5v5_xg',
                                   'away_5v5_xg',
                                   'home_spec_g',
                                   'away_spec_g','home_time_adv','home_back2back','away_back2back','home_team_reg_score','away_team_reg_score', 'home_team_fin_score','away_team_fin_score']]





# Extract the unique list of teams and assign an integer label to each one
teams = data["team"]
teams = np.sort(teams)
teams = pd.DataFrame(teams, columns=['team'])
teams['i'] = teams.index



# Create a unique list of each team combination and assign an integer label
# to each one. Also decide which team will be 'heads' in each pair.
all_teams_pair_combinations = combinations(teams['team'], 2)
team_pairs_dict = {}
team_pairs_heads_dict = {}
pair_index = 0
for pair in all_teams_pair_combinations:
    team_pairs_dict[(pair[0], pair[1])] = pair_index
    team_pairs_dict[(pair[1], pair[0])] = pair_index
    team_pairs_heads_dict[(pair[0], pair[1])] = pair[0]
    team_pairs_heads_dict[(pair[1], pair[0])] = pair[0]
    pair_index += 1

completed_games = add_team_data_labels(completed_games)


def extract_list_value(value):
    if isinstance(value, list) and len(value) == 1:
                
        if isinstance(value[0], str) and value[0].startswith('['):
            return ast.literal_eval(value[0])
        else:
            return value[0]
    else:
        return value

def compute_rest_in_five(current_game_date, team, df):
    """
    Compute total rest days in the past five days for the specified team.

    Parameters:
    - current_game_date (datetime): The date of the game in question.
    - team (str): Name of the team (home or away).
    - df (DataFrame): DataFrame containing the games information.

    Returns:
    - int: Total rest days in the past five days for the specified team.
    """

    # Calculate the start and end date for the 5-day window
    five_days_ago = current_game_date - pd.Timedelta(days=5)
    end_date = current_game_date  # Not including the current game day

    # Filter games where the specified team played within the 5-day window

    relevant_games = df[
        (df['date'] >= five_days_ago) &
        (df['date'] < end_date) &
        ((df['away_team'] == team) | (df['home_team'] == team))
        ]

    # Count the games played
    games_played = len(relevant_games)

    # Calculate rest days
    rest_days = 5 - games_played

    return rest_days




def get_observed(season_frame, neutral_games):
    # Convert to numeric and get values
    home_team = pd.to_numeric(season_frame.i_home).values
    away_team = pd.to_numeric(season_frame.i_away).values
    team_pair = pd.to_numeric(season_frame.i_pair).values
    observed_home_goals = pd.to_numeric(season_frame.home_team_reg_score).values
    observed_away_goals = pd.to_numeric(season_frame.away_team_reg_score).values
    observed_pair_outcomes = pd.to_numeric(season_frame.i_pair_winner).values
    observed_home_5v5_xg = pd.to_numeric(season_frame.home_5v5_xg).values
    observed_away_5v5_xg = pd.to_numeric(season_frame.away_5v5_xg).values
    observed_home_spec_g = pd.to_numeric(season_frame.home_spec_g).values
    observed_away_spec_g = pd.to_numeric(season_frame.away_spec_g).values
    observed_home_time_adv = pd.to_numeric(season_frame.home_time_adv).values
    observed_home_team_day_off_adv = pd.to_numeric(season_frame.home_back2back).values
    observed_away_team_day_off_adv = pd.to_numeric(season_frame.away_back2back).values
    observed_gameid = season_frame["game-ID"].values

    # Correct list comprehension for neutral games
    observed_neutral = np.array([1 if id in neutral_games else 0 for id in observed_gameid])

    return (home_team, away_team, team_pair, observed_home_goals, observed_away_goals,
            observed_pair_outcomes, observed_home_5v5_xg, observed_away_5v5_xg,
            observed_home_spec_g, observed_away_spec_g, observed_home_time_adv,
            observed_home_team_day_off_adv, observed_away_team_day_off_adv, observed_neutral)




def run_with_priors(priors, home_team, away_team, observed_home_5v5_xg, observed_away_5v5_xg, observed_home_spec_goals,observed_away_spec_goals, observed_home_time_adv,
                 
                    observed_home_team_back2back, observed_away_team_back2back,observed_neutral):
    
    with pm.Model() as model:

        hfa_5v5=0.088
        hfa_spec =0.073
        home_time_adv = 0.0197


        
        home_day_off_adv_5v5 = 0.087/2
        home_day_off_adv_spec = 0.087/2

        away_day_off_adv_5v5 = 0.087
        away_day_off_adv_spec = 0.087

        # intercept_5v5=0.67
        # intercept_spec = -0.409
        intercept_5v5=0.686
        intercept_spec = -0.412
        inverted_observed_neutral = 1 - observed_neutral

        offence_5v5 = pm.Normal('offence_5v5', mu=np.array(priors['off_5v5_mu']),sigma=np.full(num_teams, 0.085), shape=num_teams)
        defence_5v5 = pm.Normal('defence_5v5', mu=np.array(priors['def_5v5_mu']), sigma=np.full(num_teams, 0.085), shape=num_teams)
        offence_spec = pm.Normal('offence_spec', mu=np.array(priors["off_spec_mu"]), sigma=np.full(num_teams, 0.11), shape=num_teams)
        defence_spec = pm.Normal('defence_spec', mu=np.array(priors["def_spec_mu"]), sigma=np.full(num_teams, 0.11), shape=num_teams)

        

        delta_5v5 = pm.Normal('delta_5v5', mu=0.0, sigma=0.001, shape=32)
        delta_spec = pm.Normal('delta_spec', mu=0.0, sigma=0.001, shape=32)


        offence_5v5_walked = pm.Deterministic('offence_5v5_walked', offence_5v5 + delta_5v5)
        defence_5v5_walked = pm.Deterministic('defence_5v5_walked', defence_5v5 + delta_5v5)
        offence_spec_walked = pm.Deterministic('offence_spec_walked', offence_spec + delta_spec)
        defence_spec_walked = pm.Deterministic('defence_spec_walked', defence_spec + delta_spec)

        
        offence_5v5_mean = pm.math.sum(offence_5v5_walked) / num_teams
        defence_5v5_mean = pm.math.sum(defence_5v5_walked) / num_teams
        offence_spec_mean = pm.math.sum(offence_spec_walked) / num_teams
        defence_spec_mean = pm.math.sum(defence_spec_walked) / num_teams

        offence_5v5_norm = pm.Deterministic('offence_5v5_norm', offence_5v5_walked -offence_5v5_mean)
        defence_5v5_norm = pm.Deterministic('defence_5v5_norm', defence_5v5_walked - defence_5v5_mean)
        offence_spec_norm = pm.Deterministic('offence_spec_norm', offence_spec_walked -offence_spec_mean)
        defence_spec_norm = pm.Deterministic('defence_spec_norm', defence_spec_walked - defence_spec_mean)

        home_team_shared = pm.Data('home_team', home_team)
        away_team_shared = pm.Data('away_team', away_team)

        # Indexing using the shared variables
        home_5v5_offence = offence_5v5_norm[home_team_shared]
        away_5v5_offence = offence_5v5_norm[away_team_shared]
        home_5v5_defence = defence_5v5_norm[home_team_shared]
        away_5v5_defence = defence_5v5_norm[away_team_shared]
        home_spec_offence = offence_spec_norm[home_team_shared]
        away_spec_offence = offence_spec_norm[away_team_shared]
        home_spec_defence = defence_spec_norm[home_team_shared]
        away_spec_defence = defence_spec_norm[away_team_shared]

        home_theta_5v5_mu = pm.math.exp(intercept_5v5 + hfa_5v5*inverted_observed_neutral + home_5v5_offence - away_5v5_defence + home_day_off_adv_5v5*observed_away_team_back2back - away_day_off_adv_5v5*observed_home_team_back2back + home_time_adv*observed_home_time_adv)
        away_theta_5v5_mu = pm.math.exp(intercept_5v5 + away_5v5_offence - home_5v5_defence - home_day_off_adv_5v5*observed_away_team_back2back + away_day_off_adv_5v5*observed_home_team_back2back - home_time_adv*observed_home_time_adv)
        
        home_theta_spec_mu = pm.math.exp(intercept_spec + hfa_spec*inverted_observed_neutral + home_spec_offence - away_spec_defence)
        away_theta_spec_mu = pm.math.exp(intercept_spec + away_spec_offence - home_spec_defence)

        home_xgoal_obs = pm.Gamma('home_xgoal_obs', mu=home_theta_5v5_mu, sigma=home_theta_5v5_mu*0.2902 + 0.0780, observed=observed_home_5v5_xg)
        away_xgoal_obs = pm.Gamma('away_xgoal_obs', mu=away_theta_5v5_mu, sigma=away_theta_5v5_mu*0.2902 + 0.0780 , observed=observed_away_5v5_xg)

        home_SPEC_goals_obs = pm.Poisson('home_SPEC_goals_obs', mu=home_theta_spec_mu,observed=observed_home_spec_goals)
        away_SPEC_goals_obs = pm.Poisson('away_SPEC_goals_obs', mu=away_theta_spec_mu, observed=observed_away_spec_goals)
    


        trace = pm.sample(5000, tune=2500, cores=1)

        def mle_fit(data):
            """Return the mean and standard deviation of the normal distribution fitting the data using MLE."""
            return norm.fit(data)

        def extract_recent_priors(trace, num_teams):
         
            # Not flattening these as we want to index them by team later
            offence_5v5_posterior = trace.posterior['offence_5v5_norm'].values
            defence_5v5_posterior = trace.posterior['defence_5v5_norm'].values
            offence_spec_posterior = trace.posterior['offence_spec_norm'].values
            defence_spec_posterior = trace.posterior['defence_spec_norm'].values
        

            
        
           

            recent_priors = {
                

                    }

            offence_5v5_mle = [mle_fit(offence_5v5_posterior[..., i].flatten()) for i in range(num_teams)]
            defence_5v5_mle = [mle_fit(defence_5v5_posterior[..., i].flatten()) for i in range(num_teams)]
            offence_spec_mle = [mle_fit(offence_spec_posterior[..., i].flatten()) for i in range(num_teams)]
            defence_spec_mle = [mle_fit(defence_spec_posterior[..., i].flatten()) for i in range(num_teams)]


            recent_priors.update({
                'off_5v5_mu': [mu for mu, _ in offence_5v5_mle],
                'def_5v5_mu': [mu for mu, _ in defence_5v5_mle],
                'off_spec_mu': [mu for mu, _ in offence_spec_mle],
                'def_spec_mu': [mu for mu, _ in defence_spec_mle],
            })

            return recent_priors

        # Usage
        recent_priors = extract_recent_priors(trace, num_teams)

        return recent_priors




def play_game_date(row,priors_dict, home_goalie, away_goalie,neutral):

    home_idx = teams[teams["team"]==row["home_team"].iloc[0]]["i"].iloc[0]
    away_idx = teams[teams["team"]==row["away_team"].iloc[0]]["i"].iloc[0]

    away_back2back = row["away_back2back"].iloc[0]
    home_back2back = row["home_back2back"].iloc[0]
    home_time_adv = row["home_time_adv"].iloc[0]


    
    mu_off_home = priors_dict['off_5v5_mu'][home_idx] 
    mu_def_home = priors_dict['def_5v5_mu'][home_idx]
    mu_off_away = priors_dict['off_5v5_mu'][away_idx] 
    mu_def_away = priors_dict['def_5v5_mu'][away_idx]

    mu_off_SPEC_home = priors_dict['off_spec_mu'][home_idx]
    mu_def_SPEC_home = priors_dict['def_spec_mu'][home_idx]
    mu_off_SPEC_away = priors_dict['off_spec_mu'][away_idx]
    mu_def_SPEC_away = priors_dict['def_spec_mu'][away_idx]

    home_goalie_val = get_goalie_value(home_goalie,2024,row["game-ID"])*0.8
    away_goalie_val = get_goalie_value(away_goalie,2024,row["game-ID"])*0.8
    n=1000000
    
    if neutral:
        hfa_5v5=0
        hfa_spec =0
    else:
        hfa_5v5=0.088
        hfa_spec =0.073


    int_5v5=0.686
    int_spec = -0.412

    powerplay_corr = 0.08
    home_day_off_adv_5v5 = 0.077/2
    home_day_off_adv_spec = 0.077/2
    away_day_off_adv_5v5 = 0.077
    away_day_off_adv_spec = 0.063
    home_time_adv_factor = 0.0197

    
    team1_gamma = poisson_gamma(math.exp(int_5v5+ mu_off_home -mu_def_away  + hfa_5v5 + home_day_off_adv_5v5*away_back2back - away_day_off_adv_5v5*home_back2back + home_time_adv_factor*home_time_adv)- away_goalie_val, n)
    team1_poisson = poisson_draws(math.exp(int_spec+mu_off_SPEC_home-mu_def_SPEC_away + hfa_spec  + home_day_off_adv_spec*away_back2back -away_day_off_adv_spec*home_back2back)+powerplay_corr, n)
    team1 = team1_gamma + team1_poisson
    
    team2_gamma = poisson_gamma(math.exp(int_5v5+ mu_off_away-mu_def_home- home_day_off_adv_5v5*away_back2back + away_day_off_adv_5v5*home_back2back)- home_goalie_val , n)
    team2_poisson = poisson_draws(math.exp(int_spec+ mu_off_SPEC_away -mu_def_SPEC_home- home_day_off_adv_spec*away_back2back + away_day_off_adv_spec*home_back2back)+powerplay_corr , n)
    team2 = team2_gamma + team2_poisson

    # Counters for the effects
    one_goal_diff_counter = 0
    tie_breaker_counter = 0
    
    # Additional scoring logic
    additional_goals_team1 = 0
    additional_goals_team2 = 0
    
    # Track goals specifically added by tie-breakers and one-goal difference scenarios
    goals_from_tie_breaker = 0
    goals_from_one_goal_diff = 0
    
    for i in range(n):
        if (abs(team1[i] - team2[i]) <= 2)&(abs(team1[i] - team2[i]) != 0):
            one_goal_diff_counter += 1
            
            # Generate a random number to decide which event occurs
            random_value = np.random.rand()
    
            if random_value < 0.14:
                # 15% chance the losing team gets an extra point
                if team1[i] < team2[i]:
                    team1[i] += 1
                    additional_goals_team1 += 1
                    goals_from_one_goal_diff += 1
                else:
                    team2[i] += 1
                    additional_goals_team2 += 1
                    goals_from_one_goal_diff += 1
            elif random_value < 0.60:
                # 40% chance the winning team gets an extra point
                if team1[i] > team2[i]:
                    team1[i] += 1
                    additional_goals_team1 += 1
                    goals_from_one_goal_diff += 1
                else:
                    team2[i] += 1
                    additional_goals_team2 += 1
                    goals_from_one_goal_diff += 1
    
        

        home_lambda_OT = math.exp(int_5v5+ mu_off_home -mu_def_away  + hfa_5v5 + home_day_off_adv_5v5*away_back2back - away_day_off_adv_5v5*home_back2back + home_time_adv_factor*home_time_adv)- away_goalie_val
        away_lambda_OT = math.exp(int_5v5+ mu_off_away-mu_def_home- home_day_off_adv_5v5*away_back2back + away_day_off_adv_5v5*home_back2back)- home_goalie_val
        home_win_prob_OT = home_lambda_OT/(home_lambda_OT+away_lambda_OT)
        
        if team1[i] == team2[i]:
            tie_breaker_counter += 1
            if np.random.rand() < home_win_prob_OT:
                team1[i] += 1
                additional_goals_team1 += 1
                goals_from_tie_breaker += 1
            else:
                team2[i] += 1
                additional_goals_team2 += 1
                goals_from_tie_breaker += 1
               
            

            
    
    # Calculate total goals from each distribution
    total_goals_gamma = np.sum(team1_gamma) + np.sum(team2_gamma)
    total_goals_poisson = np.sum(team1_poisson) + np.sum(team2_poisson)
    total_additional_goals = additional_goals_team1 + additional_goals_team2
    total_goals = total_goals_gamma + total_goals_poisson + total_additional_goals
    
    # Calculate proportions for each goal source
    proportion_goals_gamma = total_goals_gamma / total_goals
    proportion_goals_poisson = total_goals_poisson / total_goals
    proportion_tie_breaker_goals = goals_from_tie_breaker / total_goals
    proportion_one_goal_diff_goals = goals_from_one_goal_diff / total_goals
    
    team1_wins = np.sum(team1 > team2)
    team1_win_proportion = team1_wins / n
    team1_lossesless15 = np.sum(team1+1.5 > team2)/n
    team1_winsmore15 = np.sum(team1-1.5 > team2)/n


    simulated_total=team1+team2

    possible_totals = [5,5.5,6,6.5,7,7.5,8]

    prob_over=[]
    prob_under=[]

    for total in possible_totals:
        prob_over.append(np.mean(simulated_total>total))
        prob_under.append(np.mean(simulated_total<total))

    prob_over_norm =[]
    for i,total in enumerate(possible_totals):
        prob_over_norm.append(prob_over[i]/(prob_over[i]+prob_under[i]))
                         
        
      

    return team1_win_proportion,team1_lossesless15,team1_winsmore15,prob_over_norm[0],prob_over_norm[1],prob_over_norm[2],prob_over_norm[3],prob_over_norm[4],prob_over_norm[5],prob_over_norm[6]




goalie_archive_file = "C:/Users/NolanNicholls/Documents/NHL/2024/goalie_data_archive.csv"

goalie_gsa_stats = pd.read_csv(goalie_archive_file)[["season","date", "gameid", "goaliename", "GSAx"]]
goalie_gsa_stats.columns = ["season","date", "gameid", "goaliename", "GSAx"]

# goalie_gsa_stats["date"]=pd.to_datetime(goalie_gsa_stats["date"])
goalie_gsa_stats = goalie_gsa_stats[["season", "date", "gameid", "goaliename", "GSAx"]]

goalie_stats_increment = pd.read_csv("C:/Users/NolanNicholls/Documents/NHL/2024/scott_scripts/goalie-gsax-output.csv")[["Date","gid","playerName", "gsaX"]]
goalie_stats_increment["season"]=2024
goalie_stats_increment=goalie_stats_increment[["season","Date","gid","playerName", "gsaX"]]
goalie_stats_increment.columns=["season","date", "gameid", "goaliename", "GSAx"]
# goalie_stats_increment["date"]=pd.to_datetime(goalie_stats_increment["date"])


goalie_gsa_stats=pd.concat([goalie_gsa_stats,goalie_stats_increment], axis=0)

# goalie_gsa_stats.to_csv(goalie_archive_file,index=False)



config_player_sf = 0.01

config_player_career_sf_base=0.008
config_player_career_sf_height=0.002
config_player_career_sf_mp=200

config_player_regression_career_height=0.6
config_player_regression_career_mp=200


config_player_regression_league_height=0.3
config_player_regression_league_mp=75


df_goalie_avg = goalie_gsa_stats.groupby(by="season")["GSAx"].agg("mean")

def s_curve(height, mp, x, direction='down'):
        ## calculate s-curve, which are used for progression discounting and multiplying ##
        if direction == 'down':
            return (
                1 - (1 / (1 + 1.5 ** (
                    (-1 * (x - mp)) *
                    (10 / mp)
                )))
            ) * height
        else:
            return (1-(
                1 - (1 / (1 + 1.5 ** (
                    (-1 * (x - mp)) *
                    (10 / mp)
                )))
            )) * height

def get_prev_season_league_avg(season):
    return df_goalie_avg.loc[season-1]

class Goalie:
    def __init__(self, name,init_value, gameday, season):
        self.name = None
        self.current_value = init_value
        self.current_variance = init_value
        self.rolling_value = init_value
        self.starts = 0
        self.season_starts = 0
        self.first_game_date = gameday
        self.first_game_season = season
        self.last_game_date = None
        self.last_game_season = None

def update_goalie_value(goalie, value, gameday, season):
   
    ## first store the pre-update value, which i sneeded for rolling variance ##
    old_value = goalie.current_value
    ## update the qb value after the game ##
    goalie_current_value = (
        config_player_sf * value +
        (1 - config_player_sf) * goalie.current_value
    )
    ## for rolling value, use a progressively deweighted ewma ##
    ## set rolling sf ##
    rolling_sf = (
        config_player_career_sf_base +
        s_curve(
            config_player_career_sf_height,
            config_player_career_sf_mp,
            goalie.starts,
            'down'
        )
    )
    goalie.rolling_value = (
        rolling_sf * value +
        (1 - rolling_sf) * goalie.rolling_value
    )
    ## update variance ##
    ## 𝜎2𝑛=(1−𝛼)𝜎2𝑛−1+𝛼(𝑥𝑛−𝜇𝑛−1)(𝑥𝑛−𝜇𝑛) ##
    ## https://stats.stackexchange.com/questions/6874/exponential-weighted-moving-skewness-kurtosis ##
    goalie.current_variance = (
        config_player_sf * (value - old_value) * (value - goalie.current_value) +
        (1 - config_player_sf) * goalie.current_variance
    )
    ## update meta ##
    goalie.starts += 1
    goalie.season_starts += 1
    goalie.last_game_date = gameday
    goalie.last_game_season = season
    
    ## return updated value ##
    return goalie_current_value

def get_goalie_value_dict_build(row):
    ## retrieve the current value of the goalie before the gamae ##
    ## this takes the entire row as we may need to unpack values to send to
    goalie_name = row["goaliename"]
    if row['goaliename'] not in goalies_dict.keys():
        goalies_dict[goalie_name] = Goalie(name=row['goaliename'], init_value=-0.05, gameday=row["gameid"], season=row["season"])
  
    ## handle regression ##
    if goalies_dict[goalie_name].last_game_season is None:
        return goalies_dict[goalie_name].current_value
    elif row['season'] > goalies_dict[goalie_name].last_game_season:
        goalie = handle_goalie_regression(goalies_dict[goalie_name], row['season'])
        return goalie.current_value
    ## return value ##
    return goalies_dict[goalie_name].current_value


def get_goalie_value(name,season,gameid):
    ## retrieve the current value of the goalie before the gamae ##
    ## this takes the entire row as we may need to unpack values to send to
    goalie_name = name
    if goalie_name not in goalies_dict.keys():
        goalies_dict[goalie_name] = Goalie(name=goalie_name, init_value=-0.1, gameday=gameid, season=season)
  
    ## handle regression ##
    if goalies_dict[goalie_name].last_game_season is None:
        return goalies_dict[goalie_name].current_value
    elif season > goalies_dict[goalie_name].last_game_season:
        goalie = handle_goalie_regression(goalies_dict[goalie_name], season)
        return goalie.current_value
    ## return value ##
    return goalies_dict[goalie_name].current_value
    


def handle_goalie_regression(goalie, season):
        ## regress goalie to the league average ##
        ## first, get the previous season average ##
        prev_season_league_avg = get_prev_season_league_avg(season)
        ## determine regression amounts based on model curves ##
        league_regression = s_curve(
            config_player_regression_league_height,
            config_player_regression_league_mp,
            goalie.starts,
            'down'
        )
        career_regression = s_curve(
            config_player_regression_career_height,
            config_player_regression_career_mp,
            goalie.starts,
            'up'
        )
        ## calculate the new value ##
        ## if the goalie didnt play much the previous (ie was a backup) this is ##
        ## signal that they are not league average quality ##
        ## In this case, we discount the league average regression portion ##
        league_regression = (
            league_regression *
            s_curve(
                1,
                10,
                goalie.season_starts,
                'up'
            )
        )
        ## normalize the combined career and league regression to not exceed 100% ##
        total_regression = league_regression + career_regression
        if total_regression > 1:
            league_regression = league_regression / total_regression
            career_regression = career_regression / total_regression
        ## calculate value ##
        goalie.current_value = (
            (1 - league_regression - career_regression) * goalie.current_value +
            (league_regression * prev_season_league_avg) +
            (career_regression * goalie.rolling_value)
        )
        ## update season ##
        ## return the qb object ##
        return goalie




def get_prev_season_league_avg(season):
    return df_goalie_avg.loc[season-1]




goalies_dict_archive = defaultdict(dict)
goalies_dict = {}

for i, row in goalie_gsa_stats[goalie_gsa_stats["season"] > 2015].reset_index(drop=True).iterrows():
    goalie_name = row["goaliename"]
    row['season'] = int(row['season'])
    date = row["date"]

    # Get the current goalie value or initialize if new
    if goalie_name in goalies_dict.keys():
        # Get the pre-update value
        goalie_value = get_goalie_value_dict_build(row)
    else:
        # If goalie is new, set initial value
        goalie_value = -0.05
        goalies_dict[goalie_name] = Goalie(name=goalie_name, init_value=goalie_value, gameday=row["gameid"], season=int(row["season"]))

    # If the date is not in the archive yet, create it and carry over previous values
    if date not in goalies_dict_archive:
        # Copy all goalies' current values from the last date (if it exists)
        if goalies_dict_archive:
            previous_date = max(goalies_dict_archive.keys())
            goalies_dict_archive[date] = goalies_dict_archive[previous_date].copy()

    # Update the value for the current goalie for this date (before the game)
    goalies_dict_archive[date][goalie_name] = goalie_value

    # Update the goalie's value after the game
    goalies_dict[goalie_name].current_value = update_goalie_value(goalies_dict[goalie_name], row["GSAx"], row["gameid"], int(row["season"]))


with open('C:/Users/NolanNicholls/Documents/NHL/2024/goalies_dict_archive.pkl', 'wb') as f:
    pickle.dump(goalies_dict_archive, f)




team_2024_forecasts = pd.read_csv("C:/Users/NolanNicholls/Documents/NHL/2024/2024_preseason_ratings.csv")
offence_preseason_5v5 = team_2024_forecasts["off_5v5_mu"].values.tolist()  # Convert to list
defence_preseason_5v5 = team_2024_forecasts["def_5v5_mu"].values.tolist()  # Convert to list
offence_preseason_spec = team_2024_forecasts["off_spec_mu"].values.tolist()  # Convert to list
defence_preseason_spec = team_2024_forecasts["def_spec_mu"].values.tolist()  # Convert to list



first_day_of_season = '2024-10-04'

if today.strftime("%Y-%m-%d") == first_day_of_season:
    df_priors = pd.DataFrame({
        "date": ['2024-10-04'],
        "off_5v5_mu": [offence_preseason_5v5],  # Wrap list in another list to create one row
        "def_5v5_mu": [defence_preseason_5v5],  # Wrap list in another list to create one row
        "off_spec_mu": [offence_preseason_spec],  # Wrap list in another list to create one row
        "def_spec_mu": [defence_preseason_spec]   # Wrap list in another list to create one row
    })
else:
    df_priors = pd.read_csv('C:/Users/NolanNicholls/Documents/NHL/2024/priors_saved.csv')
 



Game_DF_2024=completed_games.copy().drop_duplicates()


file_path_preds = "C:/Users/NolanNicholls/Documents/NHL/2024/predictions_saved.csv"

if today != pd.to_datetime(first_day_of_season).date():
    df_predictions = pd.read_csv(file_path_preds)
    df_predictions["date"] = pd.to_datetime(df_predictions["date"]).dt.date
    df_predictions = df_predictions[df_predictions["date"] <= today]
    dates_missing_predictions = df_predictions[df_predictions["home_win_prob"].isnull()]["date"].unique().tolist()

else:
    df_predictions = pd.DataFrame(columns=["date", "game-ID", "home_goalie", "away_goalie", "neutral", 
                                             "home_win_prob", "home_lossesless15", "home_winsmore15", 
                                             "prob_over_5", "prob_over_55", "prob_over_6", "prob_over_65", 
                                             "prob_over_7", "prob_over_75", "prob_over_8"])
    dates_missing_predictions=[]





dates_missing_predictions.append(today.strftime("%Y-%m-%d"))



for i,date in enumerate(dates_missing_predictions):
 


    if df_priors[df_priors['date'] == date].empty:

        yesterday_games = Game_DF_2024[Game_DF_2024['date'] == yesterday.strftime('%Y-%m-%d')]
        today_games = Game_DF_2024[Game_DF_2024['date'] == date]
        today_games = pd.merge(left=today_games, right=df_goalies, how='left', left_on=['home_team'], right_index=True)
        today_games.rename(
            columns={'goalie_1': 'home_goalie_1', 'goalie_2': 'home_goalie_2', 'goalie_3': 'home_goalie_3'},
            inplace=True)
        today_games = pd.merge(left=today_games, right=df_goalies, how='left', left_on=['away_team'], right_index=True)
        today_games.rename(
            columns={'goalie_1': 'away_goalie_1', 'goalie_2': 'away_goalie_2', 'goalie_3': 'away_goalie_3'},
            inplace=True)


    

        home_team_completed, away_team_completed, team_pair_completed, observed_home_goals_completed, observed_away_goals_completed, observed_pair_outcomes_completed,observed_home_5v5_xg,observed_away_5v5_xg,observed_home_spec_g,observed_away_spec_g,observed_home_time_adv,observed_home_team_day_off_adv,observed_away_team_day_off_adv,observed_neutral_games = get_observed(yesterday_games,neutral_games)

        
        loaded_priors = df_priors[df_priors['date'] == yesterday.strftime('%Y-%m-%d')].to_dict(orient='list')




        


        loaded_priors = {k: extract_list_value(v) for k, v in loaded_priors.items()}
 

        _2024_priors_update = run_with_priors(loaded_priors,home_team_completed, away_team_completed,
                                                                observed_home_5v5_xg,observed_away_5v5_xg,
                                                                observed_home_spec_g,observed_away_spec_g,
                                                                
                                                                observed_home_time_adv,
                                                                observed_home_team_day_off_adv,
                                                                observed_away_team_day_off_adv,observed_neutral_games)



        offence_5v5_mu = _2024_priors_update['off_5v5_mu']
        defence_5v5_mu = _2024_priors_update['def_5v5_mu']
        offence_spec_mu = _2024_priors_update['off_spec_mu']
        defence_spec_mu = _2024_priors_update['def_spec_mu']


        df_priors_new_row = pd.DataFrame([[date, offence_5v5_mu, defence_5v5_mu, offence_spec_mu, defence_spec_mu]],
                                         columns=['date', 'off_5v5_mu', 'def_5v5_mu', 'off_spec_mu',
                                                  'def_spec_mu'])

        df_priors = pd.concat([df_priors, df_priors_new_row], ignore_index=True)

        loaded_posteriors = df_priors[df_priors['date'] == date].to_dict(orient='list')
        loaded_posteriors = {k: extract_list_value(v) for k, v in loaded_posteriors.items()}

      

        df_priors.to_csv('C:/Users/NolanNicholls/Documents/NHL/2024/priors_saved.csv', index=False)



        for j in today_games['game-ID'].unique().tolist():

            game_row = today_games[today_games['game-ID'] == j]
            home_goalies_list = [game_row['home_goalie_1'].values[0], game_row['home_goalie_2'].values[0],
                                 game_row['home_goalie_3'].values[0]]
            away_goalies_list = [game_row['away_goalie_1'].values[0], game_row['away_goalie_2'].values[0],
                                 game_row['away_goalie_3'].values[0]]

            for home_goalie in home_goalies_list:
                if home_goalie is not None:
                    for away_goalie in away_goalies_list:
                        if away_goalie is not None:
                            if str(j) in neutral_games:
                                neutral_var = True

                            else:
                                neutral_var = False
                               
                            pred_result = play_game_date(game_row,loaded_posteriors ,home_goalie, away_goalie,neutral_var)

                            pred_block_i = pd.DataFrame([[
                            date, j,game_row["home_team"].iloc[0],game_row["away_team"].iloc[0], home_goalie, away_goalie,
                            pred_result[0], 1-pred_result[0], pred_result[1], pred_result[2], 
                            pred_result[3], pred_result[4], pred_result[5], pred_result[6], 
                            pred_result[7], pred_result[8],pred_result[9]
                        ]], columns=["date", "game-ID","home_team","away_team", "home_goalie", "away_goalie", 
                                     "home_win_prob","away_win_prob" ,"home_lossesless15", "home_winsmore15", 
                                     "prob_over_5", "prob_over_55", "prob_over_6", "prob_over_65", 
                                         "prob_over_7", "prob_over_75", "prob_over_8"])
                                



                            df_predictions = pd.concat([df_predictions, pred_block_i], ignore_index=True)

        df_predictions.to_csv('C:/Users/NolanNicholls/Documents/NHL/2024/predictions_saved.csv', index=False)

    else:
    
        today_games = Game_DF_2024[Game_DF_2024['date'] == date]
     
        today_games = pd.merge(left=today_games, right=df_goalies, how='left', left_on=['home_team'], right_index=True)
        today_games.rename(
            columns={'goalie_1': 'home_goalie_1', 'goalie_2': 'home_goalie_2', 'goalie_3': 'home_goalie_3'},
            inplace=True)
        today_games = pd.merge(left=today_games, right=df_goalies, how='left', left_on=['away_team'], right_index=True)
        today_games.rename(
            columns={'goalie_1': 'away_goalie_1', 'goalie_2': 'away_goalie_2', 'goalie_3': 'away_goalie_3'},
            inplace=True)



        loaded_posteriors = df_priors[df_priors['date'] == date].to_dict(orient='list')
        loaded_posteriors = {k: extract_list_value(v) for k, v in loaded_posteriors.items()}
        df_priors.to_csv('C:/Users/NolanNicholls/Documents/NHL/2024/priors_saved.csv', index=False)

        for j in today_games['game-ID'].unique().tolist():

            game_row = today_games[today_games['game-ID'] == j]
            home_goalies_list = [game_row['home_goalie_1'].values[0], game_row['home_goalie_2'].values[0],
                                 game_row['home_goalie_3'].values[0]]
            away_goalies_list = [game_row['away_goalie_1'].values[0], game_row['away_goalie_2'].values[0],
                                 game_row['away_goalie_3'].values[0]]

            for home_goalie in home_goalies_list:
                if home_goalie is not None:
                    for away_goalie in away_goalies_list:
                        if away_goalie is not None:
                            prob_over_norm=np.zeros(7)
                            (home_win_prob, home_lossesless15, home_winsmore15, 
                             prob_over_norm[0], prob_over_norm[1], prob_over_norm[2], 
                             prob_over_norm[3], prob_over_norm[4], prob_over_norm[5], 
                             prob_over_norm[6])= play_game_date(game_row,loaded_posteriors ,home_goalie, away_goalie,neutral=False)


                            
                            pred_block_i = pd.DataFrame([[
                                    date, j,game_row["home_team"].iloc[0],game_row["away_team"].iloc[0], home_goalie, away_goalie,
                                    home_win_prob,1-home_win_prob, home_lossesless15, home_winsmore15,
                                    prob_over_norm[0], prob_over_norm[1], prob_over_norm[2],
                                    prob_over_norm[3], prob_over_norm[4], prob_over_norm[5], prob_over_norm[6]
                                ]], columns=["date", "game-ID","home_team","away_team","home_goalie", "away_goalie", 
                                             "home_win_prob","away_win_prob", "home_lossesless15", "home_winsmore15", 
                                             "prob_over_5", "prob_over_55", "prob_over_6", "prob_over_65", 
                                             "prob_over_7", "prob_over_75", "prob_over_8"])

                            df_predictions = pd.concat([df_predictions, pred_block_i], ignore_index=True)

            df_predictions.to_csv('C:/Users/NolanNicholls/Documents/NHL/2024/predictions_saved.csv', index=False)

work_computer = True




df_predictions_today = df_predictions[df_predictions['date'] == today.strftime('%Y-%m-%d')]


sheet = client.open("NHL_Dashboard_2024").worksheet("model-input")

sheet.delete_rows(2, sheet.row_count)

df_predictions_today=df_predictions_today[["date", "game-ID","home_team","away_team", 
                                         "home_win_prob","away_win_prob","home_winsmore15", "home_lossesless15",  "home_goalie", "away_goalie", 
                                         "prob_over_5", "prob_over_55", "prob_over_6", "prob_over_65", 
                                             "prob_over_7", "prob_over_75", "prob_over_8"]]

# Update the Google Sheet with the DataFrame
set_with_dataframe(sheet, df_predictions_today, row=2, include_column_header=False,include_index=False)


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
import datetime
from PIL import Image

# Define the path to the logos and the filename mapping
current_date = datetime.datetime.now().strftime('%Y-%m-%d')
base_path = 'C:/Users/NolanNicholls/Documents/NHL/Logos/'
team_logo_map = {
    'Anaheim Ducks': 'anaheim-ducks-logo.png',
    'Boston Bruins': 'boston-bruins-logo.png',
    'Buffalo Sabres': 'buffalo-sabres-logo.png',
    'Calgary Flames': 'calgary-flames-logo.png',
    'Carolina Hurricanes': 'carolina-hurricanes-logo.png',
    'Chicago Blackhawks': 'chicago-blackhawks-logo.png',
    'Colorado Avalanche': 'colorado-avalanche-logo.png',
    'Columbus Blue Jackets': 'columbus-blue-jackets-logo.png',
    'Dallas Stars': 'dallas-stars-logo.png',
    'Detroit Red Wings': 'detroit-red-wings-logo.png',
    'Edmonton Oilers': 'edmonton-oilers-logo.png',
    'Florida Panthers': 'florida-panthers-logo.png',
    'Los Angeles Kings': 'los-angeles-kings-logo.png',
    'Minnesota Wild': 'minnesota-wild-logo.png',
    'Montréal Canadiens': 'montreal-canadiens-logo.png',
    'Nashville Predators': 'nashville-predators-logo.png',
    'New Jersey Devils': 'new-jersey-devils-logo.png',
    'New York Islanders': 'new-york-islanders-logo.png',
    'New York Rangers': 'new-york-rangers-logo.png',
    'Ottawa Senators': 'ottawa-senators-logo.png',
    'Philadelphia Flyers': 'philadelphia-flyers-logo.png',
    'Pittsburgh Penguins': 'pittsburgh-penguins-logo.png',
    'San Jose Sharks': 'san-jose-sharks-logo.png',
    'Seattle Kraken': 'seattle-kraken-logo.png',
    'St. Louis Blues': 'st-louis-blues-logo.png',
    'Tampa Bay Lightning': 'tampa-bay-lightning-logo.png',
    'Toronto Maple Leafs': 'toronto-maple-leafs-logo.png',
    'Utah Hockey Club': 'utah-hockey-club-logo.png',
    'Vancouver Canucks': 'vancouver-canucks-logo.png',
    'Vegas Golden Knights': 'vegas-golden-knights-logo.png',
    'Washington Capitals': 'washington-capitals-logo.png',
    'Winnipeg Jets': 'winnipeg-jets-logo.png'
}

# Create the scatter plot
plt.figure(figsize=(12, 8),dpi=300)

# Define a standard size for all logos
standard_size = (50, 50)  # you can adjust this as needed

# Loop through each team and place their logo on the plot
# Loop through each team and place their logo on the plot

df_ratings_5v5=pd.DataFrame({
    'team': teams['team'],
    'offensive_rating': loaded_posteriors['off_5v5_mu'],
    'defensive_rating': loaded_posteriors['def_5v5_mu']
})


for _, row in df_ratings_5v5.iterrows():
    logo_path = base_path + team_logo_map[row['team']]
    # Open the logo using PIL
    logo = Image.open(logo_path)

    # If the team is 'Seattle Kraken', adjust the size
    if (row['team'] == 'Seattle Kraken')|(row['team'] == 'Utah Hockey Club')|(row['team'] == 'Anaheim Ducks'):
        size = (int(standard_size[0] // 1.7), int(standard_size[1] // 1.7))
    else:
        size = standard_size

    # Resize the logo
    logo_resized = logo.resize(size)

    imagebox = offsetbox.OffsetImage(logo_resized, zoom=1)  # use zoom=1 since we've already resized
    ab = offsetbox.AnnotationBbox(imagebox, (row['defensive_rating'], row['offensive_rating']),
                                  frameon=False, pad=0.2)
    plt.gca().add_artist(ab)

# Set the title and labels for axes

plt.xlabel('Defensive Rating', fontsize=12)
plt.ylabel('Offensive Rating', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

plt.gca().set_aspect('equal', adjustable='box')
# Padding can be a fixed amount or a percentage of the range
padding = 0.05

# Determine the range for the x-axis (defensive ratings)
x_min = df_ratings_5v5['defensive_rating'].min() - padding
x_max = df_ratings_5v5['defensive_rating'].max() + padding

# Determine the range for the y-axis (offensive ratings)
y_min = df_ratings_5v5['offensive_rating'].min() - padding
y_max = df_ratings_5v5['offensive_rating'].max() + padding

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
# Show the plot

plt.xticks([])
plt.yticks([])

plt.annotate("Better", xy=(0.95, -0.1), xycoords="axes fraction",
             xytext=(0.75, -0.1), textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", color="black"),
             fontsize=14, va="center")

# For y-axis
plt.annotate("Better", xy=(-0.1, 0.95), xycoords="axes fraction",
             xytext=(-0.1, 0.75), textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", color="black"),
             fontsize=14, va="center", rotation=90)

plt.title(f"Team Ratings (5v5) for {current_date}", fontsize=16)
plt.tight_layout()


plt.savefig('C:/Users/NolanNicholls/Documents/NHL/2024/team_ratings_5v5.png')
# Sort by date in descending order
plt.clf()

plt.figure(figsize=(12, 8), dpi=300)

df_ratings_spec=pd.DataFrame({
    'team': teams['team'],
    'offensive_rating': loaded_posteriors['off_spec_mu'],
    'defensive_rating': loaded_posteriors['def_spec_mu']
})

# Example using the same data but you could modify it for a different type of rating
for _, row in df_ratings_spec.iterrows():
    logo_path = base_path + team_logo_map[row['team']]
    logo = Image.open(logo_path)

    size = standard_size if row['team'] not in ['Seattle Kraken', 'Utah Hockey Club', 'Anaheim Ducks'] else (int(standard_size[0] // 1.7), int(standard_size[1] // 1.7))
    logo_resized = logo.resize(size)
    imagebox = offsetbox.OffsetImage(logo_resized, zoom=1)
    ab = offsetbox.AnnotationBbox(imagebox, (row['defensive_rating'], row['offensive_rating']),
                                  frameon=False, pad=0.2)
    plt.gca().add_artist(ab)

plt.title(f"Team Ratings (Special Teams) for {current_date}", fontsize=16)
plt.xlabel('Defensive Rating', fontsize=12)
plt.ylabel('Offensive Rating', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')


# Determine the range for the x-axis (defensive ratings)
x_min = df_ratings_spec['defensive_rating'].min() - padding
x_max = df_ratings_spec['defensive_rating'].max() + padding

# Determine the range for the y-axis (offensive ratings)
y_min = df_ratings_spec['offensive_rating'].min() - padding
y_max = df_ratings_spec['offensive_rating'].max() + padding

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Set limits for the new plot
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks([])
plt.yticks([])

plt.annotate("Improvement", xy=(0.95, -0.1), xycoords="axes fraction",
             xytext=(0.75, -0.1), textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", color="black"),
             fontsize=14, va="center")

plt.annotate("Improvement", xy=(-0.1, 0.95), xycoords="axes fraction",
             xytext=(-0.1, 0.75), textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", color="black"),
             fontsize=14, va="center", rotation=90)

plt.tight_layout()
plt.savefig('C:/Users/NolanNicholls/Documents/NHL/2024/team_ratings_spec.png')

# Optionally, close the second plot
plt.close()


# Initialize the lists
current_goalies_list = []
goalie_value = []

# Loop through goalies_dict and collect goalies and their values
for goalie in goalies_dict.keys():
    
    current_goalies_list.append(goalie)
    goalie_value.append(get_goalie_value(name=goalie, season=2024, gameid= 202401))

# Create a DataFrame from the lists
goalie_df = pd.DataFrame({
    "goalie_name": current_goalies_list,
    "goalie_value": goalie_value
})

goalie_df = goalie_df.sort_values(by="goalie_value", ascending=False).reset_index(drop=True)

sheet_goalie_ratings = client.open("NHL_Dashboard_2024").worksheet("model_goalie_input")
sheet_goalie_ratings.clear()


# Update the Google Sheet with the DataFrame
set_with_dataframe(sheet_goalie_ratings, goalie_df, row=2, include_column_header=False,include_index=True)


from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


creds = ServiceAccountCredentials.from_json_keyfile_name("C:/Users/NolanNicholls/Documents/NHL/2024/credentials/smooth-topic-379501-521289118ec5.json", scope)
drive_service = build('drive', 'v3', credentials=creds)

file_metadata = {'name': 'team_ratings.png'}
media = MediaFileUpload('C:/Users/NolanNicholls/Documents/NHL/2024/team_ratings.png', mimetype='image/png')

file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
print(f"File ID: {file.get('id')}")

file_id = file['id']  # This is the ID of the uploaded file from your previous code



completed_games.to_csv("C:/Users/NolanNicholls/Documents/NHL/2024/completed_games_for_test.csv")

if __name__ == '__main__':
    print('Cool!')

