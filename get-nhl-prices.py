import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
import json
from datetime import datetime, timedelta
import pytz

# this file generates all games between today and begining of season

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

def request_data():
    response = requests.get(
        'https://api.ubiqanalytics.com/eventPriceStamps?source=PS3838&sport=Hockey&eventStartTimeFrom=2024-10-03',
        auth=HTTPBasicAuth('quant', 'FastBeefProfit3')
    )
    return response

def convert_time(time):

    utc_time = datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%fZ')
    
    # Convert UTC time to Eastern Time
    utc_zone = pytz.utc
    eastern_zone = pytz.timezone('US/Eastern')
    
    utc_time = utc_zone.localize(utc_time)  # Localize the time as UTC
    eastern_time = utc_time.astimezone(eastern_zone)  # Convert to Eastern Time
    
    # Format the Eastern Time as "YYYY-MM-DD"
    formatted_date = eastern_time.strftime('%Y-%m-%d')
    return formatted_date

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
    'CHI': 'Chicago Blackhawks'
}
# Get today and tommorows dates
today = datetime.now().date()
yday = today + timedelta(days=-1)
# Format the dates as 'YYYY-MM-DD'
formatted_today = today.strftime('%Y-%m-%d')
formatted_yday = yday.strftime('%Y-%m-%d')
gameData = request_game_data("2024-10-03", str(today))
rowData = []
for i in gameData:
    if i["gameState"] in ['FINAL', 'OFF']:
        rowData.append(parse_game_info(i))
games = pd.DataFrame(rowData, columns=["gid", "Season Type", "Neutral", "Date", "Away Team", "Away Score", "Home Team", "Home Score"])
games = games[games["Season Type"].isin([2,3])]
games = games[games["Date"] <= str(today)]
data = json.loads(request_data().text)
price_data = {}
for i in data:
    price_data[i['eventId']] = {'date':'','home_team':'','home_team_abv':'','away_team':'','away_team_abv':'','open_away_ml':0,'close_away_ml':0,'open_home_ml':0,'close_home_ml':0,'open_total_line':0,'open_total_over':0,'open_total_under':0,'close_total_line':0,'close_total_over':0,'close_total_under':0,'open_hc_line':0,'open_hc_away':0,'open_hc_home':0,'close_hc_line':0,'close_hc_away':0,'close_hc_home':0}
nhlGames = []
for i in data:  
    if i['marketType'] in (['Moneyline', 'OverUnder','AsianHandicap']) and i['outcomePeriod'] == 'Full' and i['stampType'] in (['Opening','Starting']):
        awayId = i['participants'][0]['id']
        homeId = i['participants'][1]['id']
        price_data[i['eventId']]['home_team'] = i['participants'][1]["name"]
        price_data[i['eventId']]['away_team'] = i['participants'][0]["name"]
        price_data[i['eventId']]['home_team_abv'] = i['participants'][1]['shortName']
        price_data[i['eventId']]['away_team_abv'] = i['participants'][0]['shortName']
        price_data[i['eventId']]["date"] = i['startTime']
        for j in i["selections"]:
            if j == 0:
                pass
            else:
                if j['key']['unit'] == "Goal":
                    if i["marketType"] == "Moneyline" or j['priority'] == 1:
                        if i['stampType'] == "Opening":
                            if i['marketType'] == 'OverUnder':
                                if j['key']['outcomes'][0]['outcomeType'] == "Over":
                                    price_data[i['eventId']]['open_total_line'] = j['key']['outcomes'][0]["line"]
                                    price_data[i['eventId']]['open_total_over'] = j["price"]
                                elif  j['key']['outcomes'][0]['outcomeType'] == "Under":
                                    price_data[i['eventId']]['open_total_under'] = j["price"]
                            elif j["key"]['outcomes'][0]['participantId'] == awayId:
                                if i['marketType'] == 'Moneyline':
                                    price_data[i['eventId']]['open_away_ml'] = j['price']
                                elif i['marketType'] == 'AsianHandicap':
                                    price_data[i['eventId']]['open_hc_line'] = j['key']['outcomes'][0]["line"]
                                    price_data[i['eventId']]['open_hc_away'] = j["price"]
                            elif j["key"]['outcomes'][0]['participantId'] == homeId:
                                if i['marketType'] == 'Moneyline':
                                    price_data[i['eventId']]['open_home_ml'] = j['price']
                                elif i['marketType'] == 'AsianHandicap':
                                    price_data[i['eventId']]['open_hc_home'] = j["price"]
                        elif i['stampType'] == "Starting":
                            if i['marketType'] == 'OverUnder':
                                if j['key']['outcomes'][0]['outcomeType'] == "Over":
                                    price_data[i['eventId']]['close_total_line'] = j['key']['outcomes'][0]["line"]
                                    price_data[i['eventId']]['close_total_over'] = j["price"]
                                elif  j['key']['outcomes'][0]['outcomeType'] == "Under":
                                        price_data[i['eventId']]['close_total_under'] = j["price"]
                            elif j["key"]['outcomes'][0]['participantId'] == awayId:
                                if i['marketType'] == 'Moneyline':
                                    price_data[i['eventId']]['close_away_ml'] = j['price']
                                elif i['marketType'] == 'AsianHandicap':
                                    price_data[i['eventId']]['close_hc_line'] = j['key']['outcomes'][0]["line"]
                                    price_data[i['eventId']]['close_hc_away'] = j["price"]               
                            elif j["key"]['outcomes'][0]['participantId'] == homeId:
                                if i['marketType'] == 'Moneyline':
                                    price_data[i['eventId']]['close_home_ml'] = j['price']
                                elif i['marketType'] == 'AsianHandicap':
                                    price_data[i['eventId']]['close_hc_home'] = j["price"]
df = pd.DataFrame.from_dict(price_data, orient='index').reset_index().rename({"index":"gid"},axis=1)
df['date'] = df['date'].apply(lambda x: convert_time(x[:-2] + "Z"))
df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
df = df.replace({'NAS':"NSH", 'TAM':"TBL", 'UTA':"ARI", 'WAS':"WSH", 'WNP':"WPG", "Utah Hockey Club":"ARI"})
games['map'] = games.apply(lambda x: "{}-{}-{}".format(x["Date"], x["Away Team"], x["Home Team"]), axis=1)
df['map'] = df.apply(lambda x: "{}-{}-{}".format(x["date"], x["away_team_abv"], x["home_team_abv"]), axis=1)

games = games.merge(df[['gid', 'open_away_ml', 'close_away_ml', 'open_home_ml',
       'close_home_ml', 'open_total_line', 'open_total_over',
       'open_total_under', 'close_total_line', 'close_total_over',
       'close_total_under', 'open_hc_line', 'open_hc_away', 'open_hc_home',
       'close_hc_line', 'close_hc_away', 'close_hc_home', 'map']],on='map',how='left', suffixes=("_nhl",'_ubiq'))

games.to_csv("price-export-2024.csv")