import requests
from bs4 import BeautifulSoup
import pandas as pd

#assumes 2024-10-04 date format
def get_goals(home_abv, away_abv, date, gid):
    
    splitDate = date.split("-")
    bbrefGID = f'{splitDate[0]}{splitDate[1]}{splitDate[2]}0{home_abv.upper()}'
    print(bbrefGID)

    url = "https://www.hockey-reference.com/boxscores/{}.html".format(bbrefGID)
    
    response = requests.get(url)
    print(response.status_code)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    def scrape_table(soup, table_id):
        table = soup.find('table', id=table_id)
        
        if table:
            headers = [th.get_text() for th in table.find_all('thead')[0].find_all('tr')[1].find_all('th')]
            rows = table.find_all('tr')[1:]  
            data = []
            for row in rows:
                cols = row.find_all('td')
                data.append([col.get_text() for col in cols])
            
            df = pd.DataFrame(data, columns=headers[1:])
            df = df.loc[:,~df.columns.duplicated()]
            df = df[["EV","PP", "SH"]].tail(1)
            
            return df
        else:
            return pd.DataFrame() 
    
    home_skaters = scrape_table(soup, '{}_skaters'.format(home_abv.upper()))
    away_skaters = scrape_table(soup, '{}_skaters'.format(away_abv.upper()))
    
    home_skaters["gid"] = gid
    away_skaters["gid"] = gid
    
    return home_skaters.merge(away_skaters, on='gid', how='left', suffixes=('_home_goals', '_away_goals'))

