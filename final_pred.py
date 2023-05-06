import urllib.request
from datetime import datetime
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import csv
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import re
from datetime import datetime, timedelta

# ISW DATA ///////////////////////////////////////////////////////////////////////////////////////

data = []

date = datetime.today() - pd.Timedelta(days=1)
while True:
    url = f"https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-{date.strftime('%B').lower()}-{date.strftime('%#d')}-{date.year}"
    response = requests.get(url)
    if response.status_code == 200:
        break
    date -= timedelta(days=1)
    if date.year != datetime.today().year:
        print("No valid links found")
        break

soup = bs(response.content, 'html.parser')
description = soup.find("div", {"class": "field-name-body"}).text
description = description.replace(" dot ", ".")
description = re.sub(r'\[1\].*?\[1\]', '', description)
description = re.sub(r'\[.*?\]', '', description)
description = re.sub(r'http\S+', '', description)
description = description.lower()
description = description.replace("\n", " ")
description = description.replace("\xa0", " ")
pattern = re.compile('[^A-Za-z0-9\s]')
description = pattern.sub("", description)
description = re.sub(r'^.*?(est|et)', '', description, flags=re.DOTALL)
description = description.replace(
    "click here to see isws interactive map of the russian invasion of ukraine this map is updated daily alongside the static maps present in this report",
    " ")
data.append({"Date": date.strftime("%B-%d-%Y").lower(), "Description": description})

df1 = pd.DataFrame(data)

# VECTORIZATION

# Read column names from file
with open('cols_vect.txt', 'r') as f:
    columns = [line.strip().split(',') for line in f.readlines()]

print(columns)

# Create dataframe with zero values
df2 = pd.DataFrame(0, index=range(12), columns=columns)

print(df2.head())
texts = df1["Description"].tolist()

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Preprocess the texts
processed_texts = []
for text in texts:
    # Tokenize the text into words
    words = word_tokenize(text)

    # Stem and lemmatize the words
    stemmed_words = [stemmer.stem(word) for word in words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]

    # Join the processed words back into a text
    processed_text = " ".join(lemmatized_words)
    processed_texts.append(processed_text)

print(processed_texts)

for word in processed_texts[0].split():
    if word in columns[0]:
        df2.loc[:, word] += 1
    else:
        continue

print(df2['isw'])

# REGIONS ////////////////////////////////////////////////////////////////////////////////////////

regions = pd.read_csv('regions.csv')
p = regions['center_city_en'].tolist()


# WEATHER ////////////////////////////////////////////////////////////////////////////////////////

def getNextDate(DATE):
    time = str(datetime.now().time())
    time = int(time[:2])
    if time < 12:
        return str(DATE)
    DATE = DATE + timedelta(days=1)
    return str(DATE)


#
#
def getHourWeather(LOCATION, DATE):
    requestUrl = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/" + LOCATION + "/" + DATE + "/" + NEXT_DAY + "?unitGroup=metric&key=4B3ADASLF385RSZVJ7VLZ3W87&include=hours&elements=tempmax,tempmin,temp,dew,humidity,precip,uvindex,moonphase,snow,windgust,winddir,pressure"
    req = urllib.request.urlopen(requestUrl)
    rawForecastData = req.read()
    req.close()
    return json.loads(rawForecastData)


#
#
def json_to_df(json_obj):
    df = pd.DataFrame()
    df['queryCost'] = [json_obj['queryCost']]
    df['latitude'] = [json_obj['latitude']]
    df['longitude'] = [json_obj['longitude']]
    df['resolvedAddress'] = [json_obj['resolvedAddress']]
    df['address'] = [json_obj['address']]
    df['timezone'] = [json_obj['timezone']]
    df['tzoffset'] = [json_obj['tzoffset']]

    days_data = []
    for day in json_obj['days']:
        day_dict = {}
        day_dict.update(day)
        for hour in day_dict['hours']:
            hour_dict = {}
            hour_dict.update(hour)
            hour_dict['day_tempmax'] = day_dict['tempmax']
            hour_dict['day_tempmin'] = day_dict['tempmin']
            hour_dict['day_temp'] = day_dict['temp']
            hour_dict['day_dew'] = day_dict['dew']
            hour_dict['day_humidity'] = day_dict['humidity']
            hour_dict['day_precip'] = day_dict['precip']
            hour_dict['day_snow'] = day_dict['snow']
            hour_dict['day_windgust'] = day_dict['windgust']
            hour_dict['day_winddir'] = day_dict['winddir']
            hour_dict['day_pressure'] = day_dict['pressure']
            hour_dict['day_uvindex'] = day_dict['uvindex']
            hour_dict['day_moonphase'] = day_dict['moonphase']
            days_data.append(hour_dict)

    days_df = pd.DataFrame(days_data)
    df = pd.concat([df] * len(days_df), ignore_index=True)
    df = pd.concat([df, days_df], axis=1)
    df = df.drop(
        ["queryCost", "latitude", "longitude", "resolvedAddress", "address", "timezone", "tzoffset", 'day_snow',
         'day_windgust', 'day_winddir', 'day_pressure', 'precip', 'uvindex'], axis=1)
    df.rename(columns={"temp": "hour_temp", "humidity": "hour_humidity", "dew": "hour_dew", "snow": "hour_snow",
                       "windgust": "hour_windgust", "winddir": "hour_winddir", "pressure": "hour_pressure"},
              inplace=True)

    return df


#
#
def get12HoursForecast(df):
    time = str(datetime.now().time())
    time = int(time[:2])
    endTime = time + 12
    df2 = df.iloc[time:endTime]
    return df2


#
#
def getRowIndex(LOCATION):
    index = regions.index[regions['center_city_en'] == LOCATION].tolist()[0]
    return index


# FINAL DATASET //////////////////////////////////////////////////////////////////////////////////


final_result = []
for location in p:
    if location not in ["Lviv", "Sumy"]:
        API_KEY = "2CWS9WJ85YWQNSTCRALV7ME6Q"
        LOCATION = location
        DATE = datetime.now().date()
        NEXT_DAY = getNextDate(DATE)
        DATE = str(DATE)
        df = getHourWeather(LOCATION, DATE)
        df = json_to_df(df)
        p = getRowIndex(location)
        region_id = regions.at[regions.index[p], 'region_id']
        df_weather = get12HoursForecast(df)
        df_isw = df2
        df_weather = df_weather.reset_index()
        df_isw['id_col'] = range(len(df_weather))
        dataset1 = pd.DataFrame(df_isw)
        dataset2 = pd.DataFrame(df_weather)
        result = pd.concat([df_isw, df_weather], axis=1)
        result.columns = result.columns.map(lambda x: str(x).strip("(),'"))
        result['region_id'] = region_id
        result = result.drop(columns='id_col', axis=1)
        result = result.drop(columns='index', axis=1)
        print(result)
        final_result.append(result)

final_df = pd.concat(final_result, axis=0, ignore_index=True)

final_json = final_df.to_json(orient='records')
