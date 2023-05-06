import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# datasets
weather = pd.read_csv("all_weather_by_hour.csv")
isw = pd.read_csv("isw_data.csv")
isw_vect = pd.read_csv("vectorized_texts.csv")
alarms = pd.read_csv("test.csv", sep=';')

# weather dataset
sns.scatterplot(data=weather, x="day_datetime", y="day_precip")

sns.scatterplot(data=weather, x="day_datetime", y="day_temp")

# isw dataset
sns.scatterplot(x=isw['Date'], y=isw_vect['rocket'])

isw_vect['civilian_infrastructure'] = np.minimum(isw_vect['civilian'], isw_vect['infrastructur'])
sns.scatterplot(x=isw['Date'], y=isw_vect['civilian_infrastructure'])

isw_vect['critical_infrastructure'] = np.minimum(isw_vect['critic'], isw_vect['infrastructur'])
sns.scatterplot(x=isw['Date'], y=isw_vect['critical_infrastructure'])

isw_vect['infrastructure'] = isw_vect['critical_infrastructure'] + isw_vect['civilian_infrastructure']
sns.scatterplot(x=isw['Date'], y=isw_vect['infrastructure'])

# alarms dataset

alarms['event_start'] = pd.to_datetime(alarms['event_start'])
alarms['event_end'] = pd.to_datetime(alarms['event_end'])
same_date_mask = alarms['event_start'].dt.date == alarms['event_end'].dt.date
previous_date_mask = alarms['event_start'].dt.date < alarms['event_end'].dt.date
alarms.loc[same_date_mask, 'duration'] = (alarms.loc[same_date_mask, 'event_end'] - alarms.loc[
    same_date_mask, 'event_start']).dt.total_seconds() / 3600
alarms.loc[previous_date_mask, 'duration'] = ((pd.to_datetime(
    alarms.loc[previous_date_mask, 'event_start'].dt.date) + pd.Timedelta(days=1)) - alarms.loc[
                                                  previous_date_mask, 'event_start']).dt.total_seconds() / 3600
alarms = pd.concat([alarms.loc[same_date_mask], alarms.loc[previous_date_mask]], ignore_index=True)
sns.scatterplot(x=alarms['event_start'].dt.date, y=alarms['duration'])
