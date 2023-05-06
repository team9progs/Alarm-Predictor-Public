import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import re

start_date = pd.Timestamp("2022-02-24")
end_date = pd.Timestamp("2023-01-25")
data = []
pos_of_missing_url = 0

missing_urls = [
    "https://www.understandingwar.org/backgrounder/russia-ukraine-warning-update-initial-russian-offensive-campaign-assessment",
    "https://www.understandingwar.org/backgrounder/russia-ukraine-warning-update-russian-offensive-campaign-assessment-february-25-2022",
    "https://understandingwar.org/backgrounder/russia-ukraine-warning-update-russian-offensive-campaign-assessment-february-26",
    "https://www.understandingwar.org/backgrounder/russia-ukraine-warning-update-russian-offensive-campaign-assessment-february-27",
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-28-2022",
    "https://www.understandingwar.org/backgrounder/russian-campaign-assessment-may-5",
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-update-july-11",
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-august-12-0"
]

for date in pd.date_range(start_date, end_date):

    year = date.year
    month = date.strftime("%B").lower()
    day = date.strftime("%#d")
    if year == 2022:
        date_string = f"{month}-{day}"
    if year == 2023:
        date_string = f"{month}-{day}-{year}"
    url = f"https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-{date_string}"

    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"{date} data is not available. Searching internally...")
            if pos_of_missing_url < 8:
                response = requests.get(missing_urls[pos_of_missing_url])
                pos_of_missing_url += 1
                if response.status_code != 200:
                    print(f"{date} data is not available.")
                    continue
                else:
                    print(f"Info for {date} found")
            else:
                print(f"No link founded for {date}")
                continue
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
            "")
        date_final = date + pd.Timedelta(days=1)
        data.append({"Date": date_final.strftime("%Y-%m-%d").lower(), "Description": description})
    except (requests.exceptions.RequestException, KeyError, TypeError) as error:
        print(f"{error} for {date}")
    continue

df = pd.DataFrame(data)
df.to_csv("isw_data.csv", index=False)

