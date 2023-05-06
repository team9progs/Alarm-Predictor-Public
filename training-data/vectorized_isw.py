import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import csv
import pandas as pd

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

df = pd.read_csv("data/isw_data.csv")
texts = df["Description"].tolist()

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

# Vectorize the processed texts
vectorizer = CountVectorizer()
vectorized_texts = vectorizer.fit_transform(processed_texts)
features = vectorizer.get_feature_names()

# Create a list of dictionaries representing each observation
observations = []
for i in range(len(processed_texts)):
    obs_dict = {}
    for j in range(len(features)):
        obs_dict[features[j]] = vectorized_texts[i,j]
    observations.append(obs_dict)

# Create a DataFrame from the list of dictionaries
df_vectorized = pd.DataFrame(observations)
df_vectorized['day_datetime'] = df['Date']

# Add the original text and processed text columns to the vectorized DataFrame
df_vectorized["text"] = texts
df_vectorized["processed_text"] = processed_texts

# Write the DataFrame to a CSV file
df_vectorized.to_csv("vectorized_texts.csv", index=False)
