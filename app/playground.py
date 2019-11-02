

import os

import pandas as pd
import re
from nltk.corpus import stopwords
import spacy

DATA_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data")
REVIEWS_CSV_FILEPATH = os.path.join(DATA_DIRPATH, "amazon_reviews.csv")

MATCHING_PATTERN = r'[^a-zA-Z ^0-9]'

nlp = spacy.load("en_core_web_md")

def tokenize(text):
    text = text.lower()
    text = re.sub(MATCHING_PATTERN, '', text)
    
    doc = nlp(text)  #> <class 'spacy.tokens.doc.Doc'>

    tokens = [token.lemma_ for token in doc if token.is_stop == False]
    
    #stop_words = stopwords.words("english")
    #tokens = [token for token in tokens if token not in stop_words]

    return tokens


if __name__ == "__main__":
    df = pd.read_csv(REVIEWS_CSV_FILEPATH)

    print(df.head())

    #breakpoint()

    example_test = df["reviews.text"][0]
    print("TEXT", example_test)

    tokens = tokenize(example_test)
    print("TOKENS", tokens)
