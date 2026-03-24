import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

#loading data
input_file = "cnnhealth.txt"   

with open(input_file, "r", encoding="utf-8") as f:
    lines = [re.sub(r"\|\s+", " ", line) for line in f]

df = pd.read_csv(pd.io.common.StringIO("".join(lines)), sep="|", header=None)
df.columns = ["id", "timestamp", "text"]
#cleaning

def clean_text(text):
    text = str(text)
    
    # remove leading 'RT' (retweet)
    text = re.sub(r"^\s*rt\s+", "", text, flags=re.IGNORECASE)
    
    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # remove mentions
    text = re.sub(r"@\w+", "", text)
    
    # remove hashtag symbol but keep the word
    text = re.sub(r"#", "", text)
    
    # remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # convert to lowercase
    text = text.lower()
    
    return text

df["clean_text"] = df["text"].apply(clean_text)


#vectorization

USE_TFIDF = True   #set to false for raw counts

if USE_TFIDF:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=2              # ignore very rare words
    )
else:
    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )

X = vectorizer.fit_transform(df["clean_text"])

mask = (X.sum(axis=1) > 0).A1
X = X[mask]
df = df[mask].reset_index(drop=True)

#step 4: Save outputs
from scipy.sparse import save_npz
from scipy.io import mmwrite
save_npz("tweet_features.npz", X)
mmwrite("tweet_features.mtx", X)

df["clean_text"].to_csv("cleaned_tweets.txt", index=False, header=False)

#a few sanity checks


feature_names = vectorizer.get_feature_names_out()
row_sums = X.sum(axis=1)
num_docs, num_features = X.shape
avg_nonzero = X.nnz / num_docs

# sparsity
sparsity = 1 - (X.nnz / (num_docs * num_features))

print("Documents:", num_docs)
print("Features:", num_features)
print("Avg nonzeros per doc:", avg_nonzero)
print("Sparsity:", sparsity)
