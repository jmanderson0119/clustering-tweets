import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#loading data
input_file = "../data/cnnhealth.txt"   

with open(input_file, "r", encoding="utf-8") as f:
    lines = [re.sub(r"\|\s+", " ", line) for line in f]

df = pd.read_csv(pd.io.common.StringIO("".join(lines)), sep="|", header=None)
df.columns = ["id", "timestamp", "text"]

#remove duplicates
df["urls"] = df["text"].apply(lambda x: re.findall(r"http\S+|www\S+", str(x)))

# keep only the first tweet per unique URL
url_to_keep = {}
indices_to_drop = set()

for idx, url_list in enumerate(df["urls"]):
    for url in url_list:
        if url in url_to_keep:
            indices_to_drop.add(idx)  # duplicate URL, mark for removal
        else:
            url_to_keep[url] = idx    # first occurrence

df = df.drop(index=list(indices_to_drop)).reset_index(drop=True)

# remove temporary 'urls' column
df = df.drop(columns=["urls"])

print(f"Removed {len(indices_to_drop)} tweets due to duplicate URLs. Remaining: {len(df)}")
#cleaning

def clean_text(text):
    text = str(text)
    
    # remove leading 'RT ' at the beginning
    text = re.sub(r"^\s*rt\s+", "", text, flags=re.IGNORECASE)
    
    # remove all other instances of ' rt ' (with spaces around)
    text = re.sub(r"\s+rt\s+", " ", text, flags=re.IGNORECASE)
    
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
    
    # remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

df["clean_text"] = df["text"].apply(clean_text)
before = len(df)
df = df.drop_duplicates(subset="clean_text", keep="first").reset_index(drop=True)
after = len(df)

#taking care of the trailing via
df["clean_text_stripped"] = df["clean_text"].str.replace(r"\s+via(\s+@\w+)?$", "", regex=True)

#removing dupes based on the via
before_dupes = len(df)
df = df.drop_duplicates(subset="clean_text_stripped", keep="first").reset_index(drop=True)
after_dupes = len(df)

print(f"Removed {before_dupes - after_dupes} duplicate tweets due to trailing 'via'. Remaining: {after_dupes}")
df = df.drop(columns=["clean_text_stripped"])

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
