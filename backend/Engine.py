import os
import pickle
import difflib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
 
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "books.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "similarity_matrix.pkl")
 
GENRES = [
    "Fiction", "Mystery", "Sci-Fi", "Fantasy",
    "Thriller", "Romance", "Biography",
    "Self-Help", "History", "Horror",
]


 
def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the books CSV and return a clean DataFrame.
 
    Expected columns (case-insensitive):
        title, author, genre / category, description
    """
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.lower()
 
    rename_map = {}
    if "category" in df.columns and "genre" not in df.columns:
        rename_map["category"] = "genre"
    df.rename(columns=rename_map, inplace=True)
 
    required = ["title", "author", "genre", "description"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    for col in required:
        df[col] = df[col].fillna("").astype(str).str.strip()
 
    df = df[df["title"] != ""].reset_index(drop=True)
 
    return df
 
 
def build_feature_strings(df: pd.DataFrame) -> pd.Series:
    """
    Combine title + author + genre + description into a single string
    per book for vectorisation.
    """

    return (
        df["title"]       + " " + df["title"]       + " "   # ×2
        + df["author"]    + " "
        + df["genre"]     + " " + df["genre"]        + " "   # ×2
        + df["description"]
    )
 
 
def build_similarity_matrix(df: pd.DataFrame) -> tuple:
    """
    Fit a TF-IDF vectoriser on the combined feature strings and
    compute the full cosine-similarity matrix.
 
    Returns
    -------
    (vectorizer, tfidf_matrix, similarity_matrix)
    """
    features   = build_feature_strings(df)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),  
        max_features=20_000,
    )
    tfidf_matrix   = vectorizer.fit_transform(features)
    sim_matrix     = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return vectorizer, tfidf_matrix, sim_matrix
 
 
def save_model(sim_matrix, vectorizer, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"sim_matrix": sim_matrix, "vectorizer": vectorizer}, f)
    print(f"[engine] Model saved → {path}")
 
 
def load_model(path: str = MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
 
 
