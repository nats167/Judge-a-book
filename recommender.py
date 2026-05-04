"""
Judge-a-Book: ML-based Book Recommendation Engine
Uses TF-IDF + Cosine Similarity + OCR to recommend books from a cover image.
"""

import pandas as pd
import numpy as np
import pickle
import os
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_and_clean_data(filepath="data.csv"):
    """Load the books dataset and clean it."""
    df = pd.read_csv(filepath)

    
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    for col in ["title", "authors", "categories", "description"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
        else:
            df[col] = ""


    df = df[df["title"].str.strip() != ""].reset_index(drop=True)

  
    df["combined_features"] = (
        df["title"] + " " +
        df["authors"] + " " +
        df["categories"] + " " +
        df["description"]
    )

    return df


def build_similarity_matrix(df, save_path="similarity_matrix.pkl"):
    """Compute TF-IDF matrix and cosine similarity, then save to disk."""
    print("Building TF-IDF matrix...")
    tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
    tfidf_matrix = tfidf.fit_transform(df["combined_features"])

    print("Computing cosine similarity...")
    similarity = cosine_similarity(tfidf_matrix)

    with open(save_path, "wb") as f:
        pickle.dump(similarity, f)

    print(f"Similarity matrix saved to {save_path}")
    return similarity


def load_similarity_matrix(path="similarity_matrix.pkl"):
    """Load pre-computed similarity matrix from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def match_title(query, df, n=1):
    """
    Use difflib to fuzzy-match an OCR-extracted title against the dataset.
    Returns the best-matching row index and title.
    """
    titles = df["title"].str.lower().tolist()
    matches = difflib.get_close_matches(query.lower(), titles, n=n, cutoff=0.4)

    if not matches:
        return None, None

    best_match = matches[0]
    idx = df[df["title"].str.lower() == best_match].index[0]
    return idx, df.loc[idx, "title"]


def get_recommendations(title_query, df, similarity, genre_filter=None, top_n=5):
    """
    Given a book title query, return top-N similar books.
    Optionally filter by genre.
    """
    idx, matched_title = match_title(title_query, df)

    if idx is None:
        return None, f"No match found for '{title_query}'. Try a different title."

 
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    scores = [(i, s) for i, s in scores if i != idx]

  
    if genre_filter and genre_filter.lower() != "all":
        scores = [
            (i, s) for i, s in scores
            if genre_filter.lower() in df.loc[i, "categories"].lower()
        ]

    top_indices = [i for i, _ in scores[:top_n]]
    top_scores  = [s for _, s in scores[:top_n]]

    results = df.loc[top_indices, ["title", "authors", "categories", "description"]].copy()
    results["similarity_score"] = [round(s * 100, 1) for s in top_scores]
    results["similarity_reason"] = results.apply(_build_reason, axis=1)

    return matched_title, results.reset_index(drop=True)


def _build_reason(row):
    """Generate a short human-readable similarity reason."""
    parts = []
    if row["authors"]:
        parts.append(f"by {row['authors'].split(',')[0].strip()}")
    if row["categories"]:
        cat = row["categories"].split("/")[0].strip()
        parts.append(f"in {cat}")
    if row["description"]:
        snippet = row["description"][:60].rstrip()
        parts.append(f'— "{snippet}..."')
    return " ".join(parts) if parts else "Similar themes and style"



def extract_title_from_image(image_path):
    """
    Use Pytesseract + PIL to extract text from a book cover image.
    Returns the most likely title (top non-empty lines).
    """
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(image_path).convert("L")   
        text = pytesseract.image_to_string(img)

   
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        title_candidate = " ".join(lines[:3]) if lines else ""
        return title_candidate

    except ImportError:
        print("pytesseract or Pillow not installed. Falling back to manual input.")
        return ""
    except Exception as e:
        print(f"OCR error: {e}")
        return ""


if __name__ == "__main__":
    DATA_PATH   = "data.csv"
    MATRIX_PATH = "similarity_matrix.pkl"

    df = load_and_clean_data(DATA_PATH)

    if os.path.exists(MATRIX_PATH):
        similarity = load_similarity_matrix(MATRIX_PATH)
        print("Loaded pre-computed similarity matrix.")
    else:
        similarity = build_similarity_matrix(df, MATRIX_PATH)

    matched, recs = get_recommendations("The Hobbit", df, similarity, genre_filter="Fantasy", top_n=3)

    if recs is not None:
        print(f"\nBecause you read: {matched}\n")
        for _, row in recs.iterrows():
            print(f"  📚 {row['title']}  ({row['similarity_score']}% match)")
            print(f"     {row['similarity_reason']}\n")
