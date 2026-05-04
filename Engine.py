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
