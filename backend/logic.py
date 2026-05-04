def _similarity_reason(book_row: pd.Series, query_row: pd.Series) -> str:
    """
    Generate a short human-readable reason two books are similar.
    """
    reasons = []
    if book_row["author"].lower() == query_row["author"].lower() and book_row["author"]:
        reasons.append(f"same author ({book_row['author']})")
    if book_row["genre"].lower() == query_row["genre"].lower() and book_row["genre"]:
        reasons.append(f"same genre ({book_row['genre']})")
    if not reasons:
        reasons.append("similar themes and style")
    return "Recommended because: " + ", ".join(reasons)
 
 
def recommend(
    title: str,
    df: pd.DataFrame,
    sim_matrix,
    genre_filter: str | None = None,
    top_n: int = 2,
) -> list[dict]:
    """
    Return *top_n* book recommendations for *title*.
 
    Parameters
    ----------
    title        : exact title as it appears in df (use match_title first)
    df           : cleaned books DataFrame
    sim_matrix   : pre-computed cosine-similarity matrix
    genre_filter : optional genre string; if provided only books of that genre
                   are returned
    top_n        : number of results to return
 
    Returns
    -------
    List of dicts with keys: title, author, genre, description, score, reason, tags
    """
    titles_lower = df["title"].str.lower().tolist()
    try:
        idx = titles_lower.index(title.lower())
    except ValueError:
        return []
 
    query_row   = df.iloc[idx]
    scores      = list(enumerate(sim_matrix[idx]))
    scores      = sorted(scores, key=lambda x: x[1], reverse=True)
 
    results = []
    for book_idx, score in scores:
        if book_idx == idx:             
            continue
        row = df.iloc[book_idx]
 
   
        if genre_filter and genre_filter.lower() not in row["genre"].lower():
            continue
 
        reason = _similarity_reason(row, query_row)
        tags   = _generate_tags(row)
 
        results.append({
            "title":       row["title"],
            "author":      row["author"],
            "genre":       row["genre"],
            "description": row["description"][:200] + "…" if len(row["description"]) > 200 else row["description"],
            "score":       round(float(score), 4),
            "reason":      reason,
            "tags":        tags,
        })
 
        if len(results) >= top_n:
            break
 
    return results
 
 
def _generate_tags(row: pd.Series) -> list[str]:
    """Extract 2-4 short tags from genre and description keywords."""
    tags = []
    if row["genre"]:
        tags.append(row["genre"])
    keywords = ["adventure", "romance", "mystery", "dark", "epic",
                "historical", "dystopia", "magic", "thriller", "science"]
    desc_lower = row["description"].lower()
    for kw in keywords:
        if kw in desc_lower and kw not in [t.lower() for t in tags]:
            tags.append(kw.capitalize())
        if len(tags) >= 4:
            break
    return tags
 
