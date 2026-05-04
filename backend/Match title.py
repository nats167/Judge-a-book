def match_title(query: str, df: pd.DataFrame, n: int = 1, cutoff: float = 0.3) -> str | None:
    """
    Fuzzy-match *query* against the dataset titles using difflib.
 
    Returns the best-matching title string, or None if nothing passes *cutoff*.
    """
    titles  = df["title"].str.lower().tolist()
    matches = difflib.get_close_matches(
        query.lower(), titles, n=n, cutoff=cutoff
    )
    if not matches:
        return None

    idx = titles.index(matches[0])
    return df.iloc[idx]["title"]
 
