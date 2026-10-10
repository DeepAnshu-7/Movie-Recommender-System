# Movie Recommender System

A content based movie recommender system made with Python and Streamlit.
Select a movie and it shows 5 similar movies with their posters (posters come from the TMDB API).

## How it works

- Dataset: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- Made a `tags` column for every movie using overview, genres, keywords, top 3 cast and director
- Converted tags into vectors using CountVectorizer (5000 words)
- Used cosine similarity to find the 5 most similar movies

## Files

- `project.ipynb` - data cleaning, preprocessing and model
- `MRproject.py` - Streamlit app
- `movie_list.csv` - processed movies data (made by the notebook)

## How to run

1. Install the libraries

```
pip install -r requirements.txt
```

2. Get a free API key from [TMDB](https://www.themoviedb.org/settings/api) and put it in `.streamlit/secrets.toml`

```
TMDB_API_KEY = "your_api_key"
```

3. Run the app

```
streamlit run MRproject.py
```

To run the notebook again, download `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` from Kaggle and keep them in this folder.

## Tech used

Python, Pandas, Scikit-learn, Streamlit, TMDB API
