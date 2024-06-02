import requests
import numpy as np
import pandas as pd
import json
from dotenv import load_dotenv

load_dotenv()
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/sentence-t5-base")


def get_director(id):
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer " + os.getenv("TMDB_API_KEY"),
    }
    url_credits = f"https://api.themoviedb.org/3/movie/{id}/credits?language=en-US"
    response = requests.get(url_credits, headers=headers)
    data_credits = response.json()

    for d in data_credits["crew"]:
        if d["job"] == "Director":
            return d["name"]


def get_film(id):
    url_movie = f"https://api.themoviedb.org/3/movie/{id}?language=en-US"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer " + os.getenv("TMDB_API_KEY"),
    }
    response = requests.get(url_movie, headers=headers)
    data = response.json()
    movies_keys_to_remove = [
        "backdrop_path",
        "belongs_to_collection",
        "homepage",
        "poster_path",
        "production_companies",
        "production_countries",
        "spoken_languages",
        "status",
        "tagline",
        "video",
    ]
    for k in movies_keys_to_remove:
        if k in data:
            data.pop(k)

    url_credits = f"https://api.themoviedb.org/3/movie/{id}/credits?language=en-US"
    response = requests.get(url_credits, headers=headers)
    data_credits = response.json()
    if "cast" not in data_credits:
        top_actors = []
    else:
        top_actors = data_credits["cast"][0:2]
    actors_keys_to_remove = [
        "adult",
        "known_for_department",
        "original_name",
        "cast_id",
        "credit_id",
        "order",
        "profile_path",
    ]

    for t in top_actors:
        for k in actors_keys_to_remove:
            if k in t:
                t.pop(k)

    data["actors"] = top_actors

    embeddings = model.encode(
        data["overview"] if len(data["overview"]) > 0 else data["title"]
    )
    data["overview_embedding"] = embeddings.tolist()
    data["director"] = get_director(id)
    return data


import tqdm

# %%
import json


def sample(ids, file_path):
    films = {}
    counter = 0
    for id in tqdm.tqdm(ids):
        id = int(id)
        film_data = get_film(id)
        if "overview" in film_data and film_data["overview"] != "":
            films[id] = film_data
        else:
            print(f"Film {id} has no overview")
        counter += 1
        if counter % 100 == 0:
            print(f"{counter} films sampled")
    with open(file_path, "w") as outfile:
        json.dump(films, outfile)


# %%
import pandas as pd

df = pd.read_csv("./ml-latest-small/links.csv")
df = df.dropna(axis=0, how="any")
df["tmdbId"] = df["tmdbId"].astype(int)
# %%
sample(df["tmdbId"].values, "./movielens_latest-small.json")
