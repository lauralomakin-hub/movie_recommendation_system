# Data handling
import numpy as np
import pandas as pd

# Preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

# Similarity
from sklearn.metrics.pairwise import cosine_similarity as cosine_sim

# String matching

import re

from pathlib import Path



def normalize_title(title):
    title = title.lower()
    title = title.replace(" ", "")
    title = re.sub(r"\(\d{4}\)", "", title)   # Source: See References [2] below
    return title

# Source: See chagptGPT: Is not this path flexible? 
# (see pathway to data in 03_recommendation_systems.ipynb)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"

movies = pd.read_csv(DATA_DIR / "movies.csv")
ratings = pd.read_csv(DATA_DIR / "ratings.csv")
tags = pd.read_csv(DATA_DIR / "tags.csv")


# Source: See References [3] in 03_recommendation_systems.ipynb
tags_per_movie = (
    tags.groupby("movieId")["tag"]
    .apply(lambda x: " ".join(x.dropna().astype(str)))
    .reset_index()
    .rename(columns={"tag": "tags_combined"})
)


movies = movies.merge(tags_per_movie, on="movieId", how="left")

#fill missing tags
movies["tags_combined"] = movies["tags_combined"].fillna("")



#clean title
movies["clean_title"] = movies["title"].apply(normalize_title)

#create new content column with genres and tags:            Source: See References [4] below
movies["genres_clean"] = (
    movies["genres"]
    .str.replace("(no genres listed)", "", regex=False)
    .str.replace("|", " ", regex=False)
    .str.replace("-", "", regex=False)
    .str.lower()
)

movies["tags_clean"] = (
    movies["tags_combined"]
    .str.replace("-", "", regex=False)
    .str.lower()
)

movies["content_with_tags"] = (
    movies["genres_clean"]
    + " "
    + movies["tags_clean"].str.replace(r"\b\d+\b", 
    "", 
    regex=True
    )
)

movies["content_with_tags"] = (
    movies["content_with_tags"]
    .str.replace(r"[^a-zA-Z\s]", "", regex=True)   # Only keep latin letters and spaces - chatGPT for regex
    .str.replace(r"\s+", " ", regex=True)          # Replace multiple spaces with a single space
    .str.strip()                                   # Remove leading and trailing spaces
)


#  TF-IDF vectorization on combined genres + tags
tfidf_tags = TfidfVectorizer(stop_words="english")
tfidf_matrix_tags = tfidf_tags.fit_transform(movies["content_with_tags"])
feature_names = tfidf_tags.get_feature_names_out()



# Source: See References [5] in 03_recommendation_systems.ipynb

# Calculate the number of ratings per user
user_counts = ratings.groupby("userId").size().reset_index(name="user_rating_count")

#merge in ratings
ratings = ratings.merge(user_counts, on="userId", how="left") 

#create userweight
ratings["user_weight"] = 1 / np.log1p(ratings["user_rating_count"])

# filter out users with less than 5 ratings
ratings = ratings[ratings["user_rating_count"] > 5]

#create weighted rating per row
ratings["weighted_user_rating"] = ratings["rating"] * ratings["user_weight"]


ratings_summary = (
    ratings.groupby("movieId")
    .agg(
        weighted_sum = ("weighted_user_rating", "sum"),
        weight_total = ("user_weight", "sum"),
        rating_count = ("rating", "count")
    )
    .reset_index()
)

ratings_summary["average_rating"] = (
    ratings_summary["weighted_sum"] / ratings_summary["weight_total"]
)

# Merge the average ratings back to the movies DataFrame, 
# Making sure that we don't overwrite existing columns if they exist

movies = movies.drop(
    columns = ["average_rating", "rating_count", "weighted_rating"],
    errors = "ignore"
)

movies = movies.merge(
    ratings_summary[["movieId", "average_rating", "rating_count" ]],
    on= "movieId",
    how = "left"
)

movies["average_rating"] = movies["average_rating"].fillna(0)
movies["rating_count"] = movies["rating_count"].fillna(0)


# Calculate the mean average rating across all movies. 
# c = the mean average rating across all movies
# m= the 90th percentile of the rating counts across all movies

c = movies["average_rating"].mean()

m = movies["rating_count"].quantile(0.90)

movies["weighted_rating"] = (
    (movies["rating_count"] / (movies["rating_count"] + m)) * movies["average_rating"]
    + (m / (movies["rating_count"] + m)) * c
)



# final recommendation function with reranking
def recommend_movies_with_reranking(movie_title, n=5, movie_pool=20):
    
    # find matching movies
    matches = movies[movies["title"].str.contains(
        movie_title,
        case=False,
        na=False,
        regex=False
    )]

    if len(matches) == 0:
        print("Movie could not be found")
        return None
    
    if len(matches) > 1:
        print("Multiple movies found. Using the first match: ")
        print(matches[["title"]].head(3))    

    movie_index = matches.index[0]

    title = movies.loc[movie_index, "title"]
    
    # calculate similarity
    similarity_score = cosine_sim(
        tfidf_matrix_tags[movie_index],
        tfidf_matrix_tags
    ).flatten()

    # get most similar movies
    sorted_movie_indexes = similarity_score.argsort()[::-1]
    movie_index_no_input = [movie_idx for movie_idx in sorted_movie_indexes if movie_idx != movie_index]

    candidate_indexes = movie_index_no_input[:movie_pool]

    # create result dataframe
    result = movies.iloc[candidate_indexes][
        [
            "title",
            "genres",
            "tags_combined",
            "average_rating",
            "rating_count",
            "weighted_rating"
        ]
    ].copy()

    # add similarity score to the result df
    result["similarity_score"] = similarity_score[candidate_indexes]

    # normalize rating to 0–1 safely
    min_rating = result["weighted_rating"].min()
    max_rating = result["weighted_rating"].max()

    if max_rating == min_rating:
        result["weighted_rating_norm"] = 0
    else:
        result["weighted_rating_norm"] = (
            (result["weighted_rating"] - min_rating) /
            (max_rating - min_rating)
        )

    # combine similarity, rating and popularity 
    # [8]
    result["final_score"] = (
        0.7 * result["similarity_score"] +
        0.2 * result["weighted_rating_norm"] +
        0.1 * np.log1p(result["rating_count"])
    )

    # sort by final score
    result = result.sort_values("final_score", ascending=False)


    # apply threshold to separate "safe" and "hidden" recommendations
    # just trying out different thresholds here, can be adjusted based on the distribution of rating counts in the dataset
    threshold = 75     

    # save the full ranked list before applying the threshold
    #chatGPT: What do I do if the result does not gve me 5 movies?
    full_ranked = result.copy()  
    safe = result[result["rating_count"] >= threshold].head(3)


    hidden_pool = result[
        (result["rating_count"] < threshold) &
        (result["rating_count"] >= 5)
    ]

    hidden = result[
        (result["rating_count"] < threshold) &
        (result["rating_count"] >= 5)
    ].sort_values("final_score", ascending=False).head(2)

    hidden = hidden_pool.sort_values("final_score", ascending=False).head(2)


    # combine safe and hidden
    result = pd.concat([safe, hidden])

    
    # if there are not enough safe recommendations
    # fill the remaining slots with hidden ones
    #chatGPT: What do I do if the result does not gve me 5 movies?
    if len(result) < n:
        extra = full_ranked[~full_ranked["title"].isin(result["title"])]
        result = pd.concat([result, extra]).head(n)
        
    # remove duplicate titles
    already_seen = set()
    unique_results = []

    for _, row in result.iterrows():
        name = row["title"].split("(")[0].strip().lower()
        
        if name not in already_seen:
            unique_results.append(row)
            already_seen.add(name)

        if len(unique_results) == n:
            break

    # if deduplication reduced the list below n, fill again from full ranked list
    #chatGPT: What do I do if the result does not gve me 5 movies?
    if len(unique_results) < n:
        for _, row in full_ranked.iterrows():
            name = row["title"].split("(")[0].strip().lower()

            if name not in already_seen:
                unique_results.append(row)
                already_seen.add(name)

            if len(unique_results) == n:
                break
      
    result = pd.DataFrame(unique_results)

    return result


def main():
    print("starting recommendation process...")
    result = recommend_movies_with_reranking(
        "gladiator", 
        n=5, 
        movie_pool=20   ####IS this the right varaiable name??
    )
    print("recommendation process completed.")

    if result is not None:
        print("Recommended movies:  ")

        print(result[
        [
            "title",
            "genres", 
            "tags_combined", 
            "average_rating", 
            "rating_count", 
            "weighted_rating", 
            "similarity_score", 
            "final_score"
        ]
    ])
    
    else:
        print("No result found.")


 
if __name__ == "__main__":
    print("Running the movie recommendation system...")
    main()
