# movie_recommendation_system
Movie Recommendation System

This project implements a movie recommendation system using the MovieLens dataset.

The goal is to explore and compare different recommendation techniques and present the results in an interactive Dash application.

## Planned features

-Interactive Dash interface
-Content-based movie recommendations

## Recommendation Methods

The project explores multiple recommendation approaches with increasing complexity:

- Genre-based similarity using cosine similarity on movie genres (baseline model)
- Genre and tag-based similarity using TF-IDF to capture richer movie content

To improve recommendation quality, additional ranking strategies were introduced:

- User-weighted ratings to reduce the influence of highly active users (superusers)
- An IMDb-inspired Bayesian weighted rating to balance rating quality and rating count
- A final reranking step combining similarity, rating quality, and popularity

These steps allow comparison between a simple content-based model and a more robust hybrid approach that accounts for both content and rating behavior.

Planned extensions include:
- A collaborative filtering approach to incorporate user behavior patterns
- An interactive interface built with Dash to enable exploration and evaluation of recommendations


## Dataset

## Dataset

This project uses the **MovieLens "ml-latest" dataset**, which is recommended for education and development.

Download the dataset from:
https://grouplens.org/datasets/movielens/

Extract the files into:

data/raw/

Required files:

movies.csv
ratings.csv
tags.csv
links.csv

The genome files included in the dataset are not used in this project.


**The dataset is not included in this repository.**

## Installation

Install dependencies using uv:

uv sync
Run the application
uv run python app/app.py

The Dash server will start locally and open in your browser.

## Project Structure

notebooks/ – exploratory analysis, feature engineering, and model development

reports/ – written report and documentation

Future structure:
- src/ – for modularizing data processing and recommendation logic
- app/ – for a Dash-based user interface

## Purpose

This project was developed as part of a Machine Learning course and focuses on building and improving a movie recommendation system.

It demonstrates feature engineering, similarity-based recommendation methods, and evaluation of different modeling approaches. The project also handles challenges such as missing tag data, uneven rating distribution, and popularity bias through different weighting and fallback strategies.
