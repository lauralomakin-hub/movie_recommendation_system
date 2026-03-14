# movie_recommendation_system
Movie Recommendation System

This project implements a movie recommendation system using the MovieLens dataset.

The goal is to explore and compare different recommendation techniques and present the results in an interactive Dash application.

## Features

-Interactive Dash interface
-Modular design separating model logic and UI
-Content-based movie recommendations

## Recommendation Methods

The project explores different recommendation techniques:

-Genre similarity using cosine similarity on movie genres
-Tag-based similarity using TF-IDF features extracted from movie tags

These methods allow comparison between simple content-based filtering and more feature-rich approaches.

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

src/ – data processing, feature engineering, recommendation logic

app/ – Dash user interface

notebooks/ – exploratory analysis

reports/ – written report and documentation

## Purpose
This project was developed as part of a Machine Learning course and demonstrates feature engineering, similarity metrics, and model evaluation in a recommendation system context.


