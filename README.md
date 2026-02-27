# movie_recommendation_system
Movie Recommendation System

This project implements a movie recommendation system using the MovieLens dataset.

The goal is to explore and compare different recommendation techniques and present the results in an interactive Dash application.

Features

Genre-based similarity using cosine similarity

Improved recommendations using TF-IDF on movie tags

Interactive Dash interface

Modular design separating model logic and UI

Dataset

The project uses the MovieLens dataset from GroupLens.

Download the dataset (ml-latest.zip) from:
https://grouplens.org/datasets/movielens/

Extract the files into:

data/raw/

The dataset is not included in this repository.

Installation

Create a virtual environment and install dependencies:

pip install -r requirements.txt
Run the application
python app/app.py

The Dash server will start locally and open in your browser.

Project Structure

src/ – data processing, feature engineering, recommendation logic

app/ – Dash user interface

notebooks/ – exploratory analysis

reports/ – written report and documentation

Purpose
This project was developed as part of a Machine Learning course and demonstrates feature engineering, similarity metrics, and model evaluation in a recommendation system context.

This project was developed as part of a Machine Learning course and demonstrates feature engineering, similarity metrics, and model evaluation in a recommendation system context.

