import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import argparse
import configparser
import os.path
import ast
from nltk.corpus import stopwords


def plotGenreGraph(df, png_loc):
    g = df.nlargest(columns="Count", n=50)
    plt.figure(figsize=(12, 15))
    ax = sns.barplot(data=g, x="Count", y="Genre")
    ax.set(ylabel='Count')
    plt.savefig(png_loc)


def visualizeGenreData(genres, png_loc):
    genres_list = list()
    for i in genres:
        genres_list.append(ast.literal_eval(i))
    all_genres = sum(genres_list, [])
    all_genres = nltk.FreqDist(all_genres)
    all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()),'Count': list(all_genres.values())})
    plotGenreGraph(all_genres_df, png_loc)
    return


def freq_words(x, png_file, terms=30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    fdist = nltk.FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})
    d = words_df.nlargest(columns="count", n=terms)

    # visualize words and frequencies
    plt.figure(figsize=(12, 15))
    ax = sns.barplot(data=d, x="count", y="word")
    ax.set(ylabel='Word')
    plt.savefig(png_file)

def clean_text(text):
    text = re.sub("\'", "", text)  # remove backslash-apostrophe
    text = re.sub("[^a-zA-Z]", ' ', text) # remove everything except alphabets
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = ' '.join(text.split()) # remove whitespaces
    text = text.lower() # convert text to lowercase
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text).lower()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='Configuration file', required=True)
    args = parser.parse_args()
    config_file = args.config_file

    config = configparser.ConfigParser()
    config.read_file(open(config_file))

    train_data_loc = config.get('01_data_preprocessing_visualization', 'preprocessed_data_file')
    if not os.path.isfile(train_data_loc):
        print('Please provide a valid location of the dataset.')
        exit()

    genre_visualization_loc = config.get('01_data_preprocessing_visualization', 'genre_count_visualization')
    word_freq_visualization_loc = config.get('01_data_preprocessing_visualization', 'word_freq_visualization')
    word_freq_stopword_removal_visualization_loc = config.get('01_data_preprocessing_visualization', 'word_freq_stopword_removal_visualization')

    df = pd.read_csv(train_data_loc,error_bad_lines=False)

    # visualize genre data vs count
    visualizeGenreData(df['Genres'], genre_visualization_loc)

    #clean plot summary
    df['clean_plot_summary'] = df['Plot_summary'].apply(lambda x:clean_text(x))

    #visualize word frequency, top 150
    freq_words(df['clean_plot_summary'], word_freq_visualization_loc, 150)

    df['clean_plot_summary'] = df['Plot_summary'].apply(lambda x: remove_stopwords(x))
    freq_words(df['clean_plot_summary'], word_freq_stopword_removal_visualization_loc, 150)

    log_file = config.get('01_data_preprocessing_visualization', 'save_cleaned_df')
    df.to_csv(log_file, index=False)
    print('Saved preprocessed result to log file!')
