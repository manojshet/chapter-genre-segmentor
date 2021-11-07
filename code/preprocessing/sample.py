import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("../dataset/preprocessed/01_cleaned.csv",error_bad_lines=False)
    print(len(df.Genres.value_counts().index.to_list()))