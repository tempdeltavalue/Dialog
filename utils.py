import os 
import glob 
import pandas as pd
import numpy as np
import random

replace_list = {r"i'm": 'i am',
                r"'re": ' are',
                r"let’s": 'let us',
                r"'s":  ' is',
                r"'ve": ' have',
                r"can't": 'can not',
                r"cannot": 'can not',
                r"shan’t": 'shall not',
                r"n't": ' not',
                r"'d": ' would',
                r"'ll": ' will',
                r"'scuse": 'excuse',
                ',': ' ,',
                '.': ' .',
                '!': ' !',
                '?': ' ?',
                '\s+': ' '}

def clean_text(text):
    text = text.lower()
    for s in replace_list:
        text = text.replace(s, replace_list[s])
    text = ' '.join(text.split())
    return text
  
def read_data(base_path):
    dfs = []
    for path in glob.glob(os.path.join(base_path, "*")):
        exercise_df = pd.read_csv(path)

        exercise_df["Ок/Не ок"] = exercise_df["Ок/Не ок"].apply(lambda x: x != "Не ок")

        dfs.append(exercise_df)

    final_df = pd.concat(dfs, ignore_index=True)

    return final_df

def preprocess_binary_label(x, is_remove_neok):
    if is_remove_neok and x["Ок/Не ок"] == False:
      return False

    return x["Упражнение"] == "CalmDownExercise"

def preprocess_multi_label(x, is_replace_neok):
    if x["Ок/Не ок"] == False:
      return random.randint(1,2)

    return 0 

    
def get_X_Y_dfs(path, is_remove_neok):
    exercise_df = read_data("data")

    feature_col_name = "Монолог"
    pred_col_name = ["Упражнение", "Ок/Не ок"]

    X_train = exercise_df[feature_col_name]
    Y_train = exercise_df[pred_col_name]
    # print(Y_train.head())
   

    X_train = X_train.apply(lambda p: clean_text(p))

    Y_train = Y_train.apply(lambda x: preprocess_binary_label(x, is_remove_neok), axis=1)
    print("Num of cat 0", len(Y_train.values[np.where(Y_train.values == 0)]))
    print("Num of cat 1", len(Y_train.values[np.where(Y_train.values == 1)]))

    phrase_len = X_train.apply(lambda p: len(p.split(' ')))
    max_phrase_len = phrase_len.max()

    return X_train, Y_train, max_phrase_len
  
  
def get_cleared_exercise_df(path):
    exercise_df = read_data(path)

    feature_col_name = "Монолог"

    exercise_df[feature_col_name] = exercise_df[feature_col_name].apply(lambda p: 			clean_text(p))

    phrase_len = exercise_df[feature_col_name].apply(lambda p: len(p.split(' ')))
    max_phrase_len = phrase_len.max()

    return exercise_df, max_phrase_len