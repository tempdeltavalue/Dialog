from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util

import pandas as pd
import numpy as np

def convert_string_to_np(x):
    trunc_str_lst = x.replace("[","").replace("]", "").split() # pandas ?:(
    return np.array(trunc_str_lst, dtype=np.float32)

app = Flask(__name__)

df = pd.read_csv("result", sep='\t')
df['sentence_trans_embs'] = df['sentence_trans_embs'].apply(lambda x: convert_string_to_np(x))
data_list = []

sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def show_nearest(df, text):
    df["distance"] = np.zeros(len(df))
    selected_item_embs = sentence_model.encode(text)

    for index, item in enumerate(df.iterrows()):
        item_embs = item[1]["sentence_trans_embs"]
        distance = util.cos_sim(selected_item_embs, item_embs).numpy()[0][0]
        df.at[item[0], 'distance'] = distance

    df.sort_values(by=['distance'], inplace=True, ascending=False)
    return df


class Data:
    def __init__(self, monologue, exercise, is_ok, sentence_trans_embs, init_calm_ex_p):
        self.monologue = monologue
        self.exercise = exercise
        self.is_ok = is_ok
        self.sentence_trans_embs = sentence_trans_embs
        self.init_calm_ex_p = init_calm_ex_p

    @staticmethod
    def convert_df_to_data_list(in_df):
        return in_df.apply(lambda x: Data(x["Монолог"],
                                x["Упражнение"],
                                x["Ок/Не ок"],
                                x["sentence_trans_embs"],
                                x["init_calm_ex_p"]), axis=1)


@app.route('/', methods=["POST"])
def some_function():
    text = request.form.get('search_value')
    result_df = show_nearest(df, text)  # <---Enter your text here
    filtered_data_list = Data.convert_df_to_data_list(result_df)

    return render_template('bootstrap_table.html',
                           title='Elomia data annotation tool',
                           users=filtered_data_list)
@app.route('/')
def index():
    return render_template('bootstrap_table.html',
                           title='Elomia data annotation tool',
                           users=data_list)


if __name__ == '__main__':
    data_list = Data.convert_df_to_data_list(df)

    app.run()
