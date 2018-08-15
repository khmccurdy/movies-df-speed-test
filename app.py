from flask import Flask
from flask import flash, jsonify, redirect, render_template, request, url_for
import pandas as pd
import numpy as np
import png

import load_means as lm

app = Flask(__name__)
app.secret_key = 'some_secret'

print("Loading data...")
means_matrix_1k = lm.png_to_matrix("sparse_means_1k.png")
ids_1k = list(pd.read_csv("movie_ids_1k.csv")["movie_id"])
print("Loaded!")

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route("/predict/<string>")
def get_movies(string):
    input_list = string_to_input_list(string)
    output_df=lm.get_predictions_sparse(input_list,means_matrix_1k,ids_1k).head(10)
    return str(list(output_df["movie_id"]))

def string_to_input_list(string):
    li = [[int(x) for x in string.split(",")[:5][i].split(":")] for i in range(5)]
    return li
    
if __name__ == "__main__":
    print("OK, this works as expected")
    app.run(debug=0)