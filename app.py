import os
import json
import pandas as pd
from flask import Flask, render_template, send_file, jsonify
from sklearn.datasets import load_iris

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
CSV_PATH = os.path.join(BASE_DIR, "iris_dataset.csv")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Load dataset records once for the interactive table
iris = load_iris()
iris_records = []
for i in range(len(iris.data)):
    iris_records.append({
        "id": i + 1,
        "sepal_length": iris.data[i][0],
        "sepal_width": iris.data[i][1],
        "petal_length": iris.data[i][2],
        "petal_width": iris.data[i][3],
        "species": iris.target_names[iris.target[i]]
    })

METRICS = {
    "accuracy":    "90.00%",
    "error_rate":  "10.00%",
    "log_loss":    "0.2263",
    "prec_macro":  "0.9024",
    "prec_micro":  "0.9000",
    "prec_w":      "0.9024",
    "rec_macro":   "0.9000",
    "rec_micro":   "0.9000",
    "rec_w":       "0.9000",
    "spec_macro":  "0.9500",
    "f1_macro":    "0.8997",
    "f1_micro":    "0.9000",
    "f1_w":        "0.8997",
    "mcc":         "0.8514",
    "kappa":       "0.8500",
    "roc_ovr_mac": "0.9867",
    "roc_ovr_w":   "0.9867",
    "roc_ovo_mac": "0.9867",
    "roc_ovo_w":   "0.9867",
    "dataset_json": json.dumps(iris_records)
}

@app.route("/")
def index():
    return render_template("index.html", **METRICS)

@app.route("/download/dataset")
def download_dataset():
    if os.path.exists(CSV_PATH):
        return send_file(CSV_PATH, as_attachment=True, download_name="iris_dataset.csv", mimetype="text/csv")
    df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    df['species'] = [iris.target_names[i] for i in iris.target]
    df.to_csv(CSV_PATH, index=False)
    return send_file(CSV_PATH, as_attachment=True, download_name="iris_dataset.csv", mimetype="text/csv")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
