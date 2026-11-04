import os
from flask import Flask, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder="public", static_url_path="")

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
}

@app.route("/")
def index():
    return render_template("index.html", **METRICS)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
