"""
api/index.py  –  Vercel-compatible Flask entry point.

Vercel's @vercel/python builder looks for a WSGI app called `app`
(or `handler`) in api/index.py.  Images are served as Vercel static
files from the /public directory.
"""

import os
import sys

# Make sure the templates folder (sibling of api/) is importable
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

from flask import Flask, render_template

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# ── Metrics data (kept as a dict so the template can render it nicely) ─────
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
