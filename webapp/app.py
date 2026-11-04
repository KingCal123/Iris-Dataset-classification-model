# app.py - Simple Flask webapp to showcase Iris classification results

"""
This Flask app displays:
  * A short description of the assignment
  * The classification metrics (read from the output printed by the script)
  * The confusion‑matrix and ROC‑curve images generated earlier
The app runs inside the existing virtual environment (``.venv``).
"""

from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

# Absolute path to the assignment folder – used to locate images and the metrics file
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Helper to read the printed benchmark output (the script prints a large block to stdout).
# For simplicity we embed the metrics as a static multiline string here.
METRICS = """
GLOBAL ACCURACY & LOSS
  Accuracy: 90.00% (0.9000)
  Error Rate: 10.00% (0.1000)
  Log Loss (Cross-Entropy): 0.2263

PRECISION (macro/micro/weighted): 0.9024 / 0.9000 / 0.9024
RECALL (macro/micro/weighted):    0.9000 / 0.9000 / 0.9000
SPECIFICITY (macro):               0.9500
F1‑SCORE (macro/micro/weighted):  0.8997 / 0.9000 / 0.8997

MCC: 0.8514
Cohen’s Kappa: 0.8500

ROC‑AUC (OvR & OvO, macro/weighted): 0.9867 for all four variants

PER‑CLASS BREAKDOWN
  setosa      – Precision 1.00, Recall 1.00, Specificity 1.00, F1 1.00
  versicolor  – Precision 0.818, Recall 0.90, Specificity 0.90, F1 0.857
  virginica   – Precision 0.889, Recall 0.80, Specificity 0.95, F1 0.842
"""

@app.route('/')
def index():
    # Paths to the generated images (they live in the parent assignment folder)
    confusion_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    roc_path = os.path.join(BASE_DIR, "roc_curves.png")
    # Flask will serve them via the /static route – copy‑less approach is to use absolute
    # file URLs, but browsers block local file access for security. To keep it simple we
    # expose a dedicated route that sends the files from the parent directory.
    return render_template('index.html', metrics=METRICS,
                           confusion_image='confusion', roc_image='roc')

# Serve the images from the parent folder (outside the Flask static dir)
@app.route('/confusion')
def confusion_image():
    return send_from_directory(BASE_DIR, 'confusion_matrix.png')

@app.route('/roc')
def roc_image():
    return send_from_directory(BASE_DIR, 'roc_curves.png')

if __name__ == '__main__':
    # Run on localhost:5000 – in production you would use a WSGI server (gunicorn, etc.)
    app.run(host='0.0.0.0', port=5000, debug=True)
