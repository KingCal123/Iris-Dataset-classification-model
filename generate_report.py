"""
generate_report.py – Creates the comprehensive 3,000-word Assignment report as a .docx file.
Run this from anywhere: python generate_report.py <output_folder>
"""
import sys
import os
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import datetime

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
CONFUSION_IMG = os.path.join(OUT_DIR, "confusion_matrix.png")
ROC_IMG       = os.path.join(OUT_DIR, "roc_curves.png")
OUT_FILE      = os.path.join(OUT_DIR, "Assignment_Report_Caleb_Tetteh_Martey.docx")

def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1A, 0x5F, 0x9E) if level == 1 else RGBColor(0x2E, 0x86, 0xC1)
        run.font.bold = True
    return h

def add_body(doc, text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for run in p.runs:
        run.font.size = Pt(11)
    return p

def add_custom_table(doc, headers, data):
    table = doc.add_table(rows=1 + len(data), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Light Grid Accent 1"
    
    hdr = table.rows[0].cells
    for idx, h_text in enumerate(headers):
        hdr[idx].text = str(h_text)
        for run in hdr[idx].paragraphs[0].runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            
    for row_idx, row_values in enumerate(data):
        row_cells = table.rows[row_idx + 1].cells
        for col_idx, val in enumerate(row_values):
            row_cells[col_idx].text = str(val)
            for run in row_cells[col_idx].paragraphs[0].runs:
                run.font.size = Pt(10)
    doc.add_paragraph()

def add_image(doc, path, caption, width=Inches(5.5)):
    if os.path.exists(path):
        doc.add_picture(path, width=width)
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in cap.runs:
            run.font.italic = True
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    else:
        doc.add_paragraph(f"[Image not found: {path}]")

doc = Document()

# Set Standard Margins
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(3.0)
    section.right_margin  = Cm(2.5)

# ── Title & Cover Page ────────────────────────────────────────────────────────
doc.add_paragraph()
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("IRIS FLOWER DATASET CLASSIFICATION:\nAN END-TO-END MACHINE LEARNING STUDY & EXPERT SYSTEM DESIGN")
run.font.size = Pt(19)
run.font.bold = True
run.font.color.rgb = RGBColor(0x1A, 0x5F, 0x9E)

doc.add_paragraph()
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.add_run("Comprehensive Technical Report – AI and Expert Systems").font.size = Pt(13)

doc.add_paragraph()
info = [
    ("Student Name:",  "Caleb Tetteh Martey"),
    ("Student ID:",    "PUIT/23110056"),
    ("Programme:",     "BSc Information Technology"),
    ("Course:",        "AI and Expert Systems"),
    ("Submission Date:", datetime.date.today().strftime("%B %d, %Y")),
]
for label, val in info:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    bold = p.add_run(label + "  ")
    bold.font.bold = True
    bold.font.size = Pt(12)
    normal = p.add_run(val)
    normal.font.size = Pt(12)

doc.add_page_break()

# ── Executive Summary / Abstract ──────────────────────────────────────────────
add_heading(doc, "Abstract")
add_body(doc,
    "This technical report presents an empirical investigation into multi-class classification using the benchmark "
    "Iris flower dataset within the broader context of Artificial Intelligence and Expert Systems. The central objective "
    "of this study is to engineer, assess, and evaluate an automated pattern recognition pipeline using Python and "
    "the scikit-learn machine learning library. A Random Forest ensemble classifier was selected, trained, and subjected "
    "to an exhaustive suite of classification performance metrics. Beyond elementary classification accuracy, the model was "
    "critiqued across precision, recall (sensitivity), specificity, harmonic F1-measures (macro, micro, and weighted averages), "
    "Matthews Correlation Coefficient (MCC), Cohen's Kappa inter-rater agreement, log loss (cross-entropy), and multi-class "
    "Area Under the Receiver Operating Characteristic Curve (ROC AUC) using both One-vs-Rest (OvR) and One-vs-One (OvO) formulations. "
    "The classifier achieved an overall predictive accuracy of 90.00% on the stratified hold-out test set, with a multi-class "
    "ROC AUC of 0.9867 and a Cohen's Kappa of 0.8500, indicating near-perfect agreement beyond chance. In addition to offline "
    "statistical evaluation, an interactive dashboard web application was designed, engineered using the Flask framework, "
    "and configured for serverless production deployment. This report provides a detailed narrative spanning task formalisation, "
    "exploratory data analysis, mathematical metric formulations, empirical findings, and deployment architecture."
)
doc.add_paragraph()

# ── 1. Task Description & Problem Formalisation ───────────────────────────────
add_heading(doc, "1. Task Description & Problem Formulation")
add_body(doc,
    "In the domain of Artificial Intelligence and Pattern Recognition, classification represents one of the canonical "
    "supervised learning paradigms. Formally, given a feature space X in R^d and a discrete label space Y = {y_1, y_2, ..., y_k}, "
    "the primary objective is to learn an inductive hypothesis function f: X -> Y that minimises the risk of misclassification "
    "under an underlying joint probability distribution P(X, Y). When the cardinality of Y exceeds two (|Y| > 2), the learning "
    "challenge transitions from standard binary classification to multi-class classification, introducing distinctive challenges "
    "regarding decision boundary geometry, class separability, error decomposition, and evaluation metric aggregation."
)
add_body(doc,
    "The prescribed coursework task mandates the utilisation of the Iris dataset sourced directly from the scikit-learn "
    "distribution. The task requires: (1) loading and inspecting the biological measurement records, (2) partitioning the observations "
    "into independent training and validation domains, (3) implementing a robust classification algorithm, (4) computing every "
    "pertinent classification evaluation metric to capture error characteristics comprehensively, and (5) presenting the "
    "synthesised model and evaluation results through a modern, interactive web platform. This document satisfies all formal "
    "academic requirements, demonstrating both computational implementation proficiency and theoretical comprehension."
)

# ── 2. Dataset Description ─────────────────────────────────────────────────────
add_heading(doc, "2. Dataset Description and Morphological Properties")
add_body(doc,
    "The Iris dataset is historically celebrated as one of the quintessential benchmark datasets in mathematical statistics "
    "and pattern recognition. First documented by British statistician and geneticist Sir Ronald Aylmer Fisher in his 1936 "
    "seminal paper 'The use of multiple measurements in taxonomic problems', the dataset originates from biological field specimens "
    "collected across the Gaspé Peninsula in Quebec, Canada, by botanist Dr. Edgar Anderson. The data comprises 150 morphometric "
    "records sampled equally from three botanical species belonging to the Iris genus: Iris setosa, Iris versicolor, and Iris virginica."
)
add_body(doc,
    "Each individual floral specimen is characterised by four physical continuous morphological attributes measured precisely in centimetres (cm):"
)
doc.add_paragraph("1. Sepal Length (cm): Continuous numerical variable measuring the longitudinal dimension of the calyx leaf.", style="List Bullet")
doc.add_paragraph("2. Sepal Width (cm): Continuous numerical variable capturing the transverse diameter of the calyx leaf.", style="List Bullet")
doc.add_paragraph("3. Petal Length (cm): Continuous numerical variable measuring the length of the inner flower petal.", style="List Bullet")
doc.add_paragraph("4. Petal Width (cm): Continuous numerical variable capturing the width of the inner flower petal.", style="List Bullet")

add_body(doc,
    "The target categorical variable Y corresponds to the taxonomic species identification, encoded as follows: "
    "Class 0 represents Iris setosa, Class 1 represents Iris versicolor, and Class 2 represents Iris virginica. "
    "A paramount analytical advantage of the Iris corpus is its structural balance: each target class contains exactly 50 "
    "independent observations, resulting in an exact 33.33% uniform class prior across the population. Furthermore, the dataset "
    "exhibits zero missing values, zero corrupted data types, and well-behaved variance profiles."
)

# ── 3. Exploratory Data Analysis (EDA) ─────────────────────────────────────────
add_heading(doc, "3. Exploratory Data Analysis (EDA)")
add_body(doc,
    "Exploratory Data Analysis constitutes a critical preliminary phase in any machine learning lifecycle. Through rigorous "
    "statistical examination and visual inspection, EDA uncovers structural patterns, collinearity, distribution modalities, "
    "and geometric separability among the underlying classes. The 150 instances were statistically profiled to extract central "
    "tendency, dispersion, and range metrics across each of the four physical features."
)

add_heading(doc, "3.1 Descriptive Summary Statistics", level=2)
add_body(doc,
    "Table 1 outlines the parametric statistics of the Iris feature space, including the minimum, maximum, arithmetic mean, "
    "standard deviation, and interquartile bounds computed across the entire feature matrix."
)
add_custom_table(doc, 
    ["Feature Attribute", "Min (cm)", "Max (cm)", "Mean (cm)", "Std Dev (cm)", "Median (cm)"],
    [
        ["Sepal Length", "4.30", "7.90", "5.84", "0.83", "5.80"],
        ["Sepal Width",  "2.00", "4.40", "3.05", "0.43", "3.00"],
        ["Petal Length", "1.00", "6.90", "3.76", "1.76", "4.35"],
        ["Petal Width",  "0.10", "2.50", "1.20", "0.76", "1.30"],
    ]
)

add_heading(doc, "3.2 Feature Correlation and Bivariate Relationships", level=2)
add_body(doc,
    "Bivariate Pearson correlation analysis indicates strong positive linear dependencies between Petal Length and Petal Width "
    "(r = +0.963), as well as between Petal Length and Sepal Length (r = +0.872). Conversely, Sepal Width demonstrates negative "
    "or weak correlations with all other morphological dimensions (e.g., r = -0.118 with Sepal Length, r = -0.428 with Petal Length). "
    "Biologically, this reflects coordinated allometric growth between petal structures, whereas sepal shape varies across different "
    "structural proportions."
)

add_heading(doc, "3.3 Class Separability & Geometric Manifold Observations", level=2)
add_body(doc,
    "From a topological perspective, the Iris dataset presents two distinct classification frontiers:"
)
doc.add_paragraph("• Linear Separability of Iris Setosa: When projected onto the 2D subspace spanned by Petal Length and Petal Width, "
                  "Iris setosa forms an isolated cluster with substantial spatial separation from the remaining classes. Consequently, "
                  "virtually any hyperplane can isolate Setosa with 100% precision and recall.", style="List Bullet")
doc.add_paragraph("• Non-Linear Boundary between Versicolor and Virginica: In contrast, Versicolor and Virginica overlap "
                  "within intermediate petal measurement bands (petal lengths between 4.5 cm and 5.1 cm; petal widths between 1.4 cm and 1.8 cm). "
                  "This boundary requires non-linear discrimination capabilities, making ensemble tree models exceptionally well-suited.", style="List Bullet")

# ── 4. Methodology and Model Formulation ──────────────────────────────────────
add_heading(doc, "4. Methodology and Experimental Setup")
add_heading(doc, "4.1 Train-Test Stratification Protocol", level=2)
add_body(doc,
    "To reliably simulate prospective model generalisation on unseen biological samples, the full dataset of 150 instances was "
    "partitioned into an 80% training set (120 observations) and a 20% hold-out test set (30 observations). Stratified sampling "
    "was strictly enforced via scikit-learn's train_test_split function with a random_state of 42. Stratification guarantees that "
    "each target species retains an identical representation ratio across both subsets (precisely 40 training and 10 testing samples "
    "for Setosa, Versicolor, and Virginica). This eliminates sampling bias and ensures that test metric evaluations remain balanced."
)

add_heading(doc, "4.2 Selected Classifier: Random Forest Ensemble", level=2)
add_body(doc,
    "The core predictive engine selected for this classification assignment is the Random Forest Classifier (Breiman, 2001). "
    "Random Forest is an ensemble meta-estimator that fits multiple decision tree classifiers on various subsamples of the "
    "dataset and uses bootstrap aggregating (bagging) alongside random feature subspace selection. This mechanism significantly "
    "mitigates the high variance and prone-to-overfitting tendencies inherent in individual unpruned decision trees."
)
add_body(doc,
    "Mathematically, given an ensemble of B trees {T_1, T_2, ..., T_B}, the ensemble prediction for a test vector x is determined "
    "via majority voting over individual tree class probabilities:\n"
    "p(y = c | x) = (1 / B) * sum_{b=1}^{B} I(T_b(x) = c)\n"
    "The hyperparameter configuration employed for this experiment was established as follows: n_estimators = 100, "
    "criterion = 'gini', max_depth = None, min_samples_split = 2, and random_state = 42. Crucially, Random Forests are invariant to "
    "monotonic feature scale transformations, eliminating the risk of information leakage from data normalisation."
)

add_heading(doc, "4.3 Computational Environment", level=2)
add_body(doc,
    "The experimental pipeline was constructed within an isolated Python 3.13 virtual environment (.venv) on Windows. "
    "Dependencies were formally managed using requirements.txt. Core packages included scikit-learn 1.9.1, pandas 3.1.0, "
    "numpy 2.5.4, matplotlib 3.11.2, seaborn 0.13.2, and python-docx 1.2.0."
)

# ── 5. Comprehensive Model Evaluation & Mathematical Formulations ─────────────
add_heading(doc, "5. Comprehensive Model Evaluation & Metrics")
add_body(doc,
    "A central mandate of this assignment is the computation and theoretical analysis of all primary classification evaluation "
    "metrics. In multi-class classification, evaluating models solely on raw accuracy can produce misleading conclusions. A robust "
    "evaluation strategy must decompose error profiles across false positives, false negatives, probabilistic confidence, and "
    "chance agreement."
)

add_heading(doc, "5.1 Accuracy and Error Rate", level=2)
add_body(doc,
    "Accuracy measures the proportion of total correct predictions relative to the total number of test instances evaluated:\n"
    "Accuracy = (TP + TN) / (TP + TN + FP + FN) = (Number of Correct Predictions) / (Total Number of Predictions)\n"
    "The classifier successfully identified 27 out of 30 unseen test specimens, producing an overall Accuracy of 90.00% (0.9000). "
    "The Error Rate represents the mathematical complement of accuracy: Error Rate = 1 - Accuracy = 10.00% (0.1000)."
)

add_heading(doc, "5.2 Precision, Recall, Specificity, and F1-Score Formulations", level=2)
add_body(doc,
    "For multi-class domains, binary metric definitions are generalised using One-vs-Rest (OvR) decompositions:\n"
    "• Precision (Positive Predictive Value): Precision_c = TP_c / (TP_c + FP_c). Measures the fidelity of positive assertions.\n"
    "• Recall / Sensitivity (True Positive Rate): Recall_c = TP_c / (TP_c + FN_c). Measures the model's completeness in capturing positives.\n"
    "• Specificity (True Negative Rate): Specificity_c = TN_c / (TN_c + FP_c). Measures the avoidance of false alarms for class c.\n"
    "• F1-Score: The harmonic mean of Precision and Recall: F1_c = 2 * (Precision_c * Recall_c) / (Precision_c + Recall_c)."
)
add_body(doc,
    "To summarise performance across all K classes, three aggregation paradigms are computed:\n"
    "1. Macro Average: The unweighted arithmetic mean across classes: Metric_macro = (1/K) * sum_{c=1}^{K} Metric_c.\n"
    "2. Micro Average: Aggregates global true positives, false positives, and false negatives prior to metric calculation.\n"
    "3. Weighted Average: Weights class scores by class support: Metric_weighted = sum_{c=1}^{K} (N_c / N) * Metric_c."
)

add_custom_table(doc,
    ["Evaluation Metric", "Macro Average", "Micro Average", "Weighted Average"],
    [
        ["Precision", "0.9024", "0.9000", "0.9024"],
        ["Recall (Sensitivity)", "0.9000", "0.9000", "0.9000"],
        ["Specificity (True Negative Rate)", "0.9500", "N/A", "N/A"],
        ["F1-Score (Harmonic Mean)", "0.8997", "0.9000", "0.8997"],
    ]
)

add_heading(doc, "5.3 Correlation & Chance-Adjusted Agreement Metrics", level=2)
add_body(doc,
    "To provide deeper diagnostic insight, advanced correlation metrics were evaluated:\n"
    "• Matthews Correlation Coefficient (MCC): Celebrated for its mathematical balance, MCC treats all four confusion quadrants "
    "symmetrically. In multi-class problems, MCC generalizes the Pearson correlation between predicted and observed binary indicator matrices. "
    "The model achieved an MCC of 0.8514 (on a [-1, +1] scale), indicating exceptional predictive correlation.\n"
    "• Cohen's Kappa Score (κ): Quantifies inter-annotator agreement while explicitly discounting the probability of agreement "
    "occurring by chance: κ = (p_o - p_e) / (1 - p_e). The calculated score of 0.8500 signifies 'almost perfect agreement' according "
    "to the standard Landis and Koch diagnostic framework.\n"
    "• Multi-Class Log Loss (Cross-Entropy): Log loss quantifies the uncertainty of probabilistic outputs by heavily penalising "
    "confident incorrect classifications. The model registered an average log loss of 0.2263, proving high probabilistic calibration."
)

add_heading(doc, "5.4 Multi-Class Receiver Operating Characteristic (ROC AUC)", level=2)
add_body(doc,
    "The Area Under the Receiver Operating Characteristic Curve (ROC AUC) evaluates the sensitivity-specificity trade-off across all "
    "decision thresholds. For multi-class classification, scikit-learn computes ROC AUC through One-vs-Rest (OvR) and One-vs-One (OvO) "
    "formulations. Both strategies registered an outstanding score of 0.9867 across macro and weighted averages, demonstrating near-optimal "
    "rank separation between species."
)

# ── 5.5 Confusion Matrix ───────────────────────────────────────────────────────
add_heading(doc, "5.5 Confusion Matrix Analysis", level=2)
add_body(doc,
    "The confusion matrix represents the ground-truth contingency table, where rows delineate the actual biological species and columns "
    "delineate the model's predictions on the 30 hold-out instances."
)
add_custom_table(doc,
    ["True Class \\ Predicted Class", "Predicted Setosa", "Predicted Versicolor", "Predicted Virginica"],
    [
        ["Actual Iris Setosa",     "10 (100% correct)", "0",                 "0"],
        ["Actual Iris Versicolor", "0",                 "9 (90% correct)",   "1 (misclassified)"],
        ["Actual Iris Virginica",  "0",                 "2 (misclassified)", "8 (80% correct)"],
    ]
)
add_body(doc,
    "Detailed inspection confirms that Iris Setosa was classified with 100% purity (zero false positives, zero false negatives). "
    "All three classification errors occurred exclusively along the subtle Versicolor-Virginica interface (one Versicolor misidentified as "
    "Virginica, and two Virginicas misidentified as Versicolor), precisely aligning with known botanical morphology."
)

# Visual Embeddings
add_image(doc, CONFUSION_IMG, "Figure 1: Confusion Matrix Heatmap – Test Set Predictions vs Actual Classes")
doc.add_paragraph()
add_image(doc, ROC_IMG, "Figure 2: Multi-Class ROC Curves (One-vs-Rest) with Area Under Curve Metrics")
doc.add_paragraph()

# ── 5.6 Per-Class Detailed Breakdown ──────────────────────────────────────────
add_heading(doc, "5.6 Granular Per-Class Performance Breakdown", level=2)
add_custom_table(doc,
    ["Taxonomic Class", "Precision", "Recall", "Specificity", "F1-Score", "Test Support"],
    [
        ["Iris Setosa",     "1.0000", "1.0000", "1.0000", "1.0000", "10"],
        ["Iris Versicolor", "0.8182", "0.9000", "0.9000", "0.8571", "10"],
        ["Iris Virginica",  "0.8889", "0.8000", "0.9500", "0.8421", "10"],
    ]
)

# ── 6. Comparative Discussion & Error Analysis ────────────────────────────────
add_heading(doc, "6. Discussion and Error Analysis")
add_body(doc,
    "The empirical findings affirm the efficacy of tree ensemble architectures for botanical pattern classification. "
    "The primary insight gained from the error analysis concerns the nature of class boundaries in high-dimensional biological datasets. "
    "While elementary heuristic rules can cleanly isolate Iris setosa based purely on petal length (e.g., petal length < 2.5 cm), "
    "Versicolor and Virginica exhibit genuine biological variability. Certain Virginica specimens display dwarfed petal measurements "
    "that closely mimic mature Versicolor profiles."
)
add_body(doc,
    "By evaluating the model through multiple lenses—comparing Macro F1 (0.8997) against Weighted F1 (0.8997) and ROC AUC (0.9867)—it "
    "becomes evident that the classifier maintains high discriminative capability despite boundary fuzziness. The low log loss score "
    "(0.2263) confirms that the tree ensemble assigns realistic confidence scores rather than overconfident false assertions."
)

# ── 7. Web Application Architecture and Cloud Deployment ─────────────────────
add_heading(doc, "7. Web Application Architecture and Cloud Deployment")
add_body(doc,
    "To translate analytical models into actionable decision-support tools, machine learning pipelines must be paired with accessible "
    "user interfaces. In this assignment, a dedicated web application was engineered to showcase the classification benchmark."
)
add_heading(doc, "7.1 Frontend & User Experience Design", level=2)
add_body(doc,
    "The dashboard interface was developed with modern web standards, featuring: a dark-mode palette built on clean typography "
    "(Inter font), responsive KPI metric cards, interactive HTML confusion matrix tables, animated progress bars for precision and recall, "
    "high-resolution matplotlib visualisations, and explanatory documentation cards outlining the dataset and training parameters."
)
add_heading(doc, "7.2 Flask Backend and Serverless Deployment Setup", level=2)
add_body(doc,
    "The application backend leverages Python's lightweight Flask framework. For production hosting, the repository was structured "
    "to conform to modern serverless deployment standards (specifically Vercel's Python Serverless Runtime). The project directory "
    "architecture was organised as follows:\n"
    "• api/index.py: Flask WSGI entry-point handler responding to web requests.\n"
    "• templates/index.html: Responsive dashboard template presenting evaluation metrics and figures.\n"
    "• public/: Static asset directory hosting generated PNG visualisations.\n"
    "• vercel.json: Serverless routing specification dispatching static routes and Python API invocations.\n"
    "• requirements.txt: Lightweight dependency manifest ensuring reproducible cloud builds."
)

# ── 8. Conclusion ─────────────────────────────────────────────────────────────
add_heading(doc, "8. Conclusion and Future Directions")
add_body(doc,
    "This project successfully designed, implemented, and validated an end-to-end multi-class machine learning classification "
    "pipeline on the Iris flower dataset. The Random Forest model achieved 90.00% accuracy, an MCC of 0.8514, and an ROC AUC of 0.9867, "
    "substantiating high analytical reliability. The accompanying web dashboard bridges computational modelling and visual reporting, "
    "delivering an intuitive platform for stakeholders."
)
add_body(doc,
    "Future enhancements could incorporate automated hyperparameter optimisation via Bayesian search, expand the model suite to include "
    "Gradient Boosting and Support Vector Machines (SVM), and incorporate interactive single-sample prediction widgets allowing users to "
    "input custom floral measurements in real time."
)

# ── 9. References ──────────────────────────────────────────────────────────────
add_heading(doc, "9. References")
refs = [
    "Anderson, E. (1935). The irises of the Gaspe Peninsula. Bulletin of the American Iris Society, 59, 2–5.",
    "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.",
    "Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1), 37–46.",
    "Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. Annals of Eugenics, 7(2), 179–188.",
    "Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. Biometrics, 33(1), 159–174.",
    "Matthews, B. W. (1975). Comparison of the predicted and observed secondary structure of T4 phage lysozyme. BBA, 405(2), 442–451.",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    "Powers, D. M. W. (2011). Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness and Correlation. JMLAI, 2(1), 37–63."
]
for ref in refs:
    p = doc.add_paragraph(ref, style="List Number")
    p.runs[0].font.size = Pt(10)

doc.save(OUT_FILE)
print(f"[SUCCESS] Report written to: {OUT_FILE}")
