"""
Generates outputs/project_presentation.pdf for the Fraud Detection project.
Run: python make_pdf.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable,
    Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import os

OUTPUT = "outputs/project_presentation.pdf"
PLOTS  = "outputs/plots"
PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm

doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=2.0*cm, bottomMargin=2.0*cm,
    title="Credit Card Fraud Detection — Project Presentation",
    author="Kate Pogrebnyakova",
)

NAVY  = colors.HexColor('#1a3a5c')
TEAL  = colors.HexColor('#00897b')
CORAL = colors.HexColor('#e64a19')
GREY  = colors.HexColor('#555555')
LGREY = colors.HexColor('#f5f5f5')

s_title    = ParagraphStyle('Title',    fontName='Helvetica-Bold', fontSize=18, textColor=NAVY, spaceAfter=4,  alignment=TA_CENTER)
s_subtitle = ParagraphStyle('Subtitle', fontName='Helvetica',      fontSize=11, textColor=GREY, spaceAfter=14, alignment=TA_CENTER)
s_section  = ParagraphStyle('Section', fontName='Helvetica-Bold',  fontSize=12, textColor=NAVY, spaceBefore=18, spaceAfter=6)
s_sub      = ParagraphStyle('Sub',     fontName='Helvetica-Bold',  fontSize=10, textColor=TEAL, spaceBefore=10, spaceAfter=4)
s_body     = ParagraphStyle('Body',    fontName='Helvetica',       fontSize=9.5, leading=15, textColor=colors.HexColor('#222222'), spaceAfter=5, alignment=TA_JUSTIFY)
s_bullet   = ParagraphStyle('Bullet',  fontName='Helvetica',       fontSize=9.5, leading=15, textColor=colors.HexColor('#222222'), leftIndent=16, bulletIndent=4, spaceAfter=3)
s_note     = ParagraphStyle('Note',    fontName='Helvetica-Oblique', fontSize=8.5, leading=13, textColor=GREY, spaceAfter=4, leftIndent=10)

def hr():
    return HRFlowable(width='100%', thickness=0.6, color=colors.HexColor('#cccccc'), spaceAfter=8, spaceBefore=4)
def section(t): return Paragraph(t.upper(), s_section)
def sub(t):     return Paragraph(t, s_sub)
def body(t):    return Paragraph(t, s_body)
def bullet(t):  return Paragraph(f'&bull;&nbsp;&nbsp;{t}', s_bullet)
def note(t):    return Paragraph(t, s_note)
def space(h=8): return Spacer(1, h)

def chart(fname, caption, hf=0.40):
    path = os.path.join(PLOTS, fname)
    if not os.path.exists(path):
        return [body(f'[Chart not found: {fname}]')]
    usable_w = PAGE_W - 2 * MARGIN
    return [Image(path, width=usable_w, height=usable_w * hf),
            note(f'Figure: {caption}')]

def tbl_style(t, col_widths):
    t2 = Table(t, colWidths=col_widths)
    t2.setStyle(TableStyle([
        ('BACKGROUND',     (0,0),(-1,0), NAVY),
        ('TEXTCOLOR',      (0,0),(-1,0), colors.white),
        ('FONTNAME',       (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',       (0,0),(-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1),(-1,-1), [colors.white, LGREY]),
        ('GRID',           (0,0),(-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('TOPPADDING',     (0,0),(-1,-1), 5),
        ('BOTTOMPADDING',  (0,0),(-1,-1), 5),
        ('LEFTPADDING',    (0,0),(-1,-1), 8),
    ]))
    return t2


story = []

# Title
story += [space(10),
          Paragraph("Credit Card Fraud Detection", s_title),
          Paragraph("Imbalanced Classification — Machine Learning Project", s_subtitle),
          Paragraph("Kate Pogrebnyakova &nbsp;|&nbsp; Data Scientist", s_subtitle),
          hr()]

# Business question
story.append(section("Business Question"))
story.append(body(
    "A bank processes 284,807 credit card transactions over 2 days. "
    "492 are fraudulent — just <b>0.17%</b> of all transactions. "
    "How do we build a model that catches as much fraud as possible "
    "without blocking too many legitimate customers?"
))
story.append(body(
    "This is a <b>precision-recall tradeoff problem</b>. The cost of missing fraud "
    "(bank absorbs the loss) is very different from a false alarm (customer friction). "
    "The optimal model minimises total business cost — not statistical accuracy."
))

# Why accuracy is wrong
story.append(section("Why Accuracy Is the Wrong Metric"))
story.append(body(
    "A model that predicts <i>no fraud</i> for every transaction achieves <b>99.83% accuracy</b> "
    "while catching zero fraud cases. This is the core challenge of imbalanced classification."
))
story.append(bullet("<b>Precision-Recall AUC</b> (Average Precision) is the correct metric — "
                    "it focuses entirely on minority class detection performance"))
story.append(bullet("<b>Precision</b> = of all fraud alerts fired, what % were real fraud"))
story.append(bullet("<b>Recall</b> = of all real fraud cases, what % did we catch"))
story.append(bullet("<b>ROC-AUC is misleading here</b> — it looks great even for bad models "
                    "because the True Negative Rate is inflated when negatives dominate 99.8% of data"))

# The data
story.append(section("The Data"))
story.append(body(
    "Dataset: <b>Credit Card Fraud Detection</b> (Kaggle / ULB Machine Learning Group) — "
    "284,807 European cardholder transactions from September 2013."
))

data_t = [
    ['Metric', 'Value'],
    ['Total transactions',  '284,807'],
    ['Fraud cases',         '492 (0.173%)'],
    ['Legitimate cases',    '284,315 (99.827%)'],
    ['Missing values',      'None'],
    ['Features',            'V1–V28 (PCA-transformed, anonymised) + Time + Amount'],
    ['Target',              'Class: 0 = legitimate, 1 = fraud'],
    ['Amount range',        '€0.00 – €25,691'],
]
story += [space(6), tbl_style(data_t, [6*cm, 10*cm]), space(8)]
story += chart('c1_class_imbalance', 'Extreme class imbalance: 492 fraud in 284,807 transactions', 0.38)

# Data prep
story.append(PageBreak())
story.append(section("Data Preparation"))
story.append(sub("Feature scaling"))
story.append(body(
    "V1–V28 are already PCA-transformed and centred — do <b>not</b> scale them again. "
    "Only <b>Time</b> and <b>Amount</b> are scaled with StandardScaler."
))
story.append(sub("Stratified train/test split"))
story.append(body(
    "With 0.17% fraud rate, a random split could concentrate all fraud in one set. "
    "Stratified split preserves the fraud rate in both train (394 cases) and test (98 cases)."
))
story.append(sub("SMOTE — Synthetic Minority Over-sampling Technique"))
story.append(body(
    "SMOTE creates synthetic fraud samples in the training set to balance the classes. "
    "<b>Critical: SMOTE is applied only to the training set.</b> "
    "The test set remains the original imbalanced distribution — the real-world scenario. "
    "Applying SMOTE before splitting would leak synthetic samples into evaluation."
))
story += [space(4)]
story.append(body(
    "Training set after SMOTE: 227,452 fraud | 227,452 legitimate (50/50). "
    "Test set unchanged: 98 fraud | 56,863 legitimate (0.17%)."
))

# C2 + C4
story += [space(6)]
story += chart('c2_amount_distribution', 'Fraud transactions cluster at lower amounts', 0.38)
story += [space(6)]
story += chart('c4_feature_distributions', 'Top 6 most discriminative PCA features — fraud vs legitimate', 0.44)

# Models
story.append(PageBreak())
story.append(section("Models — Three Approaches to Class Imbalance"))
story.append(body(
    "Three models were trained, each using a different strategy to handle the class imbalance. "
    "This demonstrates breadth — all three approaches are valid with different tradeoffs."
))

model_t = [
    ['Model', 'Imbalance Strategy', 'Notes'],
    ['Logistic Regression', 'SMOTE on training set', 'Baseline — linear classifier'],
    ['Random Forest',       'class_weight=\'balanced\'', 'Algorithmic upweighting of minority class'],
    ['XGBoost',             'scale_pos_weight = 577', 'Native XGBoost parameter (neg/pos ratio)'],
]
story += [space(6), tbl_style(model_t, [4.5*cm, 5.5*cm, 6*cm]), space(8)]

# Results
story.append(section("Results"))
res_t = [
    ['Model', 'PR-AUC', 'Recall', 'Precision', 'FN (missed)', 'FP (alerts)'],
    ['Logistic Regression', '0.7249', '91.8%',  '5.8%',  '8',  '1,458'],
    ['Random Forest',       '0.8653', '75.5%',  '96.1%', '24', '3'],
    ['XGBoost',             '0.8810', '83.7%',  '87.2%', '16', '12'],
]
res_tbl = Table(res_t, colWidths=[4.5*cm, 2*cm, 2*cm, 2.5*cm, 2.5*cm, 2.5*cm])
res_tbl.setStyle(TableStyle([
    ('BACKGROUND',     (0,0),(-1,0), NAVY),
    ('TEXTCOLOR',      (0,0),(-1,0), colors.white),
    ('FONTNAME',       (0,0),(-1,0), 'Helvetica-Bold'),
    ('FONTNAME',       (0,3),(0,3),  'Helvetica-Bold'),  # XGBoost row
    ('FONTSIZE',       (0,0),(-1,-1), 8.5),
    ('ROWBACKGROUNDS', (0,1),(-1,-1), [colors.white, LGREY]),
    ('GRID',           (0,0),(-1,-1), 0.5, colors.HexColor('#cccccc')),
    ('TOPPADDING',     (0,0),(-1,-1), 5),
    ('BOTTOMPADDING',  (0,0),(-1,-1), 5),
    ('LEFTPADDING',    (0,0),(-1,-1), 8),
    ('BACKGROUND',     (0,3),(-1,3), colors.HexColor('#e8f0f8')),  # highlight XGB
]))
story += [space(6), res_tbl, space(4)]
story.append(note(
    "FN = false negatives (fraud missed — bank absorbs loss). "
    "FP = false positives (legitimate transactions blocked — customer friction). "
    "XGBoost highlighted: best PR-AUC with balanced FN/FP tradeoff. "
    "Test set: 56,961 transactions, 98 fraud cases."
))

story += [space(8)]
story += chart('c6_precision_recall_curves', 'Precision-Recall curves — XGBoost achieves highest Average Precision (0.881)', 0.42)
story += [space(6)]
story += chart('c7_confusion_matrices', 'Confusion matrices — real error counts for all three models', 0.30)

# Threshold tuning
story.append(PageBreak())
story.append(section("Threshold Tuning — Business Cost Optimisation"))
story.append(body(
    "Default classification threshold is 0.5. The better question: "
    "<b>what threshold minimises total expected business cost?</b>"
))
story.append(body(
    "Cost assumptions (illustrative — set by the business):"
))
story.append(bullet("<b>False Negative</b> (missed fraud): €200 — bank absorbs average fraud loss"))
story.append(bullet("<b>False Positive</b> (false alarm): €10 — customer friction, service cost, churn risk"))
story.append(body(
    "XGBoost probabilities were swept from 0.01 to 0.99 and total expected cost "
    "computed at each threshold: <b>Total Cost = (FN × €200) + (FP × €10)</b>. "
    "The threshold at minimum cost is the operationally correct operating point."
))
story += [space(6)]
story += chart('c9_threshold_optimisation',
               'Threshold tuning — cost-minimising point balances recall and false alarm rate', 0.42)
story.append(body(
    "This framing mirrors credit risk management — where missing a bad loan costs far more "
    "than the review cost of a false flag. Kate's accounting background makes this the natural "
    "way to think about the problem."
))

# Feature importance
story += [space(6)]
story += chart('c10_feature_importance', 'XGBoost feature importance — V14, V17, V12 are most discriminative', 0.44)

# Tech stack
story.append(section("Technology Stack"))
tech_t = [
    ['Area', 'Tools'],
    ['Data handling',   'Python, pandas, numpy'],
    ['Preprocessing',   'scikit-learn (StandardScaler, train_test_split)'],
    ['Imbalance',       'imbalanced-learn (SMOTE)'],
    ['Models',          'scikit-learn (LogisticRegression, RandomForestClassifier)'],
    ['Gradient boost',  'XGBoost (XGBClassifier, scale_pos_weight)'],
    ['Evaluation',      'precision_recall_curve, average_precision_score, confusion_matrix'],
    ['Visualisation',   'matplotlib, seaborn'],
]
story += [space(6), tbl_style(tech_t, [5*cm, 11*cm])]

# Key findings
story.append(section("Key Findings"))
story.append(bullet(
    "<b>Accuracy is not the right metric</b> — 99.83% accuracy from predicting no fraud. "
    "PR-AUC is the correct evaluation metric for this problem."
))
story.append(bullet(
    "<b>XGBoost achieves the highest PR-AUC (0.881)</b> with a strong balance: "
    "catches 83.7% of fraud with 87.2% precision and only 12 false alarms on the test set."
))
story.append(bullet(
    "<b>Logistic Regression catches more fraud (91.8% recall) but fires 1,458 false alerts</b> — "
    "operationally unworkable. The tradeoff is explicit and measurable."
))
story.append(bullet(
    "<b>Threshold tuning reduces total business cost</b> below the statistical default. "
    "The optimal threshold is a business decision, not a statistics decision."
))
story.append(bullet(
    "<b>Three imbalance strategies compared</b> — SMOTE, class_weight, scale_pos_weight — "
    "demonstrating breadth of approach beyond just running a model."
))

# Footer
story += [space(16), hr(),
          Paragraph("Kate Pogrebnyakova &nbsp;|&nbsp; Data Scientist &nbsp;|&nbsp; Toronto, ON",
                    ParagraphStyle('footer', fontName='Helvetica', fontSize=8,
                                   textColor=GREY, alignment=TA_CENTER))]

doc.build(story)
print(f"Saved: {OUTPUT}")
