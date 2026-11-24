# Breast Cancer Detection and Data Analysis Using Machine Learning
AIM111 – Introduction to Data Science | Dexter Brokeshoulder | Approved final project

Educational analysis of the Wisconsin Diagnostic Breast Cancer dataset. Not for diagnosis, treatment or patient uploads.

## Setup (Windows, Python 3.11 or 3.12 recommended)
Open PowerShell in the extracted project directory:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open http://localhost:8501 if the browser does not launch. Stop with Ctrl+C. For app-only setup, install `requirements.txt` instead. On macOS/Linux use `python3 -m venv .venv` and `.venv/bin/python`.

## Dashboard
- Dataset Overview: 569 observations, 30 features, quality checks, class counts, schema and summary statistics.
- Feature Exploration: choose X/Y features and All/Malignant/Benign; scatterplot, box plot, class summary and correlation heatmap respond to filters.
- SVM Model Evaluation: fixed 70/30 stratified split, seed 42, training-only StandardScaler and linear SVC, accuracy/precision/recall/F1, confusion matrix and classification report. Filters do not retrain the model.

## Verification
```powershell
.\.venv\Scripts\python.exe -m unittest -v test_project
.\.venv\Scripts\python.exe -m jupyter lab
```
Open the notebook and select Run All. Section 8 compares the shared application analysis with the notebook. Expected test results: accuracy 98.25%, malignant precision 98.41%, recall 96.88%, F1 97.64%; matrix [[62,2],[1,106]] (malignant first). Dataset filters: All 569, Malignant 212, Benign 357. Verify the chart titles, axes, captions and filter behavior in the browser. See VERIFICATION.txt for actual execution status.

## Deploy (if required)
Create a GitHub repository and upload the extracted project files at its root (do not commit .venv). Commit app.py, analysis_core.py, requirements.txt, CSV and supporting files. Sign in to https://share.streamlit.io/, create an app from the repository and branch, select app.py as the entry point, choose Python 3.12 if available, and deploy. Check the public URL in a private browser window and repeat the dashboard checks. No live URL has been created here. Submit the approved proposal and final project files/link to Canvas as specified by your instructor; confirm submission receipt.

## Sources and limitations
- https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- Approved Exercise 32 proposal.

Historical dataset, single held-out split, no external clinical validation. Malignant is positive (0); benign is 1. False negatives and false positives are important limitations. No medical claims are made.
