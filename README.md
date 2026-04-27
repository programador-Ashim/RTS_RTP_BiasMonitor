# Real-Time Bias-Aware AI for Readiness Monitoring

This project is a **Streamlit-based machine learning dashboard** for monitoring athlete readiness, fairness, explainability, and dataset health.

It supports:
- Uploading CSV or Excel datasets
- Using synthetic demo data
- Training **Gradient Boosting** and **Random Forest** models
- Predicting **RTS (Return to Sport)** and **RTP (Return to Performance)**
- Real-time fairness monitoring using **Demographic Parity** and **Equalized Odds**
- SHAP-based explainability
- Dataset health checks
- Exporting monitoring history

## Project structure

Make sure your project folder contains these files and folders:

```text
project_folder/
│
├── app.py                 # Main Streamlit app file
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── src/
    ├── data.py
    ├── preprocess.py
    ├── train.py
    ├── monitor.py
    ├── realtime.py
    └── health.py
```

The system uses these custom modules:
- `src.data`
- `src.preprocess`
- `src.train`
- `src.monitor`
- `src.realtime`
- `src.health`

So the app will not run unless the `src` folder and these Python files are present.

## Python version

Use:

```bash
Python 3.10 or Python 3.11
```

## Install required packages

Open terminal inside your project folder and run:

```bash
pip install -r requirements.txt
```

If that does not work, try:

```bash
python -m pip install -r requirements.txt
```

Or on Mac/Linux:

```bash
python3 -m pip install -r requirements.txt
```

## Main packages used

This project uses the following libraries:
- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `scikit-learn`
- `imbalanced-learn`
- `openpyxl`
- `xlrd`
- `matplotlib`
- `shap`
- `streamlit-option-menu`

## How to run the project

### Step 1: Save your main file
Save your main Streamlit code as:

```text
app.py
```

### Step 2: Open terminal in the project folder
Example:

```bash
cd path/to/your/project_folder
```

### Step 3: Run the Streamlit app

```bash
streamlit run app_streamlit.py
```

If that does not work, use:

```bash
python -m streamlit run app_streamlit.py
```

Or:

```bash
python3 -m streamlit run app.py
```

## What the app does

### 1. Dashboard
Shows:
- Model performance metrics
- RTS and RTP evaluation results
- General system overview

### 2. Bias Monitoring
Shows:
- Real-time streaming fairness monitoring
- Demographic Parity difference
- Equalized Odds difference
- Root-cause attribution
- Batch-level fairness details

### 3. Explainability
Shows:
- SHAP summary plot
- Single prediction explanation
- Feature contribution table

### 4. Data Health
Shows:
- Missing values
- Duplicate rates
- Target balance
- Data preview
- Cleaned dataset view

### 5. Settings
Shows:
- Current model configuration
- Monitoring threshold
- Batch size
- Export option for fairness history

## Input data format

The system can work with:
- CSV files
- Excel files (`.xlsx`, `.xls`)
- Synthetic demo data

Possible useful columns include:
- `gender`
- `race`
- `age`
- `age_group`
- `fatigue_score`
- `recovery_score`
- `performance_score`
- `injury_occurred`
- `rts`
- `rtp`

The app can also try to automatically create `rts` and `rtp` if they are missing.

## Features of the system

- Automatic dataset loading
- Data cleaning support
- Smart RTS/RTP target creation
- Gradient Boosting model training
- Random Forest model training
- Real-time bias monitoring
- SHAP explainability
- Plotly charts and Streamlit interface

## If you get errors

### Missing module error
Install the package again:

```bash
pip install -r requirements.txt
```

### SHAP error
Try:

```bash
pip install shap matplotlib
```

### Excel file error
Try:

```bash
pip install openpyxl xlrd
```

### Streamlit not found
Try:

```bash
python -m pip install streamlit
```

## Recommended setup with virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Keep the `src` folder in the same project directory as `app.py`
- Make sure column names in your dataset are correct
- If your dataset does not contain RTS or RTP, the app will try to generate them automatically
- Real-time monitoring requires the model to be trained first
- Explainability requires SHAP and matplotlib to be installed

## Quick run commands

```bash
pip install -r requirements.txt
streamlit run app.py
```
