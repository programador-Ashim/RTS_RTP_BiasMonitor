# RTS & RTP Real-Time AI Bias Monitor (Streamlit)

This project keeps your **same dashboard + bias monitoring + SHAP explainability** flow,
but changes the model to predict **TWO targets**:

- **RTS** = Return to Sport (clearance / safety)
- **RTP** = Return to Performance (sustained performance)

## Dataset (real)
Use the Athlete Injury & Performance dataset from Kaggle (same source you used in your slides).
If your Kaggle file has only one label, you can still run the app using the built-in **synthetic RTS/RTP demo data**.

## Run in VS Code (Windows / Mac)

### 1) Open folder
Open this folder in VS Code.

### 2) Create a virtual environment
**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Mac/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Start the app
```bash
streamlit run app_streamlit.py
```

## How to use
1. In the sidebar, either:
   - Upload a CSV/Excel dataset, OR
   - Toggle **Use synthetic demo data** (recommended for quick testing)
2. Under **Dataset mapping**, choose:
   - RTS target column
   - RTP target column
   - Protected attributes (gender, race, age_group, etc.)
3. Click **Train model**
4. Go to **Bias Monitoring** tab and click **Start**
5. Go to **Explainability** tab and select RTS or RTP to view SHAP explanations.
