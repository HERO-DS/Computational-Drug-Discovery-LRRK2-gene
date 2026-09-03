# LRRK2 Bioactivity Predictor

An end-to-end quantitative structure–activity relationship (QSAR) application that predicts the biological potency ($pIC_{50}$) of candidate chemical compounds against **Leucine-Rich Repeat Kinase 2 (LRRK2)**, a primary therapeutic target for Parkinson's disease.

The platform computes 1,024-bit ECFP4 (Morgan) molecular fingerprints using RDKit and estimates potency using a pre-trained Support Vector Regression (SVR) model.

---

## Technical Stack & Libraries

### Core Data Science & Cheminformatics
* **Python**: Core programming language.
* **RDKit**: Native molecular structure parsing, standardization, and 1,024-bit ECFP4 fingerprint generation (radius = 2).
* **Scikit-Learn**: Machine learning modeling, feature execution, and regression inferencing.
* **Joblib**: Serialization and loading of the pre-trained SVR pipeline (`model.pkl`).
* **NumPy & Pandas**: Array operations, matrix transformations, and tabular data management.

### Interface & Web Application
* **Streamlit**: Web application framework and dashboard layout.
* **Pillow (PIL)**: Render 2D chemical structures.

---

## Project Structure

* **app.py**: Main Streamlit web application script.
* **model.pkl**: Pre-trained SVR regression model.
* **requirements.txt**: Python dependencies list.
* **notebooks/**: Google Colab benchmarking & EDA notebooks.

---

## Target Biological Overview & Metrics

* **Target Protein**: LRRK2 (Leucine-Rich Repeat Kinase 2)
* **Target Endpoint**: Half-maximal inhibitory concentration ($pIC_{50} = -\log_{10}(IC_{50})$)
* **Activity Thresholds**:
  * **Active (High Potency)**: $pIC_{50} \ge 6.0$ ($IC_{50} \le 1.0\ \mu\text{M}$)
  * **Intermediate Potency**: $5.0 \le pIC_{50} < 6.0$
  * **Inactive (Low Potency)**: $pIC_{50} < 5.0$ ($IC_{50} > 10.0\ \mu\text{M}$)

---

## Features

1. **Batch Prediction**: Process compound files (`.csv`, `.txt`, `.smi`) in bulk with automatic delimiter detection and instant CSV export.
2. **Single Compound Analysis**: Input individual SMILES strings to calculate $pIC_{50}$ values instantly.
3. **Robust Input Parsing**: Automatically handles missing/invalid SMILES syntax, skips header rows, and strips formatting artifacts.

---

## Local Setup & Deployment Instructions

### 1. Clone Repository
```bash
git clone [https://github.com/HERO-DS/Computational-Drug-Discovery-LRRK2-gene.git](https://github.com/HERO-DS/Computational-Drug-Discovery-LRRK2-gene.git)
cd Computational-Drug-Discovery-LRRK2-gene
