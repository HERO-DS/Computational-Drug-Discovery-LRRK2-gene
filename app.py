import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from rdkit import Chem
from rdkit.Chem import AllChem

# Page configuration
st.set_page_config(
    page_title="LRRK2 Bioactivity Batch Predictor",
    page_icon="🧪",
    layout="wide"
)

# Load pre-trained SVR model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error("Error loading 'model.pkl'. Please ensure 'model.pkl' is in the root directory.")
    st.stop()

# Helper: Compute ECFP4 fingerprint for a single SMILES
def smiles_to_ecfp4(smiles, n_bits=1024, radius=2):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)

# Helper: Download link generator
def filedownload(df, filename="lrrk2_predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="background-color:#0e76a8;color:white;padding:8px 16px;border-radius:4px;text-decoration:none;font-weight:bold;">⬇️ Download Batch Predictions CSV</a>'
    return href

# Title & Description
st.title("🧪 LRRK2 Inhibitor Bioactivity Predictor")
st.markdown("""
Predict the biological potency ($pIC_{50}$) of candidate molecules against **LRRK2 (Leucine-Rich Repeat Kinase 2)** using Support Vector Regression (SVR) and 1,024-bit ECFP4 (Morgan) fingerprints.

---
""")

# Sidebar Navigation & Upload
st.sidebar.header("1. Batch Processing Options")

# File Uploader (Accepts .txt, .csv, or .smi)
uploaded_file = st.sidebar.file_uploader(
    "Upload compound file (CSV or TXT)", 
    type=['txt', 'csv', 'smi'],
    help="File should contain SMILES and optional compound names/IDs (space, tab, or comma separated)."
)

# Example Data Generator
st.sidebar.markdown("---")
st.sidebar.subheader("Need Example Data?")
sample_data = "SMILES molecule_name\nCc1ccc(cc1)C2=CC(=O)c3c(c(cc(c3O2)O)O)O Compound_1\nO=C(Nc1ccc(Cl)cc1)c2cccnc2 Compound_2\nCN1CCN(Cc2ccc(cc2)C(=O)Nc3ccc(C)c(c3)N4C(=O)c5ccccc5C4=O)CC1 Compound_3"
st.sidebar.download_button(
    label="📥 Download Example Input File",
    data=sample_data,
    file_name="example_lrrk2_compounds.txt",
    mime="text/plain"
)

# MAIN CONTENT
if uploaded_file is not None:
    st.header("1. Input Data Summary")
    
    # Read file automatically based on extension/delimiter
    try:
        # Attempt to read whitespace/tab separated file
        load_data = pd.read_csv(uploaded_file, sep=r'\s+|,', engine='python', header=None)
        if load_data.shape[1] >= 2:
            load_data.columns = ['SMILES', 'molecule_name']
        else:
            load_data.columns = ['SMILES']
            load_data['molecule_name'] = [f"Molecule_{i+1}" for i in range(len(load_data))]
    except Exception as e:
        st.error(f"Error parsing uploaded file: {e}")
        st.stop()
        
    st.dataframe(load_data, use_container_width=True)

    if st.button("Run Batch Prediction", type="primary"):
        with st.spinner("Converting molecules to 1,024-bit ECFP4 fingerprints & running SVR..."):
            
            fingerprints = []
            valid_indices = []
            failed_smiles = []

            for idx, row in load_data.iterrows():
                fp = smiles_to_ecfp4(row['SMILES'])
                if fp is not None:
                    fingerprints.append(fp)
                    valid_indices.append(idx)
                else:
                    failed_smiles.append(row['SMILES'])

            if len(fingerprints) == 0:
                st.error("None of the SMILES strings in the file could be parsed by RDKit.")
            else:
                # Convert to numpy array shape (N, 1024)
                X_batch = np.array(fingerprints)
                
                # Predict pIC50
                predictions = model.predict(X_batch)
                
                # Build Result DataFrame
                results_df = load_data.iloc[valid_indices].copy()
                results_df['Predicted_pIC50'] = np.round(predictions, 3)
                
                # Add Classifications
                conditions = [
                    (results_df['Predicted_pIC50'] >= 6.0),
                    (results_df['Predicted_pIC50'] >= 5.0) & (results_df['Predicted_pIC50'] < 6.0),
                    (results_df['Predicted_pIC50'] < 5.0)
                ]
                choices = ['Active (High Potency)', 'Intermediate Potency', 'Inactive (Low Potency)']
                results_df['Bioactivity_Status'] = np.select(conditions, choices, default='Unknown')

                # Display Results
                st.header("2. Prediction Output")
                st.dataframe(results_df, use_container_width=True)
                
                # Download Button
                st.markdown("### 3. Download Results")
                st.markdown(filedownload(results_df), unsafe_allow_html=True)

                if len(failed_smiles) > 0:
                    st.warning(f"⚠️ Could not parse {len(failed_smiles)} invalid SMILES entries.")

else:
    # Single Prediction Fallback Mode
    st.info("💡 **Upload a compound file in the sidebar to run batch predictions**, or test a single SMILES string below:")
    
    st.subheader("Single Molecule Predictor")
    single_smiles = st.text_input("Enter SMILES String:", value="Cc1ccc(cc1)C2=CC(=O)c3c(c(cc(c3O2)O)O)O")
    
    if st.button("Predict Single Compound"):
        if single_smiles.strip():
            fp = smiles_to_ecfp4(single_smiles)
            if fp is None:
                st.error("Invalid SMILES string. RDKit could not parse the chemical structure.")
            else:
                pred_pIC50 = float(model.predict(fp.reshape(1, -1))[0])
                col1, col2 = st.columns(2)
                col1.metric("Predicted pIC50", f"{pred_pIC50:.3f}")
                if pred_pIC50 >= 6.0:
                    col2.success("Status: Active (High Potency)")
                elif pred_pIC50 >= 5.0:
                    col2.warning("Status: Intermediate Potency")
                else:
                    col2.error("Status: Inactive (Low Potency)")