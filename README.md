# Universal Semiconductor E–k & DFT Analysis

## Files
- `app.py`: Streamlit application.
- `requirements.txt`: Python dependencies.
- `sample_band_structure.csv`: illustrative sample data for testing the upload interface.
- `sample_dos.csv`: illustrative DOS sample data.

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload all files from this folder to the repository root.
3. In https://share.streamlit.io/ choose the repository, branch, and `app.py`.
4. Deploy.

## DFT data format expected by the app

Band CSV: one row per k-point, with `k_distance` followed by numeric band columns:
`k_distance,band_1,band_2,band_3,...`

DOS CSV: two columns:
`energy_eV,dos`

The sample CSV files are synthetic demonstration data, NOT actual silicon or any other material's DFT results.

## Important scientific limitations
- The theoretical mode generates simplified illustrative parabolic bands from a gap and effective masses. It does not reproduce a real material's complete multi-band structure.
- The DFT mode visualizes imported band eigenvalues and DOS. It does not run Quantum ESPRESSO on Streamlit Cloud.
- Correct VBM/CBM/gap determination requires a consistent energy reference and correct occupations / number of occupied bands. The app's Fermi-level split is a basic estimate, not a substitute for checking occupations, k-point sampling, spin, smearing, and band connectivity.
- High-symmetry k-paths depend on the crystal structure and must be generated for each material, not copied from silicon.
- For publication-quality work, test convergence of plane-wave cutoffs and k-mesh, validate pseudopotentials, document the XC functional, and compare with experimental or higher-level calculations where appropriate.
