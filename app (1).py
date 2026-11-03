import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="Universal Semiconductor Analyzer", page_icon="⚛️", layout="wide")

st.title("Universal Semiconductor E–k & DFT Analysis")
st.caption("Theoretical multi-band model + imported DFT band structure and DOS analysis")

with st.sidebar:
    st.header("Analysis mode")
    mode = st.radio("Choose mode", ["Theoretical multi-band model", "Import real DFT data"])
    st.divider()
    st.caption("DFT calculations are performed externally (e.g. Quantum ESPRESSO). This app analyzes the resulting data.")

def make_theoretical(material, eg, me, mh, kmax, npoints, nv, nc):
    k = np.linspace(-kmax, kmax, npoints)
    # Illustrative parabolic bands. Energies are not ab-initio results.
    bands = {}
    for i in range(nv):
        curvature = 3.809982 / (mh * (1 + 0.65*i))
        bands[f"VB{i+1}"] = -eg/2 - curvature * k**2 - 0.12*i
    for i in range(nc):
        curvature = 3.809982 / (me * (1 + 0.55*i))
        bands[f"CB{i+1}"] = eg/2 + curvature * k**2 + 0.16*i
    return k, pd.DataFrame(bands)

def plot_bands(x, df, title, vbm=None, cbm=None, labels=None):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[col], mode="lines", name=col, hovertemplate="k-distance=%{x:.5f}<br>Energy=%{y:.5f} eV<extra>"+col+"</extra>"))
    if vbm is not None: fig.add_hline(y=vbm, line_dash="dot", annotation_text="VBM")
    if cbm is not None: fig.add_hline(y=cbm, line_dash="dot", annotation_text="CBM")
    fig.update_layout(title=title, xaxis_title="Wave-vector path distance (Å⁻¹, or supplied units)", yaxis_title="Energy (eV)", hovermode="closest", legend_title="Bands")
    if labels:
        for pos, label in labels:
            fig.add_vline(x=pos, line_dash="dash", line_color="gray")
            fig.add_annotation(x=pos, y=1, yref="paper", text=label, showarrow=False, yshift=10)
    return fig

def analyze_bands(df, fermi):
    # Determine occupied/unoccupied states relative to Fermi level.
    vals = df.to_numpy(dtype=float)
    occupied = vals[vals <= fermi]
    unoccupied = vals[vals > fermi]
    if occupied.size == 0 or unoccupied.size == 0:
        return None, None, None
    vbm = float(np.max(occupied))
    cbm = float(np.min(unoccupied))
    return vbm, cbm, max(0.0, cbm-vbm)

if mode == "Theoretical multi-band model":
    st.subheader("1. Theoretical multi-band model")
    st.warning("This is an illustrative effective-mass model, not a DFT calculation and not a substitute for measured/ab-initio band eigenvalues.")
    c1,c2,c3 = st.columns(3)
    material = c1.text_input("Material name / formula", "Example semiconductor")
    eg = c2.number_input("Band gap Eg (eV)", min_value=0.01, value=1.12, step=0.01)
    me = c3.number_input("Electron effective mass (m₀)", min_value=0.01, value=0.26, step=0.01)
    c4,c5,c6 = st.columns(3)
    mh = c4.number_input("Hole effective mass (m₀)", min_value=0.01, value=0.39, step=0.01)
    kmax = c5.number_input("k range maximum (Å⁻¹)", min_value=0.01, value=0.5, step=0.01)
    npoints = c6.number_input("Number of k-points", min_value=51, max_value=2001, value=301, step=50)
    c7,c8 = st.columns(2)
    nv = c7.slider("Number of illustrative valence bands", 1, 8, 4)
    nc = c8.slider("Number of illustrative conduction bands", 1, 8, 4)
    if st.button("Generate theoretical E–k diagram", type="primary"):
        k, bands = make_theoretical(material, eg, me, mh, kmax, int(npoints), nv, nc)
        st.session_state["theory_k"] = k
        st.session_state["theory_bands"] = bands
        st.session_state["theory_name"] = material
    if "theory_bands" in st.session_state:
        bands = st.session_state["theory_bands"]
        k = st.session_state["theory_k"]
        st.plotly_chart(plot_bands(k, bands, f"{st.session_state['theory_name']} – illustrative theoretical bands"), use_container_width=True)
        st.download_button("Download band data CSV", bands.assign(k_distance=k).to_csv(index=False), "theoretical_bands.csv", "text/csv")
        st.info("Theoretical curves are generated from simplified parabolic equations. Real multi-band topology, valleys and symmetry points require DFT or validated band parameters.")

else:
    st.subheader("2. Real DFT data import")
    st.markdown("Upload band energies exported from Quantum ESPRESSO or another DFT package. Recommended CSV format: one row per k-point, one column per band, plus `k_distance`.")
    material = st.text_input("Material name / formula", "Silicon (Si)")
    fermi = st.number_input("Fermi energy (eV)", value=0.0, step=0.01, help="Use the Fermi energy from the same calculation and same energy reference as the band energies.")
    band_file = st.file_uploader("Upload band-structure CSV", type=["csv"])
    dos_file = st.file_uploader("Optional: upload DOS CSV", type=["csv"])
    if band_file:
        raw = pd.read_csv(band_file)
        st.write("Uploaded data preview", raw.head())
        if "k_distance" not in raw.columns:
            st.error("CSV must contain a `k_distance` column.")
        else:
            energy_cols = [c for c in raw.columns if c != "k_distance" and pd.api.types.is_numeric_dtype(raw[c])]
            if not energy_cols:
                st.error("No numeric band columns found. Use columns such as band_1, band_2, band_3.")
            else:
                bands = raw[energy_cols].apply(pd.to_numeric, errors="coerce")
                if bands.isna().any().any():
                    st.error("Band data contains missing or non-numeric values.")
                else:
                    x = pd.to_numeric(raw["k_distance"], errors="coerce")
                    vbm, cbm, gap = analyze_bands(bands, fermi)
                    st.plotly_chart(plot_bands(x, bands, f"{material} – imported DFT band structure", vbm, cbm), use_container_width=True)
                    if gap is None:
                        st.warning("Could not identify both occupied and unoccupied states. Check Fermi energy and energy reference.")
                    else:
                        a,b,c = st.columns(3)
                        a.metric("VBM (eV)", f"{vbm:.4f}")
                        b.metric("CBM (eV)", f"{cbm:.4f}")
                        c.metric("Gap relative to Fermi split (eV)", f"{gap:.4f}")
                        st.caption("This simple automatic estimate classifies states relative to the supplied Fermi level. For robust semiconductor gap analysis, provide band occupations or the correct number of occupied bands; metals and partially occupied bands need special handling.")
                    st.download_button("Download imported band data", raw.to_csv(index=False), "analyzed_band_data.csv", "text/csv")
    if dos_file:
        dos = pd.read_csv(dos_file)
        st.write("DOS data preview", dos.head())
        if not {"energy_eV", "dos"}.issubset(dos.columns):
            st.error("DOS CSV must contain columns `energy_eV` and `dos`.")
        else:
            dos["energy_eV"] = pd.to_numeric(dos["energy_eV"], errors="coerce")
            dos["dos"] = pd.to_numeric(dos["dos"], errors="coerce")
            dos = dos.dropna()
            fig = go.Figure(go.Scatter(x=dos["dos"], y=dos["energy_eV"], mode="lines", name="DOS"))
            fig.add_hline(y=fermi, line_dash="dash", annotation_text="Fermi level")
            fig.update_layout(title=f"{material} – Density of States", xaxis_title="DOS (states/eV)", yaxis_title="Energy (eV)")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Download DOS data", dos.to_csv(index=False), "dos_data.csv", "text/csv")

st.divider()
st.subheader("3. Quantum ESPRESSO workflow (external calculation)")
st.markdown("""
1. Prepare a valid crystal structure and element-specific UPF pseudopotentials.
2. Run a converged SCF calculation (`pw.x`).
3. Run a band-path calculation (`pw.x`, `calculation='bands'`) using the material's own symmetry path.
4. Process eigenvalues with `bands.x`; export a CSV with `k_distance, band_1, band_2, ...`.
5. Run NSCF + `dos.x` for DOS; export `energy_eV,dos`.
6. Upload those CSV files in **Import real DFT data** mode.

**Important:** Streamlit Cloud is intended here for visualization and analysis. It does not automatically run Quantum ESPRESSO. Running DFT requires a suitable local Linux/WSL or remote HPC environment, installed executables, pseudopotentials, and convergence testing.
""")
st.caption("Research integrity: always record source, code/version, pseudopotential, XC functional, cutoffs, k-mesh, structure, convergence criteria and energy reference.")
