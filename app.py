import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="E-k Diagram Analyzer",
    page_icon="⚛️",
    layout="wide"
)


# =========================================================
# CUSTOM DESIGN
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
    radial-gradient(circle at 10% 10%, rgba(37,99,235,0.12), transparent 25%),
    radial-gradient(circle at 90% 15%, rgba(124,58,237,0.12), transparent 25%),
    linear-gradient(135deg,#f8fbff,#eef4ff,#faf5ff);
}

.main-title {
    text-align:center;
    font-size:42px;
    font-weight:800;
    color:#172554;
    margin-bottom:5px;
}

.subtitle {
    text-align:center;
    color:#475569;
    font-size:17px;
    margin-bottom:25px;
}

.card {
    background:rgba(255,255,255,0.92);
    padding:22px;
    border-radius:20px;
    border:1px solid rgba(148,163,184,0.25);
    box-shadow:0 8px 25px rgba(15,23,42,0.08);
    margin-bottom:20px;
}

.result-card {
    background:linear-gradient(135deg,#ffffff,#eff6ff);
    padding:22px;
    border-radius:20px;
    border:1px solid #bfdbfe;
    box-shadow:0 8px 25px rgba(37,99,235,0.08);
}

.direct-result {
    border-left:7px solid #16a34a;
}

.indirect-result {
    border-left:7px solid #2563eb;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PHYSICAL CONSTANTS
# =========================================================

H_BAR = 1.054571817e-34       # J s
M0 = 9.1093837139e-31         # kg
EV_TO_J = 1.602176634e-19     # J


# =========================================================
# UNIT CONVERSION FUNCTIONS
# =========================================================

def energy_to_ev(value, unit):

    if unit == "eV":
        return float(value)

    if unit == "J":
        return float(value) / EV_TO_J

    return float(value)


def mass_to_kg(value, unit):

    value = float(value)

    if unit == "kg":
        return value

    if unit == "g":
        return value * 1e-3

    if unit == "mg":
        return value * 1e-6

    if unit == "m₀":
        return value * M0

    return value


def k_to_angstrom(value, unit):

    value = float(value)

    if unit == "Å⁻¹":
        return value

    if unit == "m⁻¹":
        return value / 1e10

    return value


# =========================================================
# THEORETICAL E-k MODEL
# =========================================================

def theoretical_bands(
    eg_ev,
    electron_mass,
    hole_mass,
    k_values,
    cbm_k=0.0
):

    k_meter = k_values * 1e10
    cbm_k_meter = cbm_k * 1e10

    valence_energy_j = (
        -(H_BAR ** 2) *
        k_meter ** 2 /
        (2 * hole_mass)
    )

    conduction_energy_j = (
        eg_ev * EV_TO_J
        +
        (H_BAR ** 2) *
        (k_meter - cbm_k_meter) ** 2 /
        (2 * electron_mass)
    )

    valence_energy_ev = valence_energy_j / EV_TO_J
    conduction_energy_ev = conduction_energy_j / EV_TO_J

    return valence_energy_ev, conduction_energy_ev


# =========================================================
# DFT DATA PROCESSING
# =========================================================

def prepare_dft_data(df):

    df = df.copy()

    # Convert all columns to numeric where possible
    for column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(how="all")

    if df.shape[1] < 2:
        raise ValueError(
            "CSV must contain k column and at least one band."
        )

    # First column = k
    k_column = df.columns[0]

    df = df.rename(
        columns={k_column: "k"}
    )

    df = df.dropna(
        subset=["k"]
    )

    return df


# =========================================================
# FIND VBM AND CBM
# =========================================================

def calculate_vbm_cbm(df, fermi_level):

    k_values = df["k"].values

    band_columns = [
        column
        for column in df.columns
        if column != "k"
    ]

    occupied = []
    unoccupied = []

    for band in band_columns:

        energies = df[band].values

        for i, energy in enumerate(energies):

            if np.isnan(energy):
                continue

            k = k_values[i]

            # Occupied state
            if energy <= fermi_level:
                occupied.append(
                    (energy, k, band)
                )

            # Unoccupied state
            if energy > fermi_level:
                unoccupied.append(
                    (energy, k, band)
                )

    if len(occupied) == 0:
        raise ValueError(
            "No occupied states found. Check Fermi level."
        )

    if len(unoccupied) == 0:
        raise ValueError(
            "No unoccupied states found. Check Fermi level."
        )

    # Highest occupied state
    vbm = max(
        occupied,
        key=lambda x: x[0]
    )

    # Lowest unoccupied state
    cbm = min(
        unoccupied,
        key=lambda x: x[0]
    )

    E_VBM = vbm[0]
    k_VBM = vbm[1]
    VBM_band = vbm[2]

    E_CBM = cbm[0]
    k_CBM = cbm[1]
    CBM_band = cbm[2]

    Eg = E_CBM - E_VBM

    delta_k = abs(
        k_CBM - k_VBM
    )

    # Numerical tolerance
    tolerance = 1e-6

    if delta_k <= tolerance:

        band_type = "DIRECT BAND GAP"

    else:

        band_type = "INDIRECT BAND GAP"

    return {
        "E_VBM": E_VBM,
        "k_VBM": k_VBM,
        "VBM_band": VBM_band,

        "E_CBM": E_CBM,
        "k_CBM": k_CBM,
        "CBM_band": CBM_band,

        "Eg": Eg,

        "delta_k": delta_k,

        "band_type": band_type
    }


# =========================================================
# THEORETICAL GRAPH
# =========================================================

def create_theoretical_graph(
    k,
    valence,
    conduction,
    material,
    k_vbm,
    k_cbm,
    Eg
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=k,
            y=valence,
            mode="lines",
            name="Valence Band",
            line=dict(width=3),
            hovertemplate=
            "Valence Band<br>"
            "k = %{x:.6f} Å⁻¹<br>"
            "Energy = %{y:.6f} eV"
            "<extra></extra>"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=k,
            y=conduction,
            mode="lines",
            name="Conduction Band",
            line=dict(width=3),
            hovertemplate=
            "Conduction Band<br>"
            "k = %{x:.6f} Å⁻¹<br>"
            "Energy = %{y:.6f} eV"
            "<extra></extra>"
        )
    )

    vbm_energy = max(valence)
    cbm_energy = min(conduction)

    fig.add_trace(
        go.Scatter(
            x=[k_vbm],
            y=[vbm_energy],
            mode="markers+text",
            text=["VBM"],
            textposition="top center",
            name="VBM",
            marker=dict(
                size=13,
                symbol="diamond"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[k_cbm],
            y=[cbm_energy],
            mode="markers+text",
            text=["CBM"],
            textposition="top center",
            name="CBM",
            marker=dict(
                size=13,
                symbol="diamond"
            )
        )
    )

    # Gap line
    fig.add_trace(
        go.Scatter(
            x=[k_vbm, k_cbm],
            y=[vbm_energy, cbm_energy],
            mode="lines",
            name="Band Gap",
            line=dict(
                dash="dash",
                width=2
            )
        )
    )

    # Band gap label
    fig.add_annotation(
        x=(k_vbm + k_cbm) / 2,
        y=(vbm_energy + cbm_energy) / 2,
        text=f"Eg = {Eg:.4f} eV",
        showarrow=False
    )

    if abs(k_vbm - k_cbm) < 1e-10:

        band_type = "DIRECT BAND GAP"

    else:

        band_type = "INDIRECT BAND GAP"

    fig.add_annotation(
        x=0.98,
        y=0.97,
        xref="paper",
        yref="paper",
        text=band_type,
        showarrow=False,
        xanchor="right",
        font=dict(size=16)
    )

    fig.update_layout(
        title=f"E–k Diagram — {material}",
        xaxis_title="Wave Vector k (Å⁻¹)",
        yaxis_title="Energy (eV)",
        template="plotly_white",
        height=650,
        hovermode="closest"
    )

    fig.add_hline(
        y=0,
        line_dash="dot"
    )

    return fig


# =========================================================
# DFT GRAPH
# =========================================================

def create_dft_graph(
    df,
    material,
    fermi_level,
    result
):

    fig = go.Figure()

    band_columns = [
        column
        for column in df.columns
        if column != "k"
    ]

    for band in band_columns:

        fig.add_trace(
            go.Scatter(
                x=df["k"],
                y=df[band],
                mode="lines",
                name=str(band),
                hovertemplate=
                f"{band}<br>"
                "k = %{x:.6f} Å⁻¹<br>"
                "Energy = %{y:.6f} eV"
                "<extra></extra>"
            )
        )

    # VBM
    fig.add_trace(
        go.Scatter(
            x=[result["k_VBM"]],
            y=[result["E_VBM"]],
            mode="markers+text",
            text=["VBM"],
            textposition="top center",
            name="VBM",
            marker=dict(
                size=14,
                symbol="diamond"
            ),
            hovertemplate=
            f"VBM<br>"
            f"k = {result['k_VBM']:.6f} Å⁻¹<br>"
            f"E = {result['E_VBM']:.6f} eV"
            "<extra></extra>"
        )
    )

    # CBM
    fig.add_trace(
        go.Scatter(
            x=[result["k_CBM"]],
            y=[result["E_CBM"]],
            mode="markers+text",
            text=["CBM"],
            textposition="top center",
            name="CBM",
            marker=dict(
                size=14,
                symbol="diamond"
            ),
            hovertemplate=
            f"CBM<br>"
            f"k = {result['k_CBM']:.6f} Å⁻¹<br>"
            f"E = {result['E_CBM']:.6f} eV"
            "<extra></extra>"
        )
    )

    # VBM-CBM separation
    fig.add_trace(
        go.Scatter(
            x=[
                result["k_VBM"],
                result["k_CBM"]
            ],
            y=[
                result["E_VBM"],
                result["E_CBM"]
            ],
            mode="lines",
            name="VBM–CBM",
            line=dict(
                dash="dash",
                width=2
            ),
            hoverinfo="skip"
        )
    )

    # Fermi level
    fig.add_hline(
        y=fermi_level,
        line_dash="dot",
        annotation_text=
        f"Fermi Level = {fermi_level:.4f} eV"
    )

    # Direct/Indirect label
    fig.add_annotation(
        x=0.98,
        y=0.97,
        xref="paper",
        yref="paper",
        text=result["band_type"],
        showarrow=False,
        xanchor="right",
        font=dict(size=16)
    )

    fig.update_layout(
        title=f"Multi-band E–k Diagram — {material}",
        xaxis_title="Wave Vector k (Å⁻¹)",
        yaxis_title="Energy (eV)",
        template="plotly_white",
        height=700,
        hovermode="closest"
    )

    return fig


# =========================================================
# DOS GRAPH
# =========================================================

def create_dos_graph(
    dos_df,
    material,
    fermi_level
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dos_df["energy"],
            y=dos_df["dos"],
            mode="lines",
            name="DOS",
            line=dict(width=3),
            hovertemplate=
            "Energy = %{x:.6f} eV<br>"
            "DOS = %{y:.6f}"
            "<extra></extra>"
        )
    )

    fig.add_vline(
        x=fermi_level,
        line_dash="dot",
        annotation_text=
        f"Fermi Level = {fermi_level:.4f} eV"
    )

    fig.update_layout(
        title=f"DOS — {material}",
        xaxis_title="Energy (eV)",
        yaxis_title="Density of States",
        template="plotly_white",
        height=500
    )

    return fig


# =========================================================
# PDF REPORT
# =========================================================

def create_pdf_report(
    material,
    symbol,
    source,
    fermi,
    result
):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=20,
        leading=24
    )

    normal = ParagraphStyle(
        "Normal",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14
    )

    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=14
    )

    story = []

    story.append(
        Paragraph(
            "E–k Band Structure Analysis Report",
            title
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            f"<b>Material:</b> {material} ({symbol})",
            normal
        )
    )

    story.append(
        Paragraph(
            f"<b>Data Source:</b> {source}",
            normal
        )
    )

    story.append(
        Paragraph(
            f"<b>Fermi Level:</b> {fermi:.6f} eV",
            normal
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "Calculation Result",
            heading
        )
    )

    table_data = [
        ["Parameter", "Value"],

        [
            "VBM Energy",
            f"{result['E_VBM']:.8f} eV"
        ],

        [
            "VBM k-position",
            f"{result['k_VBM']:.8f} Å⁻¹"
        ],

        [
            "VBM Band",
            result["VBM_band"]
        ],

        [
            "CBM Energy",
            f"{result['E_CBM']:.8f} eV"
        ],

        [
            "CBM k-position",
            f"{result['k_CBM']:.8f} Å⁻¹"
        ],

        [
            "CBM Band",
            result["CBM_band"]
        ],

        [
            "Band Gap Eg",
            f"{result['Eg']:.8f} eV"
        ],

        [
            "Band Gap Type",
            result["band_type"]
        ],

        [
            "Wave Vector Separation",
            f"{result['delta_k']:.8f} Å⁻¹"
        ]
    ]

    table = Table(
        table_data,
        colWidths=[
            2.5 * inch,
            3.5 * inch
        ]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0,0),
                (-1,0),
                colors.lightgrey
            ),

            (
                "GRID",
                (0,0),
                (-1,-1),
                0.5,
                colors.grey
            ),

            (
                "FONTNAME",
                (0,0),
                (-1,0),
                "Helvetica-Bold"
            ),

            (
                "VALIGN",
                (0,0),
                (-1,-1),
                "MIDDLE"
            ),

            (
                "PADDING",
                (0,0),
                (-1,-1),
                6
            )
        ])
    )

    story.append(table)

    story.append(
        Spacer(1,20)
    )

    story.append(
        Paragraph(
            "Mathematical Relations",
            heading
        )
    )

    formulas = [
        "E<sub>VBM</sub> = maximum occupied-state energy",
        "E<sub>CBM</sub> = minimum unoccupied-state energy",
        "E<sub>g</sub> = E<sub>CBM</sub> − E<sub>VBM</sub>",
        "Direct gap: k<sub>VBM</sub> = k<sub>CBM</sub>",
        "Indirect gap: k<sub>VBM</sub> ≠ k<sub>CBM</sub>"
    ]

    for formula in formulas:

        story.append(
            Paragraph(
                "• " + formula,
                normal
            )
        )

    story.append(
        Spacer(1,20)
    )

    story.append(
        Paragraph(
            "<b>Important:</b> This application analyzes supplied "
            "theoretical, DFT or experimental data. It does not perform "
            "a DFT calculation itself.",
            normal
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">⚛️ E–k DIAGRAM ANALYZER</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Semiconductor Band Structure • VBM • CBM • Band Gap • DOS'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Analysis Mode")

    mode = st.radio(
        "Select Mode",
        [
            "Theoretical E–k Model",
            "Real / DFT Data Analysis"
        ]
    )

    st.markdown("---")

    st.header("🔄 Unit Converter")

    # Energy
    st.subheader("Energy")

    energy_value = st.number_input(
        "Energy Value",
        value=1.0
    )

    energy_from = st.selectbox(
        "From",
        ["eV", "J"],
        key="energy_from"
    )

    energy_to = st.selectbox(
        "To",
        ["eV", "J"],
        key="energy_to"
    )

    energy_ev = energy_to_ev(
        energy_value,
        energy_from
    )

    if energy_to == "J":

        output_energy = (
            energy_ev * EV_TO_J
        )

    else:

        output_energy = energy_ev

    st.info(
        f"{output_energy:.8g} {energy_to}"
    )

    # Mass
    st.subheader("Mass")

    mass_value = st.number_input(
        "Mass Value",
        value=1.0
    )

    mass_from = st.selectbox(
        "From",
        ["kg", "g", "mg", "m₀"],
        key="mass_from"
    )

    mass_to = st.selectbox(
        "To",
        ["kg", "g", "mg", "m₀"],
        key="mass_to"
    )

    mass_kg = mass_to_kg(
        mass_value,
        mass_from
    )

    if mass_to == "kg":
        mass_output = mass_kg

    elif mass_to == "g":
        mass_output = mass_kg * 1000

    elif mass_to == "mg":
        mass_output = mass_kg * 1e6

    else:
        mass_output = mass_kg / M0

    st.info(
        f"{mass_output:.8g} {mass_to}"
    )

    # Wave vector
    st.subheader("Wave Vector")

    k_value = st.number_input(
        "k Value",
        value=1.0
    )

    k_from = st.selectbox(
        "From",
        ["Å⁻¹", "m⁻¹"],
        key="k_from"
    )

    k_to = st.selectbox(
        "To",
        ["Å⁻¹", "m⁻¹"],
        key="k_to"
    )

    k_ang = k_to_angstrom(
        k_value,
        k_from
    )

    if k_to == "Å⁻¹":
        k_output = k_ang
    else:
        k_output = k_ang * 1e10

    st.info(
        f"{k_output:.8g} {k_to}"
    )

    st.markdown("---")

    st.caption(
        "Scientific calculations are performed directly by Python. "
        "No AI/API is used."
    )


# =========================================================
# THEORETICAL MODE
# =========================================================

if mode == "Theoretical E–k Model":

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🧪 Material Details")

    col1, col2 = st.columns(2)

    with col1:

        material = st.text_input(
            "Material Name",
            "Silicon"
        )

        symbol = st.text_input(
            "Material Symbol",
            "Si"
        )

    with col2:

        Eg_value = st.number_input(
            "Band Gap Energy Eg",
            min_value=0.0,
            value=1.12
        )

        Eg_unit = st.selectbox(
            "Energy Unit",
            ["eV", "J"]
        )

    col1, col2 = st.columns(2)

    with col1:

        electron_mass_value = st.number_input(
            "Electron Effective Mass mₑ*",
            min_value=0.000001,
            value=0.26
        )

        electron_mass_unit = st.selectbox(
            "Electron Mass Unit",
            ["m₀", "kg", "g", "mg"]
        )

    with col2:

        hole_mass_value = st.number_input(
            "Hole Effective Mass mₕ*",
            min_value=0.000001,
            value=0.39
        )

        hole_mass_unit = st.selectbox(
            "Hole Mass Unit",
            ["m₀", "kg", "g", "mg"]
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        k_min = st.number_input(
            "k Minimum",
            value=-1.0
        )

    with col2:

        k_max = st.number_input(
            "k Maximum",
            value=1.0
        )

    with col3:

        number_points = st.number_input(
            "Number of Points",
            min_value=100,
            max_value=5000,
            value=800,
            step=100
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader(
        "📐 Theoretical Band Arrangement"
    )

    st.info(
        "Direct/Indirect is used here only to construct the theoretical "
        "model. In Real/DFT mode the program automatically determines "
        "Direct or Indirect from VBM and CBM positions."
    )

    theoretical_type = st.radio(
        "Band arrangement",
        [
            "Direct",
            "Indirect"
        ],
        horizontal=True
    )

    if theoretical_type == "Indirect":

        cbm_k = st.number_input(
            "CBM k-position (Å⁻¹)",
            value=0.80
        )

    else:

        cbm_k = 0.0

    st.markdown("</div>", unsafe_allow_html=True)

    # Generate
    if st.button(
        "🚀 Generate E–k Diagram",
        type="primary",
        use_container_width=True
    ):

        if k_max <= k_min:

            st.error(
                "k Maximum must be greater than k Minimum."
            )

            st.stop()

        try:

            Eg_ev = energy_to_ev(
                Eg_value,
                Eg_unit
            )

            me = mass_to_kg(
                electron_mass_value,
                electron_mass_unit
            )

            mh = mass_to_kg(
                hole_mass_value,
                hole_mass_unit
            )

            k = np.linspace(
                k_min,
                k_max,
                int(number_points)
            )

            valence, conduction = theoretical_bands(
                Eg_ev,
                me,
                mh,
                k,
                cbm_k
            )

            k_vbm = 0.0

            k_cbm = cbm_k

            vbm_energy = max(
                valence
            )

            cbm_energy = min(
                conduction
            )

            calculated_Eg = (
                cbm_energy -
                vbm_energy
            )

            fig = create_theoretical_graph(
                k,
                valence,
                conduction,
                material,
                k_vbm,
                k_cbm,
                calculated_Eg
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # Result
            if abs(
                k_vbm - k_cbm
            ) < 1e-10:

                band_type = (
                    "DIRECT BAND GAP"
                )

                result_class = (
                    "direct-result"
                )

            else:

                band_type = (
                    "INDIRECT BAND GAP"
                )

                result_class = (
                    "indirect-result"
                )

            st.markdown(
                f'<div class="result-card {result_class}">',
                unsafe_allow_html=True
            )

            st.subheader(
                "📊 Calculation Result"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "VBM Energy",
                    f"{vbm_energy:.6f} eV"
                )

                st.write(
                    f"kᵥᵦₘ = {k_vbm:.6f} Å⁻¹"
                )

            with c2:

                st.metric(
                    "CBM Energy",
                    f"{cbm_energy:.6f} eV"
                )

                st.write(
                    f"k꜀ᵦₘ = {k_cbm:.6f} Å⁻¹"
                )

            with c3:

                st.metric(
                    "Band Gap Eg",
                    f"{calculated_Eg:.6f} eV"
                )

                st.write(
                    band_type
                )

            st.markdown(
                f"""
### Mathematical Result

\[
E_{{VBM}} = {vbm_energy:.6f}\ eV
\]

\[
E_{{CBM}} = {cbm_energy:.6f}\ eV
\]

\[
E_g = E_{{CBM}} - E_{{VBM}}
\]

\[
E_g = {cbm_energy:.6f} - ({vbm_energy:.6f})
\]

\[
\boxed{{E_g = {calculated_Eg:.6f}\ eV}}
\]

\[
\Delta k =
|k_{{CBM}}-k_{{VBM}}|
=
{abs(k_cbm-k_vbm):.6f}\ Å^{{-1}}
\]

### Result

**{band_type}**
"""
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            result_df = pd.DataFrame([
                {
                    "Material": material,
                    "Symbol": symbol,
                    "VBM Energy (eV)": vbm_energy,
                    "VBM k (Å^-1)": k_vbm,
                    "CBM Energy (eV)": cbm_energy,
                    "CBM k (Å^-1)": k_cbm,
                    "Band Gap Eg (eV)": calculated_Eg,
                    "Band Gap Type": band_type
                }
            ])

            st.download_button(
                "📊 Download CSV Result",
                result_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=
                f"{symbol}_E-k_result.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as error:

            st.error(
                f"Calculation Error: {error}"
            )


# =========================================================
# REAL / DFT MODE
# =========================================================

else:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader(
        "🔬 Material Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        material = st.text_input(
            "Material Name",
            "Silicon"
        )

        symbol = st.text_input(
            "Material Symbol",
            "Si"
        )

    with col2:

        data_source = st.selectbox(
            "Data Source",
            [
                "DFT Calculation",
                "Experimental",
                "Other"
            ]
        )

        k_unit = st.selectbox(
            "Uploaded k Unit",
            [
                "Å⁻¹",
                "m⁻¹"
            ]
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # E-k CSV
    # -----------------------------------------------------

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader(
        "📂 Multi-band E–k Data"
    )

    st.info(
        "CSV format: first column = k, remaining columns = "
        "band energies in eV."
    )

    ek_file = st.file_uploader(
        "Upload E–k CSV",
        type=["csv"]
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    if ek_file is not None:

        try:

            raw_df = pd.read_csv(
                ek_file
            )

            ek_df = prepare_dft_data(
                raw_df
            )

            # Convert k to Å^-1
            if k_unit == "m⁻¹":

                ek_df["k"] = (
                    ek_df["k"] /
                    1e10
                )

            st.success(
                f"Data loaded successfully — "
                f"{len(ek_df)} k-points, "
                f"{len(ek_df.columns)-1} bands."
            )

            st.dataframe(
                ek_df.head(10),
                use_container_width=True
            )

            # ------------------------------------------------
            # Fermi Level
            # ------------------------------------------------

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.subheader(
                "⚡ Fermi Level"
            )

            fermi_level = st.number_input(
                "Fermi Level E_F (eV)",
                value=0.0
            )

            st.markdown(
                """
**Occupation rule used by the analyzer:**

\[
E \leq E_F
\]

→ Occupied states

\[
E > E_F
\]

→ Unoccupied states
"""
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # DOS
            # ------------------------------------------------

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.subheader(
                "📉 DOS Data — Optional"
            )

            st.info(
                "DOS CSV should contain two columns: "
                "Energy and DOS."
            )

            dos_file = st.file_uploader(
                "Upload DOS CSV",
                type=["csv"]
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            if st.button(
                "🚀 Analyze Band Structure",
                type="primary",
                use_container_width=True
            ):

                try:

                    result = calculate_vbm_cbm(
                        ek_df,
                        fermi_level
                    )

                    # -----------------------------------------
                    # E-k GRAPH
                    # -----------------------------------------

                    fig = create_dft_graph(
                        ek_df,
                        material,
                        fermi_level,
                        result
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    # -----------------------------------------
                    # RESULT
                    # -----------------------------------------

                    if (
                        result["band_type"]
                        ==
                        "DIRECT BAND GAP"
                    ):

                        result_class = (
                            "direct-result"
                        )

                    else:

                        result_class = (
                            "indirect-result"
                        )

                    st.markdown(
                        f'<div class="result-card {result_class}">',
                        unsafe_allow_html=True
                    )

                    st.subheader(
                        "📊 Calculation Result"
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.metric(
                            "VBM",
                            f"{result['E_VBM']:.6f} eV"
                        )

                        st.write(
                            f"kᵥᵦₘ = "
                            f"{result['k_VBM']:.6f} Å⁻¹"
                        )

                        st.caption(
                            f"Band: {result['VBM_band']}"
                        )

                    with c2:

                        st.metric(
                            "CBM",
                            f"{result['E_CBM']:.6f} eV"
                        )

                        st.write(
                            f"k꜀ᵦₘ = "
                            f"{result['k_CBM']:.6f} Å⁻¹"
                        )

                        st.caption(
                            f"Band: {result['CBM_band']}"
                        )

                    with c3:

                        st.metric(
                            "Band Gap Eg",
                            f"{result['Eg']:.6f} eV"
                        )

                        st.write(
                            result["band_type"]
                        )

                    st.markdown(
                        f"""
### 🔬 Scientific Calculation

**Valence Band Maximum**

\[
E_{{VBM}}
=
\max(E_{{occupied}})
=
{result['E_VBM']:.6f}\ eV
\]

**Conduction Band Minimum**

\[
E_{{CBM}}
=
\min(E_{{unoccupied}})
=
{result['E_CBM']:.6f}\ eV
\]

**Band Gap**

\[
E_g
=
E_{{CBM}}
-
E_{{VBM}}
\]

\[
E_g
=
{result['E_CBM']:.6f}
-
({result['E_VBM']:.6f})
\]

\[
\boxed{{E_g = {result['Eg']:.6f}\ eV}}
\]

**Wave-vector positions**

\[
k_{{VBM}}
=
{result['k_VBM']:.6f}\ Å^{{-1}}
\]

\[
k_{{CBM}}
=
{result['k_CBM']:.6f}\ Å^{{-1}}
\]

**Wave-vector separation**

\[
\Delta k
=
|k_{{CBM}}-k_{{VBM}}|
\]

\[
\Delta k
=
{result['delta_k']:.6f}\ Å^{{-1}}
\]

### Final Classification

\[
\boxed{{\text{{{result['band_type']}}}}}
\]
"""
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

                    # -----------------------------------------
                    # DOS
                    # -----------------------------------------

                    dos_df = None

                    if dos_file is not None:

                        dos_df = pd.read_csv(
                            dos_file
                        )

                        if len(dos_df.columns) >= 2:

                            dos_df = dos_df.iloc[:, :2]

                            dos_df.columns = [
                                "energy",
                                "dos"
                            ]

                            dos_df["energy"] = pd.to_numeric(
                                dos_df["energy"],
                                errors="coerce"
                            )

                            dos_df["dos"] = pd.to_numeric(
                                dos_df["dos"],
                                errors="coerce"
                            )

                            dos_df = dos_df.dropna()

                            st.subheader(
                                "📉 Density of States"
                            )

                            dos_fig = create_dos_graph(
                                dos_df,
                                material,
                                fermi_level
                            )

                            st.plotly_chart(
                                dos_fig,
                                use_container_width=True
                            )

                    # -----------------------------------------
                    # SAVE RESULT
                    # -----------------------------------------

                    st.session_state[
                        "dft_result"
                    ] = result

                    st.session_state[
                        "dft_material"
                    ] = material

                    st.session_state[
                        "dft_symbol"
                    ] = symbol

                    st.session_state[
                        "dft_source"
                    ] = data_source

                    st.session_state[
                        "dft_fermi"
                    ] = fermi_level

                except Exception as error:

                    st.error(
                        f"Analysis Error: {error}"
                    )

                    st.info(
                        "Check your CSV format, Fermi level, "
                        "and band energies."
                    )

        except Exception as error:

            st.error(
                f"CSV Error: {error}"
            )


# =========================================================
# REPORT SECTION
# =========================================================

if "dft_result" in st.session_state:

    st.markdown("---")

    st.subheader(
        "📄 Download Reports"
    )

    result = st.session_state[
        "dft_result"
    ]

    material = st.session_state[
        "dft_material"
    ]

    symbol = st.session_state[
        "dft_symbol"
    ]

    source = st.session_state[
        "dft_source"
    ]

    fermi = st.session_state[
        "dft_fermi"
    ]

    # CSV
    report_data = pd.DataFrame([
        {
            "Material": material,
            "Symbol": symbol,
            "Data Source": source,
            "Fermi Level (eV)": fermi,

            "VBM Energy (eV)":
                result["E_VBM"],

            "VBM k (Å^-1)":
                result["k_VBM"],

            "VBM Band":
                result["VBM_band"],

            "CBM Energy (eV)":
                result["E_CBM"],

            "CBM k (Å^-1)":
                result["k_CBM"],

            "CBM Band":
                result["CBM_band"],

            "Band Gap Eg (eV)":
                result["Eg"],

            "Delta k (Å^-1)":
                result["delta_k"],

            "Band Gap Type":
                result["band_type"]
        }
    ])

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            "📊 Download CSV Report",
            report_data.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=
            f"{symbol}_band_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:

        try:

            pdf = create_pdf_report(
                material,
                symbol,
                source,
                fermi,
                result
            )

            st.download_button(
                "📄 Download PDF Report",
                pdf,
                file_name=
                f"{symbol}_E-k_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        except Exception as error:

            st.error(
                f"PDF Error: {error}"
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
<div style="text-align:center;color:#64748b;">
⚛️ E–k Diagram Analyzer |
Theoretical + DFT/Experimental Analysis |
No AI/API Dependency
</div>
""",
    unsafe_allow_html=True
)