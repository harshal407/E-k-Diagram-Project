import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="E-k Diagram Analyzer",
    page_icon="⚛️",
    layout="wide"
)

# ============================================================
# CONSTANTS
# ============================================================

HBAR = 1.054571817e-34
M0 = 9.1093837139e-31
EV_TO_J = 1.602176634e-19


# ============================================================
# MATERIAL DATABASE
# ============================================================
# k0 is the approximate CBM valley position in Å^-1.
# k_vbm is taken at Gamma = 0 for this model.

MATERIAL_DATABASE = {

    "Silicon": {
        "symbol": "Si",
        "type": "indirect",
        "k0": 0.85
    },

    "Germanium": {
        "symbol": "Ge",
        "type": "indirect",
        "k0": 0.50
    },

    "Gallium Arsenide": {
        "symbol": "GaAs",
        "type": "direct",
        "k0": 0.0
    },

    "Indium Phosphide": {
        "symbol": "InP",
        "type": "direct",
        "k0": 0.0
    },

    "Cadmium Telluride": {
        "symbol": "CdTe",
        "type": "direct",
        "k0": 0.0
    },

    "Gallium Nitride": {
        "symbol": "GaN",
        "type": "direct",
        "k0": 0.0
    },

    "Zinc Oxide": {
        "symbol": "ZnO",
        "type": "direct",
        "k0": 0.0
    }
}


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 15% 15%,
            rgba(0, 180, 255, 0.18),
            transparent 28%
        ),
        radial-gradient(
            circle at 85% 20%,
            rgba(120, 70, 255, 0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 50% 90%,
            rgba(0, 220, 180, 0.10),
            transparent 32%
        ),
        linear-gradient(
            135deg,
            #06101d,
            #0a1830,
            #07111f
        );

    color: white;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    text-align: center;
    padding: 20px;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 43px;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero p {
    color: #a9bfd9;
    font-size: 17px;
}

.physics-strip {
    text-align: center;
    padding: 18px;
    margin-bottom: 25px;
    border-radius: 22px;

    background:
        linear-gradient(
            100deg,
            rgba(0,150,255,0.12),
            rgba(120,70,255,0.12)
        );

    border: 1px solid rgba(130,210,255,0.18);
    font-size: 27px;
    color: #72dcff;
    letter-spacing: 8px;
}

.card {
    background: rgba(12,27,48,0.76);
    border: 1px solid rgba(130,200,255,0.18);
    border-radius: 22px;
    padding: 24px;
    margin-bottom: 22px;

    box-shadow:
        0 12px 35px rgba(0,0,0,0.28),
        inset 0 1px 0 rgba(255,255,255,0.04);

    backdrop-filter: blur(12px);
}

.section-title {
    font-size: 22px;
    font-weight: 750;
    margin-bottom: 18px;
}

.result-card {
    background: rgba(8,23,42,0.85);
    border: 1px solid rgba(100,190,255,0.20);
    border-radius: 18px;
    padding: 20px;
    min-height: 145px;
}

.result-symbol {
    color: #6fdcff;
    font-size: 20px;
    font-weight: 700;
}

.result-label {
    color: #91a8c2;
    font-size: 14px;
    margin-top: 7px;
}

.result-value {
    color: white;
    font-size: 25px;
    font-weight: 750;
    margin-top: 8px;
}

.bandgap-card {
    text-align: center;
    padding: 28px;
    margin-top: 20px;

    border-radius: 22px;

    background:
        linear-gradient(
            135deg,
            rgba(0,160,255,0.18),
            rgba(120,70,255,0.18)
        );

    border: 1px solid rgba(110,210,255,0.28);
}

.bandgap-value {
    font-size: 34px;
    font-weight: 800;
    margin: 8px;
}

.direct {
    color: #53e6a4;
    font-size: 22px;
    font-weight: 800;
}

.indirect {
    color: #ffc966;
    font-size: 22px;
    font-weight: 800;
}

.stButton > button {
    width: 100%;
    min-height: 50px;
    border-radius: 14px;
    font-size: 17px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONVERSION FUNCTIONS
# ============================================================

def energy_to_joule(value, unit):

    if unit == "eV":
        return value * EV_TO_J

    return value


def joule_to_ev(value):

    return value / EV_TO_J


def mass_to_kg(value, unit):

    if unit == "kg":
        return value

    if unit == "g":
        return value * 1e-3

    if unit == "mg":
        return value * 1e-6

    if unit == "m₀":
        return value * M0

    return value


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>⚛️ E–k Diagram Analyzer</h1>

<p>
Interactive Semiconductor Band Structure & Band Gap Analysis
</p>

</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="physics-strip">
E(k) &nbsp; ψ &nbsp; ℏ &nbsp; k &nbsp; m* &nbsp; Eg &nbsp; VBM &nbsp; CBM
</div>
""", unsafe_allow_html=True)


# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns([3.5, 1])


# ============================================================
# MATERIAL DETAILS
# ============================================================

with left:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">🔬 Material Details</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        material_name = st.text_input(
            "Material Name",
            placeholder="Example: Silicon"
        )

        material_symbol = st.text_input(
            "Material Symbol",
            placeholder="Example: Si"
        )

    with c2:

        eg_c1, eg_c2 = st.columns([2.3, 1])

        with eg_c1:

            eg_value = st.number_input(
                "Band Gap Energy  E₉",
                min_value=0.000001,
                value=1.12,
                step=0.01
            )

        with eg_c2:

            eg_unit = st.selectbox(
                "Unit",
                ["eV", "J"]
            )

    st.markdown("### Electron Effective Mass  mₑ*")

    me_c1, me_c2 = st.columns([2.3, 1])

    with me_c1:

        me_value = st.number_input(
            "Electron Effective Mass",
            min_value=0.000001,
            value=0.26,
            step=0.01
        )

    with me_c2:

        me_unit = st.selectbox(
            "Electron Mass Unit",
            ["m₀", "kg", "g", "mg"]
        )

    st.markdown("### Hole Effective Mass  mₕ*")

    mh_c1, mh_c2 = st.columns([2.3, 1])

    with mh_c1:

        mh_value = st.number_input(
            "Hole Effective Mass",
            min_value=0.000001,
            value=0.39,
            step=0.01
        )

    with mh_c2:

        mh_unit = st.selectbox(
            "Hole Mass Unit",
            ["m₀", "kg", "g", "mg"]
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# K RANGE
# ============================================================

with left:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">〰️ Wave Vector Range</div>',
        unsafe_allow_html=True
    )

    k1, k2, k3 = st.columns(3)

    with k1:

        k_min = st.number_input(
            "k Minimum",
            value=-1.2
        )

    with k2:

        k_max = st.number_input(
            "k Maximum",
            value=1.2
        )

    with k3:

        k_unit = st.selectbox(
            "k Unit",
            ["Å⁻¹", "m⁻¹"]
        )

    points = st.slider(
        "Number of Points / Graph Smoothness",
        min_value=200,
        max_value=5000,
        value=1200,
        step=100
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# RIGHT UNIT CONVERTER
# ============================================================

with right:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">🔄 Unit Converter</div>',
        unsafe_allow_html=True
    )

    # ENERGY

    st.markdown("### ⚡ Energy")

    energy_value = st.number_input(
        "Energy",
        value=1.0,
        key="energy_converter"
    )

    energy_conversion = st.selectbox(
        "Conversion",
        ["eV → J", "J → eV"],
        key="energy_conversion"
    )

    if st.button(
        "Convert Energy",
        key="energy_button"
    ):

        if energy_conversion == "eV → J":

            result = energy_value * EV_TO_J

            st.success(
                f"{result:.6e} J"
            )

        else:

            result = energy_value / EV_TO_J

            st.success(
                f"{result:.6f} eV"
            )

    st.divider()

    # MASS

    st.markdown("### ⚖️ Mass")

    mass_value = st.number_input(
        "Mass",
        value=1.0,
        key="mass_converter"
    )

    mass_conversion = st.selectbox(
        "Conversion",
        [
            "kg → g",
            "g → kg",
            "mg → kg",
            "m₀ → kg"
        ],
        key="mass_conversion"
    )

    if st.button(
        "Convert Mass",
        key="mass_button"
    ):

        if mass_conversion == "kg → g":

            st.success(
                f"{mass_value * 1000:.6g} g"
            )

        elif mass_conversion == "g → kg":

            st.success(
                f"{mass_value / 1000:.6e} kg"
            )

        elif mass_conversion == "mg → kg":

            st.success(
                f"{mass_value * 1e-6:.6e} kg"
            )

        else:

            st.success(
                f"{mass_value * M0:.6e} kg"
            )

    st.divider()

    # K

    st.markdown("### 〰️ Wave Vector")

    k_value = st.number_input(
        "k",
        value=1.0,
        key="k_converter"
    )

    k_conversion = st.selectbox(
        "Conversion",
        [
            "Å⁻¹ → m⁻¹",
            "m⁻¹ → Å⁻¹"
        ],
        key="k_conversion"
    )

    if st.button(
        "Convert k",
        key="k_button"
    ):

        if k_conversion == "Å⁻¹ → m⁻¹":

            st.success(
                f"{k_value * 1e10:.6e} m⁻¹"
            )

        else:

            st.success(
                f"{k_value / 1e10:.6e} Å⁻¹"
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# GENERATE BUTTON
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

generate = st.button(
    "🚀 GENERATE E–k DIAGRAM",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if generate:

    if material_name.strip() == "":
        st.error("Please enter Material Name.")

    elif material_symbol.strip() == "":
        st.error("Please enter Material Symbol.")

    elif k_min >= k_max:
        st.error("k Minimum must be smaller than k Maximum.")

    else:

        # ----------------------------------------------------
        # FIND MATERIAL INFORMATION
        # ----------------------------------------------------

        material_info = MATERIAL_DATABASE.get(
            material_name.strip()
        )

        if material_info is None:

            st.warning(
                "This material is not available in the built-in "
                "band-valley database. The program will generate "
                "a Gamma-centered theoretical band structure, "
                "so Direct/Indirect classification for an unknown "
                "material should not be treated as experimental data."
            )

            k0 = 0.0

        else:

            k0 = material_info["k0"]

        # ----------------------------------------------------
        # CONVERT PARAMETERS
        # ----------------------------------------------------

        Eg_J = energy_to_joule(
            eg_value,
            eg_unit
        )

        me_kg = mass_to_kg(
            me_value,
            me_unit
        )

        mh_kg = mass_to_kg(
            mh_value,
            mh_unit
        )

        # ----------------------------------------------------
        # K GRID
        # ----------------------------------------------------

        k = np.linspace(
            k_min,
            k_max,
            points
        )

        # If m^-1 was selected, convert the displayed
        # range to SI for calculation.

        if k_unit == "Å⁻¹":

            k_si = k * 1e10
            k0_si = k0 * 1e10

        else:

            k_si = k
            k0_si = k0 * 1e10

        # ----------------------------------------------------
        # VALENCE BAND
        # ----------------------------------------------------
        #
        # VBM is centered around Gamma (k = 0).
        #
        # E_v(k) = -hbar² k² / 2mh*
        #

        Ev_J = -(
            HBAR**2 *
            k_si**2
            /
            (2 * mh_kg)
        )

        # ----------------------------------------------------
        # CONDUCTION BAND
        # ----------------------------------------------------
        #
        # For direct materials k0 = 0.
        #
        # For indirect materials the conduction minimum
        # occurs at k = ±k0.
        #

        Ec_J = Eg_J + (
            HBAR**2 *
            (np.abs(k_si) - abs(k0_si))**2
            /
            (2 * me_kg)
        )

        # ----------------------------------------------------
        # CONVERT TO eV
        # ----------------------------------------------------

        Ev = Ev_J / EV_TO_J
        Ec = Ec_J / EV_TO_J

        # ----------------------------------------------------
        # AUTOMATIC VBM
        # ----------------------------------------------------

        vbm_index = np.argmax(Ev)

        k_vbm = k[vbm_index]

        E_vbm = Ev[vbm_index]

        # ----------------------------------------------------
        # AUTOMATIC CBM
        # ----------------------------------------------------

        cbm_index = np.argmin(Ec)

        k_cbm = k[cbm_index]

        E_cbm = Ec[cbm_index]

        # ----------------------------------------------------
        # CALCULATED BAND GAP
        # ----------------------------------------------------

        calculated_Eg = E_cbm - E_vbm

        # ----------------------------------------------------
        # k SEPARATION
        # ----------------------------------------------------

        delta_k = abs(
            k_cbm - k_vbm
        )

        # ----------------------------------------------------
        # AUTOMATIC DIRECT / INDIRECT
        # ----------------------------------------------------

        k_step = abs(
            k[1] - k[0]
        )

        if delta_k <= k_step * 1.5:

            band_type = "DIRECT BAND GAP"

        else:

            band_type = "INDIRECT BAND GAP"

        # ====================================================
        # GRAPH
        # ====================================================

        fig = go.Figure()

        # ----------------------------------------------------
        # CONDUCTION BAND
        # ----------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=k,
                y=Ec,
                mode="lines",
                name="Conduction Band",
                line=dict(width=3),
                customdata=np.column_stack(
                    (
                        k,
                        Ec,
                        np.full(
                            len(k),
                            "Conduction Band"
                        )
                    )
                ),
                hovertemplate=
                "<b>%{customdata[2]}</b><br>"
                "k = %{customdata[0]:.6f}<br>"
                "E = %{customdata[1]:.6f} eV"
                "<extra></extra>"
            )
        )

        # ----------------------------------------------------
        # VALENCE BAND
        # ----------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=k,
                y=Ev,
                mode="lines",
                name="Valence Band",
                line=dict(width=3),
                customdata=np.column_stack(
                    (
                        k,
                        Ev,
                        np.full(
                            len(k),
                            "Valence Band"
                        )
                    )
                ),
                hovertemplate=
                "<b>%{customdata[2]}</b><br>"
                "k = %{customdata[0]:.6f}<br>"
                "E = %{customdata[1]:.6f} eV"
                "<extra></extra>"
            )
        )

        # ----------------------------------------------------
        # VBM MARKER
        # ----------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=[k_vbm],
                y=[E_vbm],
                mode="markers+text",
                name="VBM",
                text=["VBM"],
                textposition="top center",
                marker=dict(
                    size=15,
                    symbol="diamond"
                ),
                hovertemplate=
                f"<b>VBM</b><br>"
                f"k<sub>VBM</sub> = "
                f"{k_vbm:.6f} {k_unit}<br>"
                f"E<sub>VBM</sub> = "
                f"{E_vbm:.6f} eV"
                "<extra></extra>"
            )
        )

        # ----------------------------------------------------
        # CBM MARKER
        # ----------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=[k_cbm],
                y=[E_cbm],
                mode="markers+text",
                name="CBM",
                text=["CBM"],
                textposition="top center",
                marker=dict(
                    size=15,
                    symbol="diamond"
                ),
                hovertemplate=
                f"<b>CBM</b><br>"
                f"k<sub>CBM</sub> = "
                f"{k_cbm:.6f} {k_unit}<br>"
                f"E<sub>CBM</sub> = "
                f"{E_cbm:.6f} eV"
                "<extra></extra>"
            )
        )

        # ====================================================
        # BAND GAP ARROW
        # ====================================================

        fig.add_annotation(
            x=k_cbm,
            y=E_cbm,
            ax=k_cbm,
            ay=E_vbm,
            text=f"<b>Eg = {calculated_Eg:.4f} eV</b>",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.3,
            arrowwidth=2
        )

        # ====================================================
        # INDIRECT HORIZONTAL SEPARATION
        # ====================================================

        if band_type == "INDIRECT BAND GAP":

            mid_energy = (
                E_vbm +
                0.10 * calculated_Eg
            )

            fig.add_shape(
                type="line",
                x0=k_vbm,
                y0=mid_energy,
                x1=k_cbm,
                y1=mid_energy,
                line=dict(
                    width=3,
                    dash="dash"
                )
            )

            fig.add_annotation(
                x=(k_vbm + k_cbm) / 2,
                y=mid_energy,
                text=f"<b>Δk = {delta_k:.4f} {k_unit}</b>",
                showarrow=False,
                yshift=14
            )

            # Vertical guide from VBM

            fig.add_shape(
                type="line",
                x0=k_vbm,
                y0=E_vbm,
                x1=k_vbm,
                y1=mid_energy,
                line=dict(
                    width=1,
                    dash="dot"
                )
            )

            # Vertical guide from CBM

            fig.add_shape(
                type="line",
                x0=k_cbm,
                y0=E_cbm,
                x1=k_cbm,
                y1=mid_energy,
                line=dict(
                    width=1,
                    dash="dot"
                )
            )

        # ====================================================
        # DIRECT VERTICAL GUIDE
        # ====================================================

        else:

            fig.add_shape(
                type="line",
                x0=k_vbm,
                y0=E_vbm,
                x1=k_vbm,
                y1=E_cbm,
                line=dict(
                    width=1,
                    dash="dot"
                )
            )

        # ====================================================
        # GRAPH TITLE
        # ====================================================

        fig.update_layout(

            title=dict(
                text=(
                    f"<b>{material_name} "
                    f"({material_symbol}) — "
                    f"{band_type}</b>"
                ),
                font=dict(size=24)
            ),

            xaxis_title=(
                f"Wave Vector, k ({k_unit})"
            ),

            yaxis_title="Energy, E (eV)",

            height=700,

            template="plotly_dark",

            hovermode="closest",

            margin=dict(
                l=70,
                r=60,
                t=100,
                b=70
            ),

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            )
        )

        # ====================================================
        # DISPLAY GRAPH
        # ====================================================

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">'
            '📊 Interactive E–k Diagram'
            '</div>',
            unsafe_allow_html=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown(
            """
            <div style="
                text-align:center;
                color:#91a8c2;
                font-size:14px;
                padding:8px;
            ">
            🖱️ Move the cursor over the bands to read
            exact <b>k</b>, <b>Energy</b> and <b>Band</b>.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # ====================================================
        # CALCULATION RESULTS
        # ====================================================

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">'
            '🧮 Calculation Results'
            '</div>',
            unsafe_allow_html=True
        )

        r1, r2 = st.columns(2)

        # VBM

        with r1:

            st.markdown(
                f"""
                <div class="result-card">

                <div class="result-symbol">
                E<sub>VBM</sub>
                </div>

                <div class="result-label">
                Valence Band Maximum
                </div>

                <div class="result-value">
                {E_vbm:.6f} eV
                </div>

                <div style="
                color:#9db4cf;
                margin-top:10px;
                ">
                k<sub>VBM</sub> =
                {k_vbm:.6f} {k_unit}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        # CBM

        with r2:

            st.markdown(
                f"""
                <div class="result-card">

                <div class="result-symbol">
                E<sub>CBM</sub>
                </div>

                <div class="result-label">
                Conduction Band Minimum
                </div>

                <div class="result-value">
                {E_cbm:.6f} eV
                </div>

                <div style="
                color:#9db4cf;
                margin-top:10px;
                ">
                k<sub>CBM</sub> =
                {k_cbm:.6f} {k_unit}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        # BAND GAP

        st.markdown(
            f"""
            <div class="bandgap-card">

            <div style="
            color:#a6bdd5;
            font-size:15px;
            ">
            Calculated Band Gap
            </div>

            <div class="bandgap-value">
            E<sub>g</sub> =
            {calculated_Eg:.6f} eV
            </div>

            <div style="
            color:#91a8c2;
            font-size:14px;
            ">
            E<sub>g</sub> =
            E<sub>CBM</sub> − E<sub>VBM</sub>
            </div>

            <div style="margin-top:18px;">

            {
                '<span class="direct">🟢 DIRECT BAND GAP</span>'
                if band_type == "DIRECT BAND GAP"
                else
                '<span class="indirect">🟠 INDIRECT BAND GAP</span>'
            }

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # ====================================================
        # ANALYSIS INFORMATION
        # ====================================================

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">'
            '🔎 Graph Analysis'
            '</div>',
            unsafe_allow_html=True
        )

        if band_type == "DIRECT BAND GAP":

            st.success(
                f"""
                DIRECT BAND GAP detected.

                k_VBM = {k_vbm:.6f} {k_unit}

                k_CBM = {k_cbm:.6f} {k_unit}

                The VBM and CBM occur at approximately
                the same wave-vector position.
                """
            )

        else:

            st.warning(
                f"""
                INDIRECT BAND GAP detected.

                k_VBM = {k_vbm:.6f} {k_unit}

                k_CBM = {k_cbm:.6f} {k_unit}

                Δk = {delta_k:.6f} {k_unit}

                The VBM and CBM occur at different
                wave-vector positions.
                """
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ====================================================
        # FOOTER
        # ====================================================

        st.markdown(
            """
            <div style="
                text-align:center;
                color:#7088a5;
                padding:20px;
                font-size:13px;
            ">
            ⚛️ E–k Diagram Analyzer
            |
            Semiconductor Band Structure Analysis
            </div>
            """,
            unsafe_allow_html=True
        )