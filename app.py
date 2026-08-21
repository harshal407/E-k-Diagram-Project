import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONSTANTS
# ============================================================

hbar = 1.054571817e-34       # J.s
m0 = 9.1093837139e-31        # kg
eV_to_J = 1.602176634e-19    # J/eV
angstrom = 1e-10             # m

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="E–k Diagram Analyzer",
    page_icon="⚛️",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .result-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        background-color: #f8f9fa;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">⚛️ E–k Diagram Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Electron Energy vs Wave Vector Analysis Tool</div>',
    unsafe_allow_html=True
)

# ============================================================
# TWO MAIN COLUMNS
# ============================================================

left, right = st.columns([1.65, 1])

# ============================================================
# LEFT SIDE — E-k INPUTS
# ============================================================

with left:

    st.markdown(
        '<div class="section-title">🔬 E–k Diagram Input Parameters</div>',
        unsafe_allow_html=True
    )

    st.subheader("1. Band Gap Energy")

    eg_col1, eg_col2 = st.columns([2, 1])

    with eg_col1:
        eg_value = st.number_input(
            "Band Gap Energy (Eg)",
            min_value=0.0,
            value=1.12,
            step=0.01,
            format="%.6f"
        )

    with eg_col2:
        eg_unit = st.selectbox(
            "Unit",
            ["eV", "J"],
            key="eg_unit"
        )

    st.subheader("2. Electron Effective Mass")

    me_col1, me_col2 = st.columns([2, 1])

    with me_col1:
        me_value = st.number_input(
            "Electron Effective Mass (mₑ*)",
            min_value=0.0,
            value=0.26,
            step=0.01,
            format="%.8e"
        )

    with me_col2:
        me_unit = st.selectbox(
            "Unit",
            ["m₀", "kg", "g", "mg"],
            key="me_unit"
        )

    st.subheader("3. Hole Effective Mass")

    mh_col1, mh_col2 = st.columns([2, 1])

    with mh_col1:
        mh_value = st.number_input(
            "Hole Effective Mass (mₕ*)",
            min_value=0.0,
            value=0.39,
            step=0.01,
            format="%.8e"
        )

    with mh_col2:
        mh_unit = st.selectbox(
            "Unit",
            ["m₀", "kg", "g", "mg"],
            key="mh_unit"
        )

    st.subheader("4. Wave Vector (k) Range")

    k_col1, k_col2 = st.columns(2)

    with k_col1:
        k_min = st.number_input(
            "Minimum k",
            value=-1.0,
            step=0.1,
            format="%.6f"
        )

    with k_col2:
        k_max = st.number_input(
            "Maximum k",
            value=1.0,
            step=0.1,
            format="%.6f"
        )

    k_unit = st.selectbox(
        "k Unit",
        ["Å⁻¹", "m⁻¹"],
        key="k_unit"
    )

    st.subheader("5. Number of Points")

    points = st.number_input(
        "Number of Points (N)",
        min_value=50,
        max_value=10000,
        value=500,
        step=50
    )

    generate = st.button(
        "🚀 Generate E–k Diagram",
        type="primary",
        use_container_width=True
    )

# ============================================================
# UNIT CONVERSION FUNCTIONS
# ============================================================

def energy_to_joule(value, unit):

    if unit == "eV":
        return value * eV_to_J

    return value


def mass_to_kg(value, unit):

    if unit == "m₀":
        return value * m0

    elif unit == "kg":
        return value

    elif unit == "g":
        return value * 1e-3

    elif unit == "mg":
        return value * 1e-6


def k_to_inverse_meter(value, unit):

    if unit == "Å⁻¹":
        return value * 1e10

    return value


def joule_to_ev(value):

    return value / eV_to_J


# ============================================================
# RIGHT SIDE — UNIT CONVERTER
# ============================================================

with right:

    st.markdown(
        '<div class="section-title">🔄 Unit Converter</div>',
        unsafe_allow_html=True
    )

    converter_type = st.selectbox(
        "Select Conversion Type",
        [
            "Energy: eV ↔ J",
            "Mass: kg ↔ g ↔ mg",
            "Mass: m₀ ↔ kg",
            "Wave Vector: Å⁻¹ ↔ m⁻¹"
        ]
    )

    converter_value = st.number_input(
        "Enter Value",
        value=1.0,
        format="%.10e"
    )

    # --------------------------------------------------------
    # ENERGY CONVERTER
    # --------------------------------------------------------

    if converter_type == "Energy: eV ↔ J":

        c1, c2 = st.columns(2)

        with c1:
            energy_from = st.selectbox(
                "From",
                ["eV", "J"],
                key="energy_from"
            )

        with c2:
            energy_to = st.selectbox(
                "To",
                ["J", "eV"],
                key="energy_to"
            )

        if st.button("Convert Energy", use_container_width=True):

            if energy_from == "eV" and energy_to == "J":
                result = converter_value * eV_to_J

            elif energy_from == "J" and energy_to == "eV":
                result = converter_value / eV_to_J

            else:
                result = converter_value

            st.success(f"{converter_value:.6e} {energy_from}")
            st.info(f"= {result:.6e} {energy_to}")

    # --------------------------------------------------------
    # MASS CONVERTER
    # --------------------------------------------------------

    elif converter_type == "Mass: kg ↔ g ↔ mg":

        c1, c2 = st.columns(2)

        with c1:
            mass_from = st.selectbox(
                "From",
                ["kg", "g", "mg"],
                key="mass_from"
            )

        with c2:
            mass_to = st.selectbox(
                "To",
                ["kg", "g", "mg"],
                key="mass_to"
            )

        if st.button("Convert Mass", use_container_width=True):

            mass_kg = mass_to_kg(converter_value, mass_from)

            if mass_to == "kg":
                result = mass_kg

            elif mass_to == "g":
                result = mass_kg * 1000

            else:
                result = mass_kg * 1e6

            st.info(f"{result:.6e} {mass_to}")

    # --------------------------------------------------------
    # m0 CONVERTER
    # --------------------------------------------------------

    elif converter_type == "Mass: m₀ ↔ kg":

        c1, c2 = st.columns(2)

        with c1:
            m0_from = st.selectbox(
                "From",
                ["m₀", "kg"],
                key="m0_from"
            )

        with c2:
            m0_to = st.selectbox(
                "To",
                ["kg", "m₀"],
                key="m0_to"
            )

        if st.button("Convert m₀", use_container_width=True):

            if m0_from == "m₀" and m0_to == "kg":
                result = converter_value * m0

            elif m0_from == "kg" and m0_to == "m₀":
                result = converter_value / m0

            else:
                result = converter_value

            st.info(f"{result:.6e} {m0_to}")

    # --------------------------------------------------------
    # WAVE VECTOR CONVERTER
    # --------------------------------------------------------

    else:

        c1, c2 = st.columns(2)

        with c1:
            k_from = st.selectbox(
                "From",
                ["Å⁻¹", "m⁻¹"],
                key="k_from"
            )

        with c2:
            k_to = st.selectbox(
                "To",
                ["m⁻¹", "Å⁻¹"],
                key="k_to"
            )

        if st.button("Convert k", use_container_width=True):

            k_m = k_to_inverse_meter(converter_value, k_from)

            if k_to == "m⁻¹":
                result = k_m

            else:
                result = k_m / 1e10

            st.info(f"{result:.6e} {k_to}")


# ============================================================
# E-k CALCULATION
# ============================================================

if generate:

    # Validation
    if k_min >= k_max:
        st.error("❌ Minimum k must be smaller than Maximum k.")

    elif me_value <= 0:
        st.error("❌ Electron effective mass must be greater than zero.")

    elif mh_value <= 0:
        st.error("❌ Hole effective mass must be greater than zero.")

    elif eg_value <= 0:
        st.error("❌ Band gap must be greater than zero.")

    else:

        # ----------------------------------------------------
        # Convert inputs to SI units
        # ----------------------------------------------------

        Eg_J = energy_to_joule(eg_value, eg_unit)

        me_kg = mass_to_kg(me_value, me_unit)

        mh_kg = mass_to_kg(mh_value, mh_unit)

        k_min_si = k_to_inverse_meter(k_min, k_unit)

        k_max_si = k_to_inverse_meter(k_max, k_unit)

        # ----------------------------------------------------
        # Create k values
        # ----------------------------------------------------

        k = np.linspace(
            k_min_si,
            k_max_si,
            int(points)
        )

        # ----------------------------------------------------
        # E-k Equations
        # ----------------------------------------------------

        Ec_J = Eg_J + (hbar**2 * k**2) / (2 * me_kg)

        Ev_J = -(hbar**2 * k**2) / (2 * mh_kg)

        # Convert energy to eV
        Ec_eV = Ec_J / eV_to_J
        Ev_eV = Ev_J / eV_to_J

        # k for graph in selected unit
        if k_unit == "Å⁻¹":
            k_graph = k / 1e10
        else:
            k_graph = k

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        st.markdown("---")

        st.header("📊 E–k Diagram")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            k_graph,
            Ec_eV,
            linewidth=2.5,
            label="Conduction Band"
        )

        ax.plot(
            k_graph,
            Ev_eV,
            linewidth=2.5,
            label="Valence Band"
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1
        )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1
        )

        ax.set_xlabel(
            f"Wave Vector k ({k_unit})",
            fontsize=12
        )

        ax.set_ylabel(
            "Energy E (eV)",
            fontsize=12
        )

        ax.set_title(
            "Electron Energy vs Wave Vector",
            fontsize=15
        )

        ax.grid(True, alpha=0.3)

        ax.legend()

        st.pyplot(fig)

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        st.header("📋 Calculation Results")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Band Gap",
                f"{Eg_J / eV_to_J:.4f} eV"
            )

        with r2:
            st.metric(
                "Electron Mass",
                f"{me_kg:.4e} kg"
            )

        with r3:
            st.metric(
                "Hole Mass",
                f"{mh_kg:.4e} kg"
            )

        # ----------------------------------------------------
        # ANALYSIS
        # ----------------------------------------------------

        st.header("🔎 Basic Analysis")

        electron_curvature = 1 / me_kg
        hole_curvature = 1 / mh_kg

        st.write(
            f"**Band Gap:** {Eg_J / eV_to_J:.4f} eV"
        )

        st.write(
            f"**Electron effective mass:** {me_kg:.4e} kg"
        )

        st.write(
            f"**Hole effective mass:** {mh_kg:.4e} kg"
        )

        st.write(
            f"**Number of points used:** {int(points)}"
        )

        st.write(
            f"**k range:** {k_min} to {k_max} {k_unit}"
        )

        if me_kg < mh_kg:
            st.write(
                "📌 The electron effective mass is smaller than "
                "the hole effective mass, so the conduction band "
                "has greater curvature."
            )

        elif me_kg > mh_kg:
            st.write(
                "📌 The hole effective mass is smaller than "
                "the electron effective mass, so the valence band "
                "has greater curvature."
            )

        else:
            st.write(
                "📌 Electron and hole effective masses are equal."
            )

        st.success(
            "✅ E–k diagram generated successfully."
        )