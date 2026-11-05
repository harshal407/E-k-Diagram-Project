"""Semiconductor E–k Diagram & DFT Analysis Platform (Streamlit)

Inputs : 1) Database / custom material (manual parameters)  2) CIF upload  3) DFT output upload
Modes  : Theoretical (model bands)  |  Real DFT (Quantum ESPRESSO)
Run    : streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    HAS_PMG = True
except Exception:
    HAS_PMG = False
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

HB2_2M = 3.80998  # hbar^2 / 2 m0 in eV*Angstrom^2

# ---------------------------------------------------------------- database
# Reference values live in the code only. They are never shown in the input fields and are
# used only to check the values the user types (error line when a value is far off).
# Approximate room-temperature literature values (verify before publishing).
# dG/dL/dX = conduction valley energy above the CBM (eV). m in units of m0.
DB = pd.DataFrame([
    # name, formula, structure, a, c, Eg, me, mhh, mlh, so, dG, dL, dX, class
    ("Silicon", "Si", "Diamond", 5.431, 5.431, 1.12, 0.26, 0.49, 0.16, 0.044, 2.30, 1.10, 0.00, "Elemental"),
    ("Germanium", "Ge", "Diamond", 5.658, 5.658, 0.66, 0.12, 0.28, 0.044, 0.29, 0.14, 0.00, 0.19, "Elemental"),
    ("Gallium Arsenide", "GaAs", "Zinc Blende", 5.653, 5.653, 1.42, 0.067, 0.45, 0.082, 0.34, 0.00, 0.29, 0.48, "Compound"),
    ("Zinc Oxide", "ZnO", "Wurtzite", 3.25, 5.207, 3.37, 0.24, 0.59, 0.59, 0.01, 0.00, 1.50, 2.00, "Wide gap"),
    ("Gallium Nitride", "GaN", "Wurtzite", 3.189, 5.185, 3.40, 0.20, 0.80, 0.80, 0.017, 0.00, 1.50, 2.00, "Wide gap"),
    ("Cadmium Telluride", "CdTe", "Zinc Blende", 6.482, 6.482, 1.50, 0.11, 0.72, 0.13, 0.91, 0.00, 0.50, 0.90, "Compound"),
], columns=["name", "formula", "structure", "a", "c", "Eg", "me", "mhh", "mlh",
            "so", "dG", "dL", "dX", "class"])

CUSTOM = "➕ New material (custom)"
KEYS = ["a", "Eg", "me", "mhh", "mlh", "so", "dG", "dL", "dX"]
REQUIRED = ["a", "Eg", "me", "mhh", "mlh"]          # must be > 0
LABEL = {"a": "a (Å)", "Eg": "Eg (eV)", "me": "m*e", "mhh": "m*hh", "mlh": "m*lh",
         "so": "Δso (eV)", "dG": "Γ valley (eV)", "dL": "L valley (eV)", "dX": "X valley (eV)"}
ABS_TOL = {"a": 0.02, "Eg": 0.02, "me": 0.01, "mhh": 0.01, "mlh": 0.01,
           "so": 0.02, "dG": 0.02, "dL": 0.02, "dX": 0.02}
REL_TOL = 0.10  # a typed value may differ from the reference by 10 % (+ small absolute margin)


# ---------------------------------------------------------------- validation
def validate(p):
    """Hard errors: with any of these the Generate button stays disabled."""
    errs = []
    for key in REQUIRED:
        if p[key] <= 0:
            errs.append(f"{LABEL[key]} 0 kinva negative aahe, positive value dya.")
    if p["so"] < 0:
        errs.append("Δso negative aahe, 0 kinva positive dya.")
    valleys = [p["dG"], p["dL"], p["dX"]]
    if min(valleys) < 0:
        errs.append("Valley energy negative aahe, 0 kinva positive dya.")
    elif min(valleys) > 1e-9:
        errs.append("Γ, L, X valley madhe kimaan ek value 0 pahije "
                    "(valley energy CBM peksha kiti war aahe te dya; lowest valley = 0).")
    return errs


def soft_warnings(p):
    w = []
    valleys = [p["dG"], p["dL"], p["dX"]]
    if sum(v < 1e-9 for v in valleys) > 1:
        w.append("Ek peksha jast valley 0 war aahet: CBM tie aahe, Type (direct/indirect) nakki nahi.")
    if p["mlh"] > p["mhh"] > 0:
        w.append("m*lh, m*hh peksha motha aahe: heavy/light hole ulat tar nahi na?")
    return w


def compare_with_reference(p, row):
    """Error lines when a typed value of a known material is far from the stored reference."""
    lines = []
    for key in KEYS:
        v, ref = float(p[key]), float(row[key])
        if key in REQUIRED and v <= 0:
            continue  # already reported by validate()
        if abs(v - ref) > REL_TOL * abs(ref) + ABS_TOL[key]:
            lines.append(f"{row.formula} sathi {LABEL[key]} = {v:g} takla, reference value {ref:g} aahe. "
                         "Value tapasa.")
    return lines


# ---------------------------------------------------------------- model
def band(x, centre, e0, sign, mass, a):
    """Bounded effective-mass band: curvature at `centre` equals hbar^2/2m*.
    Valleys at the zone edge (centre = +-1) use a half-frequency cosine so the band has
    its minimum only at its own edge (not also at the opposite edge)."""
    w = float(np.clip(4 * HB2_2M / (mass * a ** 2), 0.8, 8.0))
    d = np.abs(x - centre)
    if abs(centre) > 0.5:
        return e0 + sign * min(4 * w, 16.0) * (1 - np.cos(np.pi * d / 2)) / 2
    return e0 + sign * w * (1 - np.cos(np.pi * (x - centre))) / 2


def theoretical_bands(p, n=301):
    """Multi-band model along L - Gamma - X. Returns k, bands(nb,nk), nocc, ticks."""
    x = np.linspace(-1, 1, n)
    a, eg = p["a"], p["Eg"]
    so = p["so"]
    vb = [band(x, 0, 0.0, -1, p["mhh"], a),
          band(x, 0, 0.0, -1, p["mlh"], a),
          band(x, 0, -so, -1, max(0.6 * p["mlh"] + 0.2 * p["mhh"], 0.05), a)]
    cb = [band(x, 0, eg + p["dG"], +1, p["me"], a),
          band(x, -1, eg + p["dL"], +1, 1.5 * p["me"] + 0.1, a),
          band(x, +1, eg + p["dX"], +1, 1.5 * p["me"] + 0.1, a)]
    bands = np.array(vb + cb)
    return x, bands, 3, {-1: "L", 0: "Γ", 1: "X"}


# ---------------------------------------------------------------- analysis
def analyze(k, bands, nocc):
    vb, cb = bands[:nocc], bands[nocc:]
    iv = np.unravel_index(np.argmax(vb), vb.shape)
    ic = np.unravel_index(np.argmin(cb), cb.shape)
    vbm, cbm = vb[iv], cb[ic]
    kv, kc = k[iv[1]], k[ic[1]]
    span = (k.max() - k.min()) or 1.0
    direct = abs(kv - kc) < 0.01 * span
    return dict(VBM=vbm, CBM=cbm, Gap=cbm - vbm, k_VBM=kv, k_CBM=kc,
                Type="Direct" if direct else "Indirect",
                i_VBM=int(iv[1]), i_CBM=int(ic[1]))


def dos(bands, emin, emax, sigma=0.05, n=600):
    e = np.linspace(emin, emax, n)
    g = np.zeros_like(e)
    for en in bands.ravel():
        g += np.exp(-((e - en) ** 2) / (2 * sigma ** 2))
    return e, g / (sigma * np.sqrt(2 * np.pi)) / bands.shape[1]


def _fit_mass(k, e, i0, side, kfac, npts=6):
    """Parabolic fit E = E0 + c x^2 (+ linear term) on one side of an extremum.
    x in 1/Angstrom; m*/m0 = HB2_2M / c.  Returns nan if too few continuous points."""
    step = np.median(np.abs(np.diff(k))) or 1e-9
    idx = [i0]
    for j in range(1, npts + 1):
        i = i0 + side * j
        if i < 0 or i >= len(k) or abs(k[i] - k[i - side]) > 3 * step:
            break
        idx.append(i)
    if len(idx) < 4:
        return np.nan
    x = (k[idx] - k[i0]) * kfac
    c = np.polyfit(x, e[idx] - e[i0], 2)[0]
    return abs(HB2_2M / c) if abs(c) > 1e-9 else np.nan


def full_analysis(k, bands, nocc, ticks, kfac):
    """Everything below is computed from the band data only.
    kfac converts path units to 1/Angstrom (DFT: 2*pi/alat)."""
    base = analyze(k, bands, nocc)
    iV, iC = base["i_VBM"], base["i_CBM"]
    rows = []
    for name, b, i0 in [("VB1 (top)", nocc - 1, iV), ("VB2", nocc - 2, iV),
                        ("VB3", nocc - 3, iV), ("CB1 (lowest)", nocc, iC)]:
        if b < 0 or b >= bands.shape[0]:
            continue
        for side, lab in [(-1, "toward path start"), (1, "toward path end")]:
            m = _fit_mass(k, bands[b], i0, side, kfac)
            if np.isfinite(m):
                rows.append({"band": name, "k": round(float(k[i0]), 4),
                             "direction": lab, "m*/m0": round(float(m), 3)})
    masses = pd.DataFrame(rows)

    ev = np.sort(bands[:nocc, iV])[::-1]
    levels = [ev[0]]
    for e in ev[1:]:
        if levels[-1] - e > 1e-3:
            levels.append(e)
    split = levels[0] - levels[1] if len(levels) > 1 and levels[0] - levels[1] < 1.5 else np.nan

    pts = []
    for x, lab in ticks.items():
        i = int(np.argmin(np.abs(k - x)))
        vtop, cbot = bands[:nocc, i].max(), bands[nocc:, i].min()
        pts.append({"point": lab, "VB top − VBM (eV)": round(vtop - base["VBM"], 3),
                    "CB min − CBM (eV)": round(cbot - base["CBM"], 3),
                    "vertical gap (eV)": round(cbot - vtop, 3)})
    dg = bands[nocc:].min(axis=0) - bands[:nocc].max(axis=0)
    j = int(np.argmin(dg))
    return dict(base=base, masses=masses, split=split, points=pd.DataFrame(pts),
                vert=(float(k[j]), float(dg[j])))


def render_analysis(k, bands, nocc, ticks, kfac):
    ana = full_analysis(k, bands, nocc, ticks, kfac)
    st.subheader("Automatic analysis (computed from the bands)")
    c = st.columns(3)
    c[0].metric("Minimum vertical transition", f"{ana['vert'][1]:.3f} eV", f"at k = {ana['vert'][0]:.3f}")
    c[1].metric("Top-valence splitting (Δso if SOC on)",
                "none ≤1.5 eV" if np.isnan(ana["split"]) else f"{ana['split']:.3f} eV")
    c[2].metric("k(VBM) → k(CBM)", f"{ana['base']['k_VBM']:.3f} → {ana['base']['k_CBM']:.3f}")
    st.markdown("**Effective masses** (parabolic fit near the extremum, along the plotted path only)")
    st.dataframe(ana["masses"], hide_index=True)
    st.markdown("**Valley separation and vertical gaps at high-symmetry points**")
    st.dataframe(ana["points"], hide_index=True)
    st.caption("Masses depend on k-point density near the extremum and on the direction of the path. "
               "Δso is meaningful only if the calculation included spin-orbit coupling.")
    return ana


# ---------------------------------------------------------------- plots
def energy_window(b, nocc, res):
    return max(float(b[:nocc].min()), -8.0), min(float(b[nocc:].max()), res["Gap"] + 8.0)


def dos_curve(b, nocc, res, dos_data):
    lo, hi = energy_window(b, nocc, res)
    if dos_data is not None:
        e, g = dos_data
        sel = (e >= lo) & (e <= hi)
        return e[sel], g[sel]
    return dos(b, lo, hi)


def plot_ek(k, bands, nocc, res, ticks, title, kfac):
    """Plotly E-k diagram: hover shows band number, k (path units and 1/Angstrom), E and E-VBM."""
    b = bands - res["VBM"]
    fig = go.Figure()
    for i, y in enumerate(b):
        fig.add_trace(go.Scatter(
            x=k, y=y, mode="lines", line=dict(width=1.6, color="royalblue" if i < nocc else "crimson"),
            name=f"{'VB' if i < nocc else 'CB'} band {i + 1}", showlegend=False,
            customdata=np.column_stack([np.full(len(k), i + 1), k * kfac, bands[i]]),
            hovertemplate=("band %{customdata[0]:.0f}<br>k = %{x:.4f} (path units)<br>"
                           "k = %{customdata[1]:.4f} 1/Å<br>E − VBM = %{y:.4f} eV<br>"
                           "E (absolute) = %{customdata[2]:.4f} eV<extra></extra>")))
    for key, nm, col, sym in [("VBM", "VBM", "navy", "triangle-up"), ("CBM", "CBM", "darkred", "triangle-down")]:
        fig.add_trace(go.Scatter(x=[res["k_" + key]], y=[res[key] - res["VBM"]], mode="markers+text",
                                 marker=dict(size=11, color=col, symbol=sym), text=[nm], textposition="top right",
                                 name=nm, hovertemplate=f"{nm}<br>k = %{{x:.4f}}<br>E − VBM = %{{y:.4f}} eV<extra></extra>"))
    for x in ticks:
        fig.add_vline(x=x, line_width=0.7, line_color="gray")
    lo, hi = energy_window(b, nocc, res)
    fig.update_xaxes(tickvals=list(ticks.keys()), ticktext=list(ticks.values()),
                     range=[float(k.min()), float(k.max())])
    fig.update_yaxes(range=[lo, hi], title_text="E − E_VBM (eV)")
    fig.update_layout(title=f"{title} | Eg = {res['Gap']:.3f} eV ({res['Type']})", height=620,
                      hovermode="closest", showlegend=True, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_dos(bands, nocc, res, title, dos_data=None):
    """Plotly DOS (energy on the vertical axis, same E - VBM zero as the E-k diagram)."""
    b = bands - res["VBM"]
    e, g = dos_curve(b, nocc, res, dos_data)
    lo, hi = energy_window(b, nocc, res)
    fig = go.Figure(go.Scatter(x=g, y=e, fill="tozerox", line=dict(color="seagreen"), name="DOS",
                               hovertemplate="DOS = %{x:.2f}<br>E − VBM = %{y:.3f} eV<extra></extra>"))
    fig.add_hline(y=0.0, line_dash="dot", line_color="navy", annotation_text="VBM")
    fig.add_hline(y=res["Gap"], line_dash="dot", line_color="darkred", annotation_text="CBM")
    fig.update_xaxes(title_text="DOS (dos.x, full BZ)" if dos_data is not None else "DOS (path-sampled, arb.)")
    fig.update_yaxes(range=[lo, hi], title_text="E − E_VBM (eV)")
    fig.update_layout(title=f"{title} | DOS", height=620, showlegend=False,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_matplotlib(k, bands, nocc, res, ticks, title, dos_data=None):
    """Fallback (only when plotly is not installed): E-k and DOS side by side."""
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(10, 6), sharey=True,
                                  gridspec_kw={"width_ratios": [3, 1]})
    b = bands - res["VBM"]
    for i, y in enumerate(b):
        ax.plot(k, y, color="tab:blue" if i < nocc else "tab:red", lw=1.4)
    ax.scatter([res["k_VBM"]], [0.0], c="navy", zorder=5, label="VBM")
    ax.scatter([res["k_CBM"]], [res["Gap"]], c="darkred", zorder=5, label="CBM")
    for t in ticks:
        ax.axvline(t, c="k", lw=0.5)
    ax.set_xticks(list(ticks.keys()))
    ax.set_xticklabels(list(ticks.values()))
    ax.set_xlim(k.min(), k.max())
    lo, hi = energy_window(b, nocc, res)
    ax.set_ylim(lo, hi)
    ax.set_ylabel("E − E_VBM (eV)")
    ax.set_title(f"{title}  |  Eg = {res['Gap']:.3f} eV ({res['Type']})")
    ax.legend(loc="upper right")
    e, g = dos_curve(b, nocc, res, dos_data)
    axd.fill_betweenx(e, g, color="tab:green", alpha=0.5)
    axd.set_xlabel("DOS")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------- DFT I/O
def parse_dos(text):
    """dos.x output (fildos): header '# E (eV) dos(E) Int dos(E) EFermi = x eV', then E, dos, int-dos."""
    ef, rows = None, []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            if "EFermi" in line or "Efermi" in line:
                try:
                    ef = float(line.split("=")[-1].split()[0])
                except ValueError:
                    pass
            continue
        p = line.split()
        if len(p) >= 2:
            rows.append((float(p[0]), float(p[1])))
    a = np.array(rows)
    return a[:, 0], a[:, 1], ef


def dos_gap(e, g, vbm_rel, cbm_rel):
    """Gap seen in the DOS: distance between the last occupied-side and first empty-side states."""
    thr = 0.02 * g.max()
    mid = 0.5 * (vbm_rel + cbm_rel)
    lo, hi = e[(e < mid) & (g > thr)], e[(e > mid) & (g > thr)]
    return (hi.min() - lo.max()) if len(lo) and len(hi) else np.nan


def parse_gnu(text):
    """Quantum ESPRESSO bands.dat.gnu: 'k E' rows, blank line between bands."""
    ks, es, kcur, ecur = None, [], [], []
    for line in text.splitlines() + [""]:
        s = line.split()
        if len(s) >= 2:
            kcur.append(float(s[0]))
            ecur.append(float(s[1]))
        elif kcur:
            ks = np.array(kcur) if ks is None else ks
            es.append(ecur)
            kcur, ecur = [], []
    return ks, np.array(es)


def show_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Universal Semiconductor Analyzer", layout="wide")
st.title("Universal Semiconductor E–k & DFT Analysis Platform")

mode = st.sidebar.radio("Mode", ["Theoretical (model)", "Real DFT (Quantum ESPRESSO)"])
src = st.sidebar.radio("Material input", ["Database", "CIF upload", "DFT output upload"])

k = bands = ticks = dos_raw = None
nocc = 3
title, source_note, kfac = "", "", 1.0

if src == "Database":
    choice = st.selectbox("Semiconductor", list(DB.name) + [CUSTOM])
    known = choice != CUSTOM
    row = None
    if known:
        row = DB[DB.name == choice].iloc[0]
        label = row.formula
        st.caption(f"{row.formula} · {row.structure} · class: {row['class']}")
    else:
        label = st.text_input("Material name", "Custom material", key="custom_name")

    # ---- parameter window (always starts at 0) ----
    with st.expander("Model parameters (manual entry, sagle 0 pasun suru)", expanded=True):
        def num(col, key, vmax):
            return col.number_input(LABEL[key], min_value=0.0, max_value=vmax, value=0.0,
                                    step=0.01, format="%.3f", key=f"{choice}_{key}")
        c = st.columns(5)
        entered = {"a": num(c[0], "a", 20.0), "Eg": num(c[1], "Eg", 15.0),
                   "me": num(c[2], "me", 5.0), "mhh": num(c[3], "mhh", 5.0),
                   "mlh": num(c[4], "mlh", 5.0)}
        c = st.columns(4)
        entered.update({"so": num(c[0], "so", 3.0), "dG": num(c[1], "dG", 10.0),
                        "dL": num(c[2], "dL", 10.0), "dX": num(c[3], "dX", 10.0)})

    any_entered = any(v != 0 for v in entered.values())
    errs = []
    if any_entered:
        errs = validate(entered)
        for e in errs:
            st.error(e)
        if known:
            for line in compare_with_reference(entered, row):
                st.error(line)
        if not errs:
            for w in soft_warnings(entered):
                st.warning(w)
    else:
        st.info("Parameter bhara, mag 'Generate E–k diagram' dabva. Jopparyant input nahi, output nahi.")

    # ---- generate button ----
    if st.button("Generate E–k diagram", type="primary", disabled=(not any_entered) or bool(errs)):
        st.session_state["gen"] = {"choice": choice, "p": dict(entered)}
    gen = st.session_state.get("gen")

    if any_entered and not errs:
        if gen and gen["choice"] == choice and gen["p"] == entered:
            p = entered
            st.warning("Illustrative model: bands tumchya parameters war banlele aahet, mhanun khalcha analysis "
                       "tyach parameters parat deto. Real output DFT bands (upload) madhun yeto.")
            k, bands, nocc, ticks = theoretical_bands(p)
            kfac = np.pi / p["a"]
            title = f"{label} (user parameters)"
            source_note = "Source: USER parameters (not reference data), theoretical model — not DFT."
        else:
            st.info("Parameter tayar aahet. 'Generate E–k diagram' dabva.")

elif src == "CIF upload":
    up = st.file_uploader("CIF file", type="cif")
    if up is None:
        st.info("CIF upload kara, structure summary disel.")
    elif not HAS_PMG:
        st.error("pymatgen install nahi: pip install pymatgen")
    else:
        s = Structure.from_str(up.getvalue().decode(), fmt="cif")
        sga = SpacegroupAnalyzer(s, symprec=1e-3)
        st.success(f"{s.composition.reduced_formula} · {sga.get_space_group_symbol()} · "
                   f"{sga.get_crystal_system()} · {len(s)} atoms")
        st.write(pd.DataFrame({"a,b,c (Å)": s.lattice.abc, "α,β,γ (°)": s.lattice.angles}))

else:
    up = st.file_uploader("bands.dat.gnu (QE bands.x output)", type=["gnu", "dat", "txt"])
    c = st.columns(4)
    nocc = c[0].number_input("Occupied bands", 1, 400, 4,
                             help="Valence electrons / 2 (without SOC); = valence electrons with SOC.")
    alat = c[1].number_input("alat (Å)", 1.0, 30.0, 5.43,
                             help="QE alat = length of the first CELL_PARAMETERS vector (Å).")
    title = c[2].text_input("Title", "My material (DFT)")
    lblstr = c[3].text_input("Ticks  label:k,...", "", help="k from the high-symmetry points in the gnu file")
    dos_up = st.file_uploader("mat.dos from dos.x (optional, gives the real DOS)", type=["dos", "txt", "dat"])
    if dos_up is not None:
        dos_raw = parse_dos(dos_up.getvalue().decode())
    if up is not None:
        k, bands = parse_gnu(up.getvalue().decode())
        kfac = 2 * np.pi / alat
        ticks = {}
        for item in filter(None, [t.strip() for t in lblstr.split(",")]):
            lab, val = item.split(":")
            ticks[float(val)] = "Γ" if lab in ("G", "Gamma") else lab
        if not ticks:
            ticks = {float(k.min()): "start", float(k.max()): "end"}
        source_note = "Source: uploaded DFT eigenvalues (Quantum ESPRESSO / user)."
        if int(nocc) >= bands.shape[0]:
            st.error("Occupied bands ≥ total bands: increase nbnd or lower the occupied-band count.")
            bands = None

# ---------------------------------------------------------------- output (only when bands exist)
if bands is not None:
    res = analyze(k, bands, int(nocc))
    dos_data = None
    if dos_raw is not None:
        dos_data = (dos_raw[0] - res["VBM"], dos_raw[1])

    if HAS_PLOTLY:
        tab_ek, tab_dos = st.tabs(["E–k diagram", "DOS"])
        with tab_ek:
            show_chart(plot_ek(k, bands, int(nocc), res, ticks, title, kfac))
            st.caption("Hover on any band: band number, k, E − VBM and absolute E are shown.")
        with tab_dos:
            show_chart(plot_dos(bands, int(nocc), res, title, dos_data))
    else:
        st.pyplot(plot_matplotlib(k, bands, int(nocc), res, ticks, title, dos_data))
        st.info("pip install plotly for the interactive E–k / DOS tabs.")

    m = st.columns(5)
    m[0].metric("Band gap", f"{res['Gap']:.3f} eV")
    m[1].metric("Type", res["Type"])
    m[2].metric("VBM (eV)", f"{res['VBM']:.3f}")
    m[3].metric("CBM (eV)", f"{res['CBM']:.3f}")
    m[4].metric("Bands", f"{int(nocc)} VB / {bands.shape[0] - int(nocc)} CB")
    st.caption(source_note)
    if dos_data is not None:
        dg = dos_gap(dos_data[0], dos_data[1], 0.0, res["Gap"])
        st.success(f"Real DOS loaded (dos.x). Gap from DOS ≈ {dg:.2f} eV vs {res['Gap']:.3f} eV from bands "
                   "(they differ slightly because of smearing and the finite k-mesh).")
        if dos_raw[2] is not None:
            st.caption(f"Fermi level in mat.dos: {dos_raw[2]:.3f} eV (absolute). "
                       "Use the same SCF/nscf run as the bands file so both share one energy zero.")
    render_analysis(k, bands, int(nocc), ticks, kfac)
