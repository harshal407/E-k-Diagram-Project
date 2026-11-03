"""Universal Semiconductor E-k & DFT Analysis Platform (Streamlit)

Inputs : 1) Database material  2) CIF upload  3) DFT output upload
Modes  : Theoretical (model bands)  |  Real DFT (Quantum ESPRESSO)
Run    : streamlit run app.py
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    HAS_PMG = True
except Exception:
    HAS_PMG = False
try:
    import seekpath
    HAS_SEEK = True
except Exception:
    HAS_SEEK = False

HB2_2M = 3.80998  # hbar^2 / 2 m0 in eV*Angstrom^2

# ---------------------------------------------------------------- database
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

# ---------------------------------------------------------------- model
def band(x, centre, e0, sign, mass, a):
    """Bounded effective-mass band: curvature at `centre` equals hbar^2/2m*."""
    w = float(np.clip(4 * HB2_2M / (mass * a ** 2), 0.8, 8.0))
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
                Type="Direct" if direct else "Indirect")


def dos(bands, emin, emax, sigma=0.05, n=600):
    e = np.linspace(emin, emax, n)
    g = np.zeros_like(e)
    for en in bands.ravel():
        g += np.exp(-((e - en) ** 2) / (2 * sigma ** 2))
    return e, g / (sigma * np.sqrt(2 * np.pi)) / bands.shape[1]


def plot_all(k, bands, nocc, res, ticks, title, shift):
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(10, 6), sharey=True,
                                  gridspec_kw={"width_ratios": [3, 1]})
    b = bands - shift
    for i, y in enumerate(b):
        ax.plot(k, y, color="tab:blue" if i < nocc else "tab:red", lw=1.4)
    ax.axhline(res["VBM"] - shift, ls=":", c="gray")
    ax.axhline(res["CBM"] - shift, ls=":", c="gray")
    ax.scatter([res["k_VBM"]], [res["VBM"] - shift], c="navy", zorder=5, label="VBM")
    ax.scatter([res["k_CBM"]], [res["CBM"] - shift], c="darkred", zorder=5, label="CBM")
    for t in ticks:
        ax.axvline(t, c="k", lw=0.5)
    ax.set_xticks(list(ticks.keys()))
    ax.set_xticklabels(list(ticks.values()))
    ax.set_xlim(k.min(), k.max())
    lo = b[:nocc].min() - 1 if nocc else b.min()
    ax.set_ylim(max(lo, b.min() - 0.5), min(b[nocc:].max(), b[:nocc].max() + res["Gap"] + 6) + 0.5)
    ax.set_ylabel("E − E_VBM (eV)")
    ax.set_title(f"{title}  |  Eg = {res['Gap']:.3f} eV ({res['Type']})")
    ax.legend(loc="upper right")
    e, g = dos(b, *ax.get_ylim())
    axd.fill_betweenx(e, g, color="tab:green", alpha=0.5)
    axd.set_xlabel("DOS (path-sampled, arb.)")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------- DFT I/O
def parse_gnu(text):
    """Quantum ESPRESSO bands.dat.gnu: 'k E' rows, blank line between bands."""
    ks, es, kcur, ecur = None, [], [], []
    for line in text.splitlines() + [""]:
        s = line.split()
        if len(s) >= 2:
            kcur.append(float(s[0])); ecur.append(float(s[1]))
        elif kcur:
            ks = np.array(kcur) if ks is None else ks
            es.append(ecur); kcur, ecur = [], []
    return ks, np.array(es)


def qe_inputs(struct, nbnd, ecut, kmesh, functional, pseudo_dir):
    sga = SpacegroupAnalyzer(struct, symprec=1e-3)
    info = f"{sga.get_space_group_symbol()} (No. {sga.get_space_group_number()}), {sga.get_crystal_system()}"
    cell = (struct.lattice.matrix, struct.frac_coords, [s.specie.Z for s in struct])
    if HAS_SEEK:
        kp = seekpath.get_explicit_k_path(cell)
        lat, pos, num = kp["primitive_lattice"], kp["primitive_positions"], kp["primitive_types"]
        kpts, labels, lin = kp["explicit_kpoints_rel"], kp["explicit_kpoints_labels"], kp["explicit_kpoints_linearcoord"]
    else:
        lat, pos, num = struct.lattice.matrix, struct.frac_coords, [s.specie.Z for s in struct]
        kpts, labels, lin = None, None, None
    from pymatgen.core import Element
    els = sorted(set(num))
    sym = {z: Element.from_Z(z).symbol for z in els}
    names = [sym[z] for z in els]

    def block(calc, kblock):
        s = f"""&CONTROL
  calculation='{calc}', prefix='mat', outdir='./tmp', pseudo_dir='{pseudo_dir}'
/
&SYSTEM
  ibrav=0, nat={len(pos)}, ntyp={len(els)}, ecutwfc={ecut}, nbnd={nbnd}
  occupations='fixed'
/
&ELECTRONS
  conv_thr=1e-9
/
ATOMIC_SPECIES
"""
        for z in els:
            s += f"{sym[z]} {Element.from_Z(z).atomic_mass:.4f} {sym[z]}.UPF\n"
        s += "CELL_PARAMETERS angstrom\n"
        s += "\n".join(" ".join(f"{v:.8f}" for v in r) for r in lat) + "\nATOMIC_POSITIONS crystal\n"
        s += "\n".join(f"{sym[z]} " + " ".join(f"{v:.8f}" for v in p) for z, p in zip(num, pos)) + "\n"
        return s + kblock

    scf = block("scf", f"K_POINTS automatic\n{kmesh[0]} {kmesh[1]} {kmesh[2]} 0 0 0\n")
    bands_in = None
    if kpts is not None:
        kb = f"K_POINTS crystal\n{len(kpts)}\n" + "\n".join(
            f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} 1" for p in kpts) + "\n"
        bands_in = block("bands", kb)
    ticks = {}
    if labels is not None:
        for l, x in zip(labels, lin):
            if l:
                ticks[float(x)] = l.replace("GAMMA", "Γ")
    return info, names, scf, bands_in, ticks


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Universal Semiconductor Analyzer", layout="wide")
st.title("Universal Semiconductor E–k & DFT Analysis Platform")

mode = st.sidebar.radio("Mode", ["Theoretical (model)", "Real DFT (Quantum ESPRESSO)"])
src = st.sidebar.radio("Material input", ["Database", "CIF upload", "DFT output upload"])
tabs = st.tabs(["Bands & Analysis", "Quantum ESPRESSO input", "Report", "About"])

k = bands = ticks = None
nocc = 3
title, source_note = "", ""

with tabs[0]:
    if src == "Database":
        row = DB[DB.name == st.selectbox("Semiconductor", DB.name)].iloc[0]
        st.caption(f"{row.formula} · {row.structure} · a = {row.a} Å · class: {row['class']}")
        p = row.to_dict()
        with st.expander("Model parameters (editable)"):
            c = st.columns(4)
            p["Eg"] = c[0].number_input("Eg (eV)", 0.1, 10.0, float(p["Eg"]))
            p["me"] = c[1].number_input("m*e", 0.01, 5.0, float(p["me"]))
            p["mhh"] = c[2].number_input("m*hh", 0.01, 5.0, float(p["mhh"]))
            p["mlh"] = c[3].number_input("m*lh", 0.01, 5.0, float(p["mlh"]))
            c = st.columns(4)
            p["so"] = c[0].number_input("Δso (eV)", 0.0, 2.0, float(p["so"]))
            p["dG"] = c[1].number_input("Γ valley (eV)", 0.0, 5.0, float(p["dG"]))
            p["dL"] = c[2].number_input("L valley (eV)", 0.0, 5.0, float(p["dL"]))
            p["dX"] = c[3].number_input("X valley (eV)", 0.0, 5.0, float(p["dX"]))
        if mode.startswith("Real"):
            st.info("Real DFT needs eigenvalues: generate QE inputs from a CIF (tab 2), run pw.x, "
                    "then upload bands.dat.gnu using 'DFT output upload'. Showing the model meanwhile.")
        k, bands, nocc, ticks = theoretical_bands(p)
        title = f"{row.formula} (model)"
        source_note = "Source: theoretical multi-band effective-mass model (not DFT)."

    elif src == "CIF upload":
        up = st.file_uploader("CIF file", type="cif")
        if up is None:
            st.info("CIF upload करा; QE input 'Quantum ESPRESSO input' tab मध्ये तयार होईल.")
        elif not HAS_PMG:
            st.error("pymatgen install नाही: pip install pymatgen seekpath")
        else:
            s = Structure.from_str(up.getvalue().decode(), fmt="cif")
            st.session_state["struct"] = s
            sga = SpacegroupAnalyzer(s, symprec=1e-3)
            st.success(f"{s.composition.reduced_formula} · {sga.get_space_group_symbol()} · "
                       f"{sga.get_crystal_system()} · {len(s)} atoms")
            st.write(pd.DataFrame({"a,b,c (Å)": s.lattice.abc, "α,β,γ (°)": s.lattice.angles}))
            st.caption("Bands साठी: QE tab मधून inputs घ्या → pw.x चालवा → 'DFT output upload'.")

    else:
        up = st.file_uploader("bands.dat.gnu (QE bands.x output)", type=["gnu", "dat", "txt"])
        c = st.columns(3)
        nocc = c[0].number_input("Occupied (valence) bands", 1, 200, 4)
        ef_lbl = c[1].text_input("Title", "My material (DFT)")
        lblstr = c[2].text_input("Ticks  label:k,label:k", "")
        if up is not None:
            k, bands = parse_gnu(up.getvalue().decode())
            title = ef_lbl
            ticks = {}
            for item in filter(None, [t.strip() for t in lblstr.split(",")]):
                lab, val = item.split(":")
                ticks[float(val)] = lab.replace("G", "Γ") if lab == "G" else lab
            if not ticks:
                ticks = {k.min(): "start", k.max(): "end"}
            source_note = "Source: uploaded DFT eigenvalues (Quantum ESPRESSO / user)."

    if bands is not None:
        res = analyze(k, bands, int(nocc))
        fig = plot_all(k, bands, int(nocc), res, ticks, title, res["VBM"])
        st.pyplot(fig)
        m = st.columns(5)
        m[0].metric("Band gap", f"{res['Gap']:.3f} eV")
        m[1].metric("Type", res["Type"])
        m[2].metric("VBM (eV)", f"{res['VBM']:.3f}")
        m[3].metric("CBM (eV)", f"{res['CBM']:.3f}")
        m[4].metric("Bands", f"{int(nocc)} VB / {bands.shape[0] - int(nocc)} CB")
        st.caption(source_note)
        st.session_state["report"] = (k, bands, int(nocc), res, ticks, title, source_note, fig)

with tabs[1]:
    s = st.session_state.get("struct")
    if s is None:
        st.info("आधी CIF upload करा.")
    elif not HAS_PMG:
        st.error("pymatgen required")
    else:
        c = st.columns(5)
        nbnd = c[0].number_input("nbnd", 4, 400, 16)
        ecut = c[1].number_input("ecutwfc (Ry)", 10, 200, 50)
        km = [c[2].number_input("k1", 1, 30, 8), c[3].number_input("k2", 1, 30, 8),
              c[4].number_input("k3", 1, 30, 8)]
        pdir = st.text_input("pseudo_dir", "./pseudo")
        info, names, scf, bin_, tk = qe_inputs(s, int(nbnd), int(ecut), km, "PBE", pdir)
        st.write(f"Symmetry: **{info}** · Elements: {', '.join(names)} · "
                 f"आवश्यक UPF: {', '.join(n + '.UPF' for n in names)}")
        if not HAS_SEEK:
            st.warning("seekpath install नाही → bands input नाही. pip install seekpath")
        st.download_button("scf.in", scf, "scf.in")
        st.code(scf)
        if bin_:
            st.download_button("bands.in", bin_, "bands.in")
            st.code(bin_[:1500] + "\n...")
            st.write("High-symmetry points (linear coord):", tk)
            st.markdown("Run: `pw.x -in scf.in > scf.out` → `pw.x -in bands.in > bands.out` → "
                        "`bands.x` (filband='bands.dat', plot_2d no) → upload `bands.dat.gnu`.")

with tabs[2]:
    rep = st.session_state.get("report")
    if rep is None:
        st.info("आधी bands तयार करा.")
    else:
        k_, b_, no_, r_, tk_, t_, note_, fig_ = rep
        df = pd.DataFrame(b_.T - r_["VBM"], columns=[f"band_{i + 1}" for i in range(b_.shape[0])])
        df.insert(0, "k", k_)
        st.download_button("CSV (E − VBM)", df.to_csv(index=False), "bands.csv")
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            pdf.savefig(fig_)
            tf = plt.figure(figsize=(8.5, 6)); tf.text(0.05, 0.95, f"Report: {t_}", fontsize=14, va="top")
            lines = [f"Band gap: {r_['Gap']:.4f} eV ({r_['Type']})", f"VBM: {r_['VBM']:.4f} eV at k={r_['k_VBM']:.4f}",
                     f"CBM: {r_['CBM']:.4f} eV at k={r_['k_CBM']:.4f}", f"Valence bands: {no_}", note_]
            tf.text(0.05, 0.85, "\n".join(lines), fontsize=11, va="top"); pdf.savefig(tf)
        st.download_button("PDF report", buf.getvalue(), "report.pdf")
        st.dataframe(pd.DataFrame([r_]))

with tabs[3]:
    st.markdown("""
**Scientific limits**
- Theoretical mode is a bounded effective-mass multi-band model along L–Γ–X: it does not give the true band structure.
  Gap + effective mass alone cannot reproduce real bands; use DFT eigenvalues for that.
- Database values are approximate literature numbers; edit them as needed.
- Plain PBE underestimates gaps; use HSE/GW for quantitative gaps.
- Run heavy QE jobs locally or on a Linux server, not on Streamlit Cloud. Use Streamlit only for inputs, analysis and plots.
- DFT data and model data are always labelled with their source.
""")
