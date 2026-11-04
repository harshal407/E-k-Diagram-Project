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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
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


def plot_all(k, bands, nocc, res, ticks, title, shift, dos_data=None):
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
    if dos_data is not None:
        e, g = dos_data
        axd.set_xlabel("DOS (dos.x, full BZ)")
    else:
        e, g = dos(b, *ax.get_ylim())
        axd.set_xlabel("DOS (path-sampled, arb.)")
    axd.fill_betweenx(e, g, color="tab:green", alpha=0.5)
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


def plot_interactive(k, bands, nocc, res, ticks, title, kfac, dos_data=None):
    """Plotly E-k (+DOS): hover shows band number, k (path units and 1/Angstrom), E and E-VBM."""
    b = bands - res["VBM"]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.78, 0.22],
                        horizontal_spacing=0.02)
    for i, y in enumerate(b):
        fig.add_trace(go.Scatter(
            x=k, y=y, mode="lines", line=dict(width=1.6, color="royalblue" if i < nocc else "crimson"),
            name=f"{'VB' if i < nocc else 'CB'} band {i + 1}", showlegend=False,
            customdata=np.column_stack([np.full(len(k), i + 1), k * kfac, bands[i]]),
            hovertemplate=("band %{customdata[0]:.0f}<br>k = %{x:.4f} (2π/alat)<br>"
                           "k = %{customdata[1]:.4f} 1/Å<br>E − VBM = %{y:.4f} eV<br>"
                           "E (absolute) = %{customdata[2]:.4f} eV<extra></extra>")), row=1, col=1)
    for key, nm, col, sym in [("VBM", "VBM", "navy", "triangle-up"), ("CBM", "CBM", "darkred", "triangle-down")]:
        fig.add_trace(go.Scatter(x=[res["k_" + key]], y=[res[key] - res["VBM"]], mode="markers+text",
                                 marker=dict(size=11, color=col, symbol=sym), text=[nm], textposition="top right",
                                 name=nm, hovertemplate=f"{nm}<br>k = %{{x:.4f}}<br>E − VBM = %{{y:.4f}} eV<extra></extra>"),
                      row=1, col=1)
    for x, lab in ticks.items():
        fig.add_vline(x=x, line_width=0.7, line_color="gray", row=1, col=1)
    lo, hi = float(b[:nocc].min()), float(b[nocc:].max())
    if dos_data is not None:
        e, g = dos_data
        sel = (e >= max(lo, -8)) & (e <= min(hi, res["Gap"] + 8))
        e, g = e[sel], g[sel]
    else:
        e, g = dos(b, max(lo, -8), min(hi, res["Gap"] + 8))
    fig.add_trace(go.Scatter(x=g, y=e, fill="tozerox", line=dict(color="seagreen"), name="DOS",
                             hovertemplate="DOS = %{x:.2f}<br>E − VBM = %{y:.3f} eV<extra></extra>"), row=1, col=2)
    fig.update_xaxes(tickvals=list(ticks.keys()), ticktext=list(ticks.values()),
                     range=[float(k.min()), float(k.max())], row=1, col=1)
    fig.update_yaxes(range=[max(lo, -8), min(hi, res["Gap"] + 8)], title_text="E − E_VBM (eV)", row=1, col=1)
    fig.update_xaxes(title_text="DOS (dos.x, full BZ)" if dos_data is not None else "DOS (path-sampled)", row=1, col=2)
    fig.update_layout(title=f"{title} | Eg = {res['Gap']:.3f} eV ({res['Type']})", height=620,
                      hovermode="closest", showlegend=True, margin=dict(l=10, r=10, t=50, b=10))
    return fig


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


def kronig_penney(P, a, mr, emax=15.0, n=8000):
    """cos(ka) = cos(alpha a) + P sin(alpha a)/(alpha a),  alpha = sqrt(2 m E)/hbar.
    P -> 0: free electron;  P -> infinity: E_n = n^2 h^2 / (8 m a^2)."""
    E = np.linspace(1e-4, emax, n)
    al = np.sqrt(E * mr / HB2_2M)
    f = np.cos(al * a) + P * np.sin(al * a) / (al * a)
    ok = np.abs(f) <= 1
    return E, np.arccos(np.clip(f, -1, 1)) / a, ok


def segments(ok):
    idx = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    return list(zip(idx[::2], idx[1::2]))


def qe_inputs(struct, nbnd, ecut, kmesh, pseudo_dir, soc, nscf_mesh):
    from pymatgen.core import Element
    sga = SpacegroupAnalyzer(struct, symprec=1e-3)
    info = f"{sga.get_space_group_symbol()} (No. {sga.get_space_group_number()}), {sga.get_crystal_system()}"
    cell = (struct.lattice.matrix.tolist(), struct.frac_coords.tolist(), [s.specie.Z for s in struct])
    kpts = labels = None
    if HAS_SEEK:
        kp = seekpath.get_explicit_k_path(cell)
        lat, pos, num = np.array(kp["primitive_lattice"]), kp["primitive_positions"], kp["primitive_types"]
        kpts, labels = np.array(kp["explicit_kpoints_rel"]), kp["explicit_kpoints_labels"]
        recip = np.array(kp["reciprocal_primitive_lattice"])
    else:
        lat, pos, num = struct.lattice.matrix, struct.frac_coords, [s.specie.Z for s in struct]
    alat = float(np.linalg.norm(lat[0]))  # QE alat when CELL_PARAMETERS are in angstrom
    els = sorted(set(num))
    sym = {z: Element.from_Z(z).symbol for z in els}
    names = [sym[z] for z in els]

    # tick positions in the same units bands.x writes (cumulative distance, 2pi/alat)
    ticks = {}
    if kpts is not None:
        kc = kpts @ recip / (2 * np.pi / alat)
        dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(kc, axis=0), axis=1))]
        for l, d in zip(labels, dist):
            if l:
                ticks[float(d)] = "Γ" if l == "GAMMA" else l

    def block(calc, kblock, bands=True):
        so = ",\n  noncolin=.true., lspinorbit=.true." if soc else ""
        nb = f", nbnd={nbnd}" if bands else ""
        s = (f"&CONTROL\n  calculation='{calc}', prefix='mat', outdir='./tmp', pseudo_dir='{pseudo_dir}'\n/\n"
             f"&SYSTEM\n  ibrav=0, nat={len(pos)}, ntyp={len(els)}, ecutwfc={ecut}{nb},\n"
             f"  occupations='fixed'{so}\n/\n&ELECTRONS\n  conv_thr=1e-9\n/\n")
        if calc == "vc-relax":
            s += "&IONS\n/\n&CELL\n  press_conv_thr=0.5\n/\n"
        s += "ATOMIC_SPECIES\n"
        for z in els:
            s += f"{sym[z]} {Element.from_Z(z).atomic_mass:.4f} {sym[z]}{'.rel' if soc else ''}.UPF\n"
        s += "CELL_PARAMETERS angstrom\n" + "\n".join(" ".join(f"{v:.8f}" for v in r) for r in lat)
        s += "\nATOMIC_POSITIONS crystal\n" + "\n".join(
            f"{sym[z]} " + " ".join(f"{v:.8f}" for v in p) for z, p in zip(num, pos)) + "\n"
        return s + kblock

    mesh = lambda m: f"K_POINTS automatic\n{m[0]} {m[1]} {m[2]} 0 0 0\n"
    files = {"relax.in": block("vc-relax", mesh(kmesh), False),
             "scf.in": block("scf", mesh(kmesh), False)}
    if kpts is not None:
        files["bands.in"] = block("bands", f"K_POINTS crystal\n{len(kpts)}\n" + "\n".join(
            f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} 1" for p in kpts) + "\n")
    files["nscf.in"] = block("nscf", mesh(nscf_mesh))
    files["bands_post.in"] = "&BANDS\n  prefix='mat', outdir='./tmp', filband='bands.dat'\n/\n"
    files["dos.in"] = "&DOS\n  prefix='mat', outdir='./tmp', fildos='mat.dos', DeltaE=0.01, degauss=0.005\n/\n"
    files["projwfc.in"] = "&PROJWFC\n  prefix='mat', outdir='./tmp', filpdos='mat', DeltaE=0.02\n/\n"
    conv = ("#!/bin/bash\n# convergence of total energy: edit the lists, run, compare lines starting with '!'\n"
            "for e in 30 40 50 60 70 80; do\n  sed \"s/ecutwfc=[0-9.]*/ecutwfc=$e/\" scf.in > c_ecut_$e.in\n"
            "  pw.x -in c_ecut_$e.in > c_ecut_$e.out; echo ecut=$e $(grep '^!' c_ecut_$e.out)\ndone\n"
            "for n in 4 6 8 10 12; do\n  sed \"s/^[0-9]* [0-9]* [0-9]* 0 0 0/$n $n $n 0 0 0/\" scf.in > c_k_$n.in\n"
            "  pw.x -in c_k_$n.in > c_k_$n.out; echo k=$n $(grep '^!' c_k_$n.out)\ndone\n")
    files["convergence.sh"] = conv
    return dict(info=info, names=names, alat=alat, files=files, ticks=ticks)


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Universal Semiconductor Analyzer", layout="wide")
st.title("Universal Semiconductor E–k & DFT Analysis Platform")

mode = st.sidebar.radio("Mode", ["Theoretical (model)", "Real DFT (Quantum ESPRESSO)"])
src = st.sidebar.radio("Material input", ["Database", "CIF upload", "DFT output upload"])
tabs = st.tabs(["Bands & Analysis", "Theory (E–k equations)", "Quantum ESPRESSO workflow", "Report", "About"])

k = bands = ticks = dos_raw = None
nocc = 3
title, source_note, kfac = "", "", 1.0

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
        st.warning("Illustrative model: these bands are built from the parameters above, so the analysis below "
                   "only returns them. Real outputs come from DFT bands (upload) — see the workflow tab.")
        k, bands, nocc, ticks = theoretical_bands(p)
        kfac = np.pi / p["a"]
        title = f"{row.formula} (model)"
        source_note = "Source: theoretical multi-band effective-mass model (not DFT)."

    elif src == "CIF upload":
        up = st.file_uploader("CIF file", type="cif")
        if up is None:
            st.info("CIF upload करा; relax/scf/bands/nscf inputs 'Quantum ESPRESSO workflow' tab मध्ये तयार होतील.")
        elif not HAS_PMG:
            st.error("pymatgen install नाही: pip install pymatgen seekpath")
        else:
            s = Structure.from_str(up.getvalue().decode(), fmt="cif")
            st.session_state["struct"] = s
            sga = SpacegroupAnalyzer(s, symprec=1e-3)
            st.success(f"{s.composition.reduced_formula} · {sga.get_space_group_symbol()} · "
                       f"{sga.get_crystal_system()} · {len(s)} atoms")
            st.write(pd.DataFrame({"a,b,c (Å)": s.lattice.abc, "α,β,γ (°)": s.lattice.angles}))

    else:
        up = st.file_uploader("bands.dat.gnu (QE bands.x output)", type=["gnu", "dat", "txt"])
        c = st.columns(4)
        nocc = c[0].number_input("Occupied bands", 1, 400, int(st.session_state.get("nocc_rec", 4)),
                                 help="Valence electrons / 2 (without SOC); = valence electrons with SOC.")
        alat = c[1].number_input("alat (Å)", 1.0, 30.0, float(st.session_state.get("alat", 5.43)),
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
            if not ticks and "ticks" in st.session_state:
                ticks = st.session_state["ticks"]
                st.caption("Ticks taken from the CIF workflow tab.")
            if not ticks:
                ticks = {float(k.min()): "start", float(k.max()): "end"}
            source_note = "Source: uploaded DFT eigenvalues (Quantum ESPRESSO / user)."
            if int(nocc) >= bands.shape[0]:
                st.error("Occupied bands ≥ total bands: increase nbnd or lower the occupied-band count.")
                bands = None

    if bands is not None:
        res = analyze(k, bands, int(nocc))
        dos_data = None
        if dos_raw is not None:
            dos_data = (dos_raw[0] - res["VBM"], dos_raw[1])
        fig = plot_all(k, bands, int(nocc), res, ticks, title, res["VBM"], dos_data)
        if HAS_PLOTLY:
            pfig = plot_interactive(k, bands, int(nocc), res, ticks, title, kfac, dos_data)
            try:
                st.plotly_chart(pfig, width="stretch")
            except TypeError:
                st.plotly_chart(pfig, use_container_width=True)
            st.caption("Hover on any band: band number, k, E − VBM and absolute E are shown.")
        else:
            st.pyplot(fig)
            st.info("pip install plotly for the interactive hover version.")
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
                       "(DOS value is smeared by degauss, so it is slightly smaller).")
            if dos_raw[2] is not None:
                st.caption(f"Fermi level in mat.dos: {dos_raw[2]:.3f} eV (absolute). "
                           "Use the same SCF/nscf run as the bands file so both share one energy zero.")
        ana = render_analysis(k, bands, int(nocc), ticks, kfac)
        st.session_state["report"] = (k, bands, int(nocc), res, ticks, title, source_note, fig, ana)

with tabs[1]:
    st.subheader("E–k relations behind the diagrams")
    c1, c2 = st.columns(2)
    c1.markdown("**Free particle**")
    c1.latex(r"E=\frac{p^2}{2m},\quad p=\hbar k \;\Rightarrow\; E=\frac{\hbar^2k^2}{2m}=Ak^2,\quad A=\frac{\hbar^2}{2m}")
    c1.markdown("**Crystal (effective mass)**")
    c1.latex(r"E=\frac{\hbar^2k^2}{2m^*},\qquad m^*=\hbar^2\left(\frac{d^2E}{dk^2}\right)^{-1}")
    c2.markdown("**3D density of states (parabolic band)**")
    c2.latex(r"g(E)=\frac{1}{2\pi^2}\left(\frac{2m^*}{\hbar^2}\right)^{3/2}\sqrt{E},\qquad k\propto\sqrt{E}")
    c2.markdown("**Periodic potential (Kronig–Penney)**")
    c2.latex(r"\cos ka=\cos\alpha a+P\,\frac{\sin\alpha a}{\alpha a},\quad \alpha=\frac{\sqrt{2mE}}{\hbar}")
    st.markdown(r"$P\to0$: free electron, $E=\hbar^2k^2/2m$. $P\to\infty$: isolated levels, "
                r"$E_n=n^2h^2/(8ma^2)$ (particle in a box).")
    st.caption("Note on the whiteboard: in 3D, E ∝ k², so k ∝ E^(1/2) and g(E) ∝ E^(1/2). "
               "The line 'E ∝ k^(1/2)' should read k ∝ E^(1/2).")
    c = st.columns(4)
    P = c[0].slider("P (barrier strength)", 0.0, 40.0, 3.0, 0.5)
    a_kp = c[1].slider("a (Å)", 2.0, 8.0, 5.0, 0.1)
    mr = c[2].slider("m*/m0", 0.05, 2.0, 1.0, 0.05)
    emax = c[3].slider("E max (eV)", 5.0, 40.0, 15.0, 1.0)
    E, kk, ok = kronig_penney(P, a_kp, mr, emax)
    fig2, (axk, axg) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [2, 1]})
    segs = segments(ok)
    for s0, s1 in segs:
        axk.plot(kk[s0:s1], E[s0:s1], c="tab:blue"); axk.plot(-kk[s0:s1], E[s0:s1], c="tab:blue")
    kx = np.linspace(-np.pi / a_kp, np.pi / a_kp, 200)
    axk.plot(kx, HB2_2M * kx ** 2 / mr, "--", c="gray", label="E = ħ²k²/2m* (parabola)")
    axk.set_xlim(-np.pi / a_kp, np.pi / a_kp); axk.set_ylim(0, emax)
    axk.set_xlabel("k (1/Å)"); axk.set_ylabel("E (eV)"); axk.legend(); axk.set_title("Kronig–Penney bands")
    eg_ = np.linspace(0, emax, 300)
    axg.plot((mr / HB2_2M) ** 1.5 / (2 * np.pi ** 2) * np.sqrt(eg_), eg_)
    axg.set_xlabel("g(E) (states/eV/Å³)"); axg.set_title("Parabolic DOS ∝ √E")
    plt.tight_layout(); st.pyplot(fig2)
    gaps = [(E[segs[i][1] - 1], E[segs[i + 1][0]]) for i in range(len(segs) - 1)]
    st.write("Band gaps (eV):", ", ".join(f"{lo:.2f}–{hi:.2f} (width {hi - lo:.2f})" for lo, hi in gaps) or "none")
    st.write("Box levels E_n = n²h²/(8ma²) (eV):",
             ", ".join(f"n={n}: {np.pi ** 2 * n ** 2 * HB2_2M / (mr * a_kp ** 2):.2f}" for n in (1, 2, 3)))

with tabs[2]:
    st.markdown("**Workflow:** relax → scf → bands (→ bands.x) → upload `bands.dat.gnu`. "
                "Real DOS: nscf (dense mesh) → dos.x → upload `mat.dos`. PDOS: projwfc.x. "
                "Use the relaxed structure for scf/bands (copy it from relax.out into the later inputs).")
    s = st.session_state.get("struct")
    if s is None:
        st.info("आधी CIF upload करा.")
    elif not HAS_PMG:
        st.error("pymatgen required")
    else:
        c = st.columns(3)
        soc = c[0].checkbox("Spin-orbit coupling (needs fully-relativistic UPF)")
        nel = c[1].number_input("Total valence electrons (Σ z_valence from UPF)", 0, 2000, 0)
        ecut = c[2].number_input("ecutwfc (Ry)", 10, 200, 50)
        rec_occ = (nel if soc else nel // 2) if nel else 0
        rec_nb = rec_occ + (16 if soc else 8) if nel else 16
        if nel:
            st.session_state["nocc_rec"] = rec_occ
            st.caption(f"Occupied bands = {rec_occ}; suggested nbnd = {rec_nb} (valence + conduction bands).")
        c = st.columns(7)
        nbnd = c[0].number_input("nbnd", 4, 800, int(rec_nb), key=f"nb{rec_nb}")
        km = [c[1].number_input("k1", 1, 40, 8), c[2].number_input("k2", 1, 40, 8), c[3].number_input("k3", 1, 40, 8)]
        nk = [c[4].number_input("nscf k1", 1, 80, 16), c[5].number_input("nscf k2", 1, 80, 16),
              c[6].number_input("nscf k3", 1, 80, 16)]
        pdir = st.text_input("pseudo_dir", "./pseudo")
        out = qe_inputs(s, int(nbnd), int(ecut), km, pdir, soc, nk)
        st.session_state["alat"] = out["alat"]
        st.session_state["ticks"] = out["ticks"]
        st.write(f"Symmetry: **{out['info']}** · Elements: {', '.join(out['names'])} · alat = {out['alat']:.4f} Å · "
                 f"UPF: {', '.join(n + ('.rel' if soc else '') + '.UPF' for n in out['names'])}")
        if "bands.in" not in out["files"]:
            st.warning("seekpath install नाही → bands input नाही. pip install seekpath")
        else:
            st.write("High-symmetry ticks (bands.x k-units):", {v: round(x, 4) for x, v in out["ticks"].items()})
        for name, txt in out["files"].items():
            with st.expander(name):
                st.download_button(f"Download {name}", txt, name, key="dl" + name)
                st.code(txt if len(txt) < 2500 else txt[:2500] + "\n...")
        st.markdown("**Checklist**")
        for item in ["Lattice/atoms relaxed (vc-relax) and relaxed structure reused",
                     "ecutwfc and k-mesh converged (convergence.sh)",
                     "Correct pseudopotentials (SOC: fully relativistic)",
                     "nbnd ≥ occupied + 6–8 conduction bands",
                     "Dense k-points near VBM/CBM for effective masses",
                     "Functional noted (PBE underestimates gaps; HSE/GW for accuracy)"]:
            st.checkbox(item, key=item)

with tabs[3]:
    rep = st.session_state.get("report")
    if rep is None:
        st.info("आधी bands तयार करा.")
    else:
        k_, b_, no_, r_, tk_, t_, note_, fig_, ana_ = rep
        df = pd.DataFrame(b_.T - r_["VBM"], columns=[f"band_{i + 1}" for i in range(b_.shape[0])])
        df.insert(0, "k", k_)
        c = st.columns(3)
        c[0].download_button("Bands CSV (E − VBM)", df.to_csv(index=False), "bands.csv")
        c[1].download_button("Effective masses CSV", ana_["masses"].to_csv(index=False), "masses.csv")
        c[2].download_button("High-symmetry points CSV", ana_["points"].to_csv(index=False), "points.csv")
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            pdf.savefig(fig_)
            tf = plt.figure(figsize=(8.5, 11))
            lines = [f"Report: {t_}", "", f"Band gap: {r_['Gap']:.4f} eV ({r_['Type']})",
                     f"VBM: {r_['VBM']:.4f} eV at k={r_['k_VBM']:.4f}",
                     f"CBM: {r_['CBM']:.4f} eV at k={r_['k_CBM']:.4f}", f"Valence bands: {no_}",
                     f"Min vertical transition: {ana_['vert'][1]:.4f} eV at k={ana_['vert'][0]:.4f}",
                     f"Top-valence splitting: {ana_['split']:.4f} eV", "", "Effective masses (m*/m0):",
                     ana_["masses"].to_string(index=False), "", "High-symmetry points:",
                     ana_["points"].to_string(index=False), "", note_]
            tf.text(0.05, 0.97, "\n".join(lines), fontsize=8, va="top", family="monospace")
            pdf.savefig(tf)
        st.download_button("PDF report", buf.getvalue(), "report.pdf")

with tabs[4]:
    st.markdown("""
**Scientific limits**
- Database/Theoretical mode is an illustrative bounded effective-mass model; its 'analysis' returns its own inputs.
  Research outputs come from DFT (or EPM/tight-binding) bands.
- Effective masses are parabolic fits along the plotted path; they need dense k-points near the extremum.
- Δso needs spin-orbit coupling; plain PBE underestimates gaps (use HSE/GW for accuracy).
- Database values are approximate literature numbers; verify before publishing.
- Run heavy QE jobs locally or on a Linux server, not on Streamlit Cloud.
""")
