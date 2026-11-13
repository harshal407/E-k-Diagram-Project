"""E-k Theoretical mode: parameters -> bands -> calculations FROM the bands.

Flow
1. Reference values (your table) are stored here, fixed, hidden from the input fields.
2. Manual fields start at 0 for every material.
   Known material: a field left at 0 uses the reference value.
   If you type a different value: a short "Not ideal value" box appears (it disappears
   by itself) and the diagram + all results follow YOUR value.
3. New material: everything 0; diagram + results follow whatever you enter.
4. Every result (gap, VBM/CBM, direct/indirect, valley separations, effective
   masses, Delta_so, vertical transitions) is computed numerically from the
   plotted band arrays, not read back from the inputs.
Run:  streamlit run ek_parameters_module.py
"""
import numpy as np

PARAMS = ["a", "c", "Eg", "me", "mhh", "mlh", "so", "dG", "dL", "dX", "Ep", "dcr"]
REFERENCE = {
    "Silicon":           dict(a=5.431, c=5.431, Eg=1.12, me=0.26,  mhh=0.49, mlh=0.16,  so=0.044, dG=2.30, dL=1.10, dX=0.00),
    "Germanium":         dict(a=5.658, c=5.658, Eg=0.66, me=0.12,  mhh=0.28, mlh=0.044, so=0.29,  dG=0.14, dL=0.00, dX=0.19),
    "Gallium Arsenide":  dict(a=5.653, c=5.653, Eg=1.42, me=0.067, mhh=0.45, mlh=0.082, so=0.34,  dG=0.00, dL=0.29, dX=0.48),
    "Zinc Oxide":        dict(a=3.25,  c=5.207, Eg=3.37, me=0.24,  mhh=0.59, mlh=0.59,  so=0.01,  dG=0.00, dL=1.50, dX=2.00),
    "Gallium Nitride":   dict(a=3.189, c=5.185, Eg=3.40, me=0.20,  mhh=0.80, mlh=0.80,  so=0.017, dG=0.00, dL=1.50, dX=2.00),
    "Cadmium Telluride": dict(a=6.482, c=6.482, Eg=1.50, me=0.11,  mhh=0.72, mlh=0.13,  so=0.91,  dG=0.00, dL=0.50, dX=0.90),
}
# Kane energy Ep (eV): only values I am fairly sure about. 0 = derive it from me
# (only possible when Gamma is the lowest CB minimum). Verify before citing.
EP_REF = {"Gallium Arsenide": 28.8, "Germanium": 26.3}
for _n, _v in REFERENCE.items():
    _v["Ep"] = EP_REF.get(_n, 0.0)
# Crystal-field splitting (eV), wurtzite only. 0 = not applicable / unknown.
# ZnO value is uncertain (valence-band ordering is debated); verify before citing.
DCR_REF = {"Zinc Oxide": 0.04, "Gallium Nitride": 0.010}
for _n, _v in REFERENCE.items():
    _v["dcr"] = DCR_REF.get(_n, 0.0)
STRUCTURE = {"Silicon": "Diamond", "Germanium": "Diamond", "Gallium Arsenide": "Zinc Blende",
             "Zinc Oxide": "Wurtzite", "Gallium Nitride": "Wurtzite", "Cadmium Telluride": "Zinc Blende"}
STRUCTURES = ["Diamond", "Zinc Blende", "Wurtzite"]
NEW = "New material"
HB2M = 3.81  # hbar^2/(2 m0), eV*Angstrom^2
LABELS = {-1: "L", 0: "\u0393", 1: "X"}
LABELS_WZ = {-1: "A", 0: "\u0393", 1: "M"}


def resolve(material, entered, params=None, ref=None):
    params = params or PARAMS
    ref = ref if ref is not None else REFERENCE.get(material)
    used, notes = {}, []
    for p in params:
        v = entered[p]
        if ref is None:
            used[p] = v
        elif v == 0:
            used[p] = ref[p]
        else:
            used[p] = v
            if abs(v - ref[p]) > 1e-9 and not (p in ("Ep", "dcr") and ref[p] == 0):
                notes.append(f"{p} = {v} (reference {ref[p]})")
    return used, notes


def so_mass(p, EG):
    """Split-off mass from the 3-band Kane model.
    gamma1 = (1/mhh + 1/mlh)/2 (Luttinger, [001]);
    m0/m_so = gamma1 - Ep*D / (3*EG*(EG+D)),  EG = energy of the Gamma CB minimum.
    Ep = entered/reference value, else derived from me (only if Gamma is the CBM)."""
    D = p["so"]
    info = dict(gamma1=None, gamma2=None, Ep=None, Ep_src="", mso=None, note="")
    if p["mhh"] <= 0 or p["mlh"] <= 0:
        info["note"] = "needs mhh and mlh"
        return info
    g1 = 0.5 * (1 / p["mhh"] + 1 / p["mlh"])
    g2 = 0.25 * (1 / p["mlh"] - 1 / p["mhh"])
    info.update(gamma1=g1, gamma2=g2)
    Ep, src = p.get("Ep", 0.0), "entered/reference"
    if Ep <= 0 and p["me"] > 0 and abs(EG - p["Eg"]) < 1e-9 and D >= 0:
        Ep = (1 / p["me"] - 1) * EG * (EG + D) / (EG + 2 * D / 3)   # no remote-band term
        src = "derived from me (estimate)"
    if Ep > 0 and EG > 0:
        inv = g1 - Ep * D / (3 * EG * (EG + D))
        if inv > 0:
            info.update(Ep=Ep, Ep_src=src, mso=1 / inv)
            return info
    info.update(mso=p["mlh"],
                note="Ep not available (Gamma not the CBM) -> m_so assumed = mlh")
    return info


def wz_levels(D1, Dso):
    """Quasi-cubic wurtzite valence levels at Gamma (A, B, C), A set to 0.
    D1 = crystal field, D2 = D3 = Dso/3.
    E_A = D1 + D2;  E_B,C = (D1-D2)/2 +/- sqrt(((D1-D2)/2)^2 + 2*D3^2)"""
    D2 = D3 = Dso / 3
    EA = D1 + D2
    h = (D1 - D2) / 2
    r = np.sqrt(h * h + 2 * D3 * D3)
    EB, EC = h + r, h - r
    return {"A": 0.0, "B": EB - EA, "C": EC - EA}, dict(D1=D1, D2=D2, D3=D3)


def wz_bands(p, n=2001):
    """Wurtzite along A - Gamma - M (x=-1 A, 0 Gamma, +1 M). Isotropic parabolic
    placeholder masses: A = mhh, B = C = mlh (mlh<=0 -> mhh). Only the Gamma CB valley
    is drawn; dL, dX, Ep are not used (L/X do not exist in this zone)."""
    x = np.linspace(-1, 1, n)
    kA, kM = np.pi / p["c"], 2 * np.pi / (np.sqrt(3) * p["a"])
    k = np.where(x < 0, -x * kA, x * kM)
    off, lv = wz_levels(p["dcr"], p["so"])
    ml = p["mlh"] if p["mlh"] > 0 else p["mhh"]
    mass = {"A": p["mhh"], "B": ml, "C": ml}
    cb = {"CB \u0393": p["Eg"] + HB2M / p["me"] * k**2}
    vb = {n_: off[n_] - HB2M / mass[n_] * k**2 for n_ in "ABC"}
    info = dict(gamma1=None, gamma2=None, Ep=None, Ep_src="", mso=None,
                note="wurtzite: quasi-cubic A/B/C model, isotropic placeholder masses", wz=lv)
    return x, k, cb, vb, info


def bands(p, structure="Zinc Blende", n=2001):
    if structure == "Wurtzite":
        return wz_bands(p, n)
    return cubic_bands(p, n)


def cubic_bands(p, n=2001):
    """Parabolic bands along L - Gamma - X (x=-1 L, 0 Gamma, +1 X)."""
    x = np.linspace(-1, 1, n)
    kL, kX = np.pi * np.sqrt(3) / p["a"], 2 * np.pi / p["a"]
    k = np.where(x < 0, -x * kL, x * kX)
    d = {"G": p["dG"], "L": p["dL"], "X": p["dX"]}
    dmin = min(d.values())
    E0 = {v: p["Eg"] + d[v] - dmin for v in d}      # lowest CB minimum = Eg
    cb, vb = {}, {}
    if p["me"] > 0:
        c = HB2M / p["me"]
        cb["CB \u0393"] = E0["G"] + c * k**2
        cb["CB L"] = np.where(x < 0, E0["L"] + c * (k - kL) ** 2, np.nan)
        cb["CB X"] = np.where(x > 0, E0["X"] + c * (k - kX) ** 2, np.nan)
    if p["mhh"] > 0:
        vb["HH"] = -HB2M / p["mhh"] * k**2
    if p["mlh"] > 0:
        vb["LH"] = -HB2M / p["mlh"] * k**2
    info = so_mass(p, E0["G"])
    if info["mso"]:
        vb["SO"] = -p["so"] - HB2M / info["mso"] * k**2
    return x, k, cb, vb, info


def _label(x, labels=LABELS):
    return labels[min(labels, key=lambda t: abs(t - x))]


def _mass(x, k, y, side=+1, w=0.03):
    """Effective mass (m0) from a parabola fit around Gamma, on one side."""
    m = (x * side > 0) & (abs(x) <= w)
    if m.sum() < 5:
        return None
    c2 = np.polyfit(k[m], y[m], 2)[0]
    return HB2M / abs(c2) if c2 != 0 else None


def analyze(x, k, cb, vb, labels=LABELS):
    """All numbers below come from the band arrays."""
    if not cb or not vb:
        return None
    ivb = {n: np.nanargmax(y) for n, y in vb.items()}
    nvb = max(ivb, key=lambda n: vb[n][ivb[n]])
    vbm, xv = vb[nvb][ivb[nvb]], x[ivb[nvb]]
    icb = {n: np.nanargmin(y) for n, y in cb.items()}
    ncb = min(icb, key=lambda n: cb[n][icb[n]])
    cbm, xc = cb[ncb][icb[ncb]], x[icb[ncb]]
    g0 = len(x) // 2                                  # index of Gamma
    r = dict(
        gap=cbm - vbm, vbm=(vbm, _label(xv, labels)), cbm=(cbm, _label(xc, labels)),
        type="Direct" if abs(xv - xc) < 0.02 else "Indirect",
        valley={n: cb[n][icb[n]] - cbm for n in cb},   # valley minima above CBM
        mass={}, trans={},
    )
    r["direct_gap_G"] = cb["CB \u0393"][g0] - vbm
    for n, y in vb.items():
        r["mass"][n] = _mass(x, k, y)
        r["trans"][n] = cb["CB \u0393"][g0] - y[g0]   # vertical at Gamma
    r["mass"]["CB \u0393"] = _mass(x, k, cb["CB \u0393"])
    if all(n in vb for n in "ABC"):
        r["split"] = {"A-B": vb["A"][g0] - vb["B"][g0], "B-C": vb["B"][g0] - vb["C"][g0],
                      "A-C": vb["A"][g0] - vb["C"][g0]}
    if "SO" in vb and "HH" in vb:
        r["delta_so"] = max(vb["HH"][g0], vb["LH"][g0]) - vb["SO"][g0]
    return r



# ---------------------------------------------------------------------------
# Full-band empirical pseudopotential model (EPM) for diamond / zinc-blende.
# Form factors in Ry (Cohen & Bergstresser, Phys. Rev. 141, 789 (1966)).
# Values below are from memory: verify against the paper before citing.
# No spin-orbit in this local EPM (Delta_so is not an output here).
# ---------------------------------------------------------------------------
from functools import lru_cache

RY = 13.6057
EPM_PARAMS = ["a", "V3S", "V8S", "V11S", "V3A", "V4A", "V11A"]
EPM_REF = {
    "Silicon":           dict(a=5.431, V3S=-0.21, V8S=0.04, V11S=0.08, V3A=0.0,  V4A=0.0,  V11A=0.0),
    "Germanium":         dict(a=5.658, V3S=-0.23, V8S=0.01, V11S=0.06, V3A=0.0,  V4A=0.0,  V11A=0.0),
    "Gallium Arsenide":  dict(a=5.653, V3S=-0.23, V8S=0.01, V11S=0.06, V3A=0.07, V4A=0.05, V11A=0.01),
    "Cadmium Telluride": dict(a=6.482, V3S=-0.20, V8S=0.00, V11S=0.04, V3A=0.15, V4A=0.09, V11A=0.04),  # uncertain
}
HS = {"L": (.5, .5, .5), "\u0393": (0, 0, 0), "X": (1, 0, 0), "U": (1, .25, .25), "K": (.75, .75, 0)}
SEGS = [("L", "\u0393", "\u039b"), ("\u0393", "X", "\u0394"), ("X", "U", "X-U"), ("K", "\u0393", "\u03a3")]
NB = 10   # bands computed (4 valence + 6 conduction)


def _gvecs(cut=12):
    r = range(-4, 5)
    return np.array([(h, k, l) for h in r for k in r for l in r
                     if h % 2 == k % 2 == l % 2 and h*h + k*k + l*l <= cut], float)


def epm_path(npts=400):
    lens = [np.linalg.norm(np.array(HS[b]) - np.array(HS[a])) for a, b, _ in SEGS]
    ks, xs, seg, hs_idx, x0, n_tot = [], [], [], {}, 0.0, 0
    ticks = []
    for (a, b, name), ln in zip(SEGS, lens):
        n = max(int(round(npts * ln / sum(lens))), 8)
        t = np.linspace(0, 1, n)
        ks.append(np.array(HS[a])[None] + t[:, None] * (np.array(HS[b]) - np.array(HS[a]))[None])
        xs.append(x0 + t * ln)
        seg += [name] * n
        for lab, i in ((a, n_tot), (b, n_tot + n - 1)):
            hs_idx.setdefault(lab, i)
        ticks.append((x0, a if a != "U" else "U,K"))
        x0 += ln
        n_tot += n
    ticks = [(0.0, "L"), (lens[0], "\u0393"), (lens[0] + lens[1], "X"),
             (lens[0] + lens[1] + lens[2], "U,K"), (x0, "\u0393")]
    return np.vstack(ks), np.concatenate(xs), np.array(seg), hs_idx, ticks


@lru_cache(maxsize=32)
def compute_epm(a, v3s, v8s, v11s, v3a, v4a, v11a, npts=400):
    G = _gvecs()
    dG = G[:, None, :] - G[None, :, :]
    s2 = np.rint((dG ** 2).sum(-1)).astype(int)
    VS, VA = np.zeros(80), np.zeros(80)
    VS[3], VS[8], VS[11] = v3s, v8s, v11s
    VA[3], VA[4], VA[11] = v3a, v4a, v11a
    ph = np.pi * dG.sum(-1) / 4
    V = (VS[s2] * np.cos(ph) + 1j * VA[s2] * np.sin(ph)) * RY
    V[s2 == 0] = 0
    ks, x, seg, hs_idx, ticks = epm_path(npts)
    pref = HB2M * (2 * np.pi / a) ** 2
    E = np.empty((len(ks), NB))
    for i, k in enumerate(ks):
        H = V + np.diag(pref * ((k[None] + G) ** 2).sum(-1))
        E[i] = np.linalg.eigvalsh(H)[:NB]
    E -= E[:, 3].max()                          # VBM = 0
    return ks, x, seg, hs_idx, ticks, E


def _fit_mass(s, y, a):
    c = np.polyfit(s * 2 * np.pi / a, y, 2)[0]
    return HB2M / abs(c) if c else None


def analyze_epm(a, ks, x, seg, hs_idx, E):
    vb, cb = E[:, :4], E[:, 4:]
    ivb, icb = int(np.argmax(E[:, 3])), int(np.argmin(E[:, 4]))
    gap = E[icb, 4] - E[ivb, 3]
    vert = E[:, 4] - E[:, 3]
    iv = int(np.argmin(vert))

    def where(i):
        for n, j in hs_idx.items():
            if np.linalg.norm(ks[i] - ks[j]) < 0.03:
                return n
        k = ks[i]
        return f"{seg[i]} ({k[0]:.2f}, {k[1]:.2f}, {k[2]:.2f})"
    r = dict(gap=gap, vbm=E[ivb, 3], cbm=E[icb, 4], vbm_at=where(ivb), cbm_at=where(icb),
             type="Direct" if vert[iv] - gap < 0.01 else "Indirect",
             vert=(vert[iv], where(iv)),
             direct={n: vert[hs_idx[n]] for n in ("\u0393", "L", "X")},
             valley={n: E[hs_idx[n], 4] - E[icb, 4] for n in ("\u0393", "L", "X")},
             vb_width=E[hs_idx["\u0393"], 3] - E[hs_idx["\u0393"], 0], mass={})
    d = (seg == "\u0394") & (ks[:, 0] <= 0.06)
    for b, nme in ((3, "VB1"), (2, "VB2"), (1, "VB3")):
        if d.sum() >= 5:
            r["mass"][f"{nme} (\u0393\u2192X)"] = _fit_mass(ks[d, 0], E[d, b], a)
    w = (seg == seg[icb]) & (abs(x - x[icb]) <= 0.07)
    if w.sum() >= 5:
        r["mass"]["CB min (along path)"] = _fit_mass(x[w] - x[icb], E[w, 4], a)
    return r


def plot_fig(go, x, curves, ticks, ylim=None, lines=None, extra=None):
    """curves: {name: (y, colour, customdata)}; hover shows band, E and k."""
    fig = go.Figure()
    for name, (y, col, cd) in curves.items():
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=name, line=dict(color=col, width=2),
            customdata=cd, hovertemplate=name + "<br>E = %{y:.3f} eV<br>k = %{customdata}<extra></extra>"))
    for t, _ in ticks:
        fig.add_vline(x=t, line_width=1, line_color="#2b4a9b")
    for yv, txt in (lines or []):
        fig.add_hline(y=yv, line_color="red", line_width=2, annotation_text=txt, annotation_position="top right")
    fig.update_xaxes(tickvals=[t for t, _ in ticks], ticktext=[l for _, l in ticks], showgrid=False,
                     title="Wave vector k")
    fig.update_yaxes(title="E (eV)", range=ylim, zeroline=False)
    fig.update_layout(height=620, hovermode="closest", dragmode="zoom", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=-0.15))
    return fig


PLOT_CONFIG = {"scrollZoom": True, "displaylogo": False,
               "modeBarButtonsToAdd": ["zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"]}


def main():
    import streamlit as st
    import plotly.graph_objects as go
    st.title("E-k diagram (Theoretical mode)")
    material = st.selectbox("Material", list(REFERENCE) + [NEW])
    if material in STRUCTURE:
        structure = STRUCTURE[material]
        st.caption(f"Structure: {structure}")
    else:
        structure = st.selectbox("Structure", STRUCTURES, index=1)
    wz = structure == "Wurtzite"
    if wz:
        mode = "Simple"
        st.info("Full-band (EPM) parameters for wurtzite are not available here; using the simple "
                "quasi-cubic model near \u0393. For full bands use DFT mode.")
    else:
        mode = st.radio("Model", ["Full band (EPM)", "Simple (parabolic)"], horizontal=True).split()[0]
    params = EPM_PARAMS if mode == "Full" else PARAMS
    cols = st.columns(6)
    entered = {p: cols[i % 6].number_input(p, value=0.0, format="%.4f", key=f"{material}_{mode}_{p}")
               for i, p in enumerate(params)}
    ref = EPM_REF.get(material) if mode == "Full" else None
    used, notes = resolve(material, entered, params, ref if mode == "Full" else None)
    if notes:
        st.toast("Not ideal value: " + "; ".join(notes))
    st.caption("Scroll to zoom, drag to pan, double-click to reset; hover a band for E and k.")

    if mode == "Full":
        if used["a"] <= 0 or not any(used[p] for p in EPM_PARAMS[1:]):
            st.info("Enter a and the pseudopotential form factors (Ry) to draw the diagram.")
            return
        ks, x, seg, hs_idx, ticks, E = compute_epm(*(round(used[p], 6) for p in EPM_PARAMS))
        r = analyze_epm(used["a"], ks, x, seg, hs_idx, E)
        cd = [f"({k[0]:.2f}, {k[1]:.2f}, {k[2]:.2f}) 2\u03c0/a" for k in ks]
        curves = {f"Band {b + 1}": (E[:, b], "#1a7f37" if b < 4 else "#c2410c", cd) for b in range(NB)}
        fig = plot_fig(go, x, curves, ticks, ylim=[E[:, 0].min() - 0.5, r["cbm"] + 5.5],
                       lines=[(r["vbm"], ""), (r["cbm"], "Energy gap")])
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
        st.subheader("Results (computed from the plotted bands)")
        st.write(f"**Band gap:** {r['gap']:.3f} eV | **Type:** {r['type']}")
        st.write(f"**VBM at** {r['vbm_at']} | **CBM at** {r['cbm_at']}")
        st.write(f"**Minimum vertical gap:** {r['vert'][0]:.3f} eV at {r['vert'][1]}")
        st.write("**Direct gap at:** " + ", ".join(f"{n}: {v:.3f} eV" for n, v in r["direct"].items()))
        st.write("**CB valley energy above CBM:** " + ", ".join(f"{n}: {v:.3f} eV" for n, v in r["valley"].items()))
        st.write(f"**Valence bandwidth (\u0393):** {r['vb_width']:.2f} eV")
        st.write("**Effective masses (m0):** " + ", ".join(f"{n}: {v:.3f}" for n, v in r["mass"].items() if v))
        st.caption("Local EPM has no spin-orbit, so \u0394so is not an output here (use Simple mode).")
        return

    need = used["a"] <= 0 or used["Eg"] <= 0 or used["me"] <= 0 or used["mhh"] <= 0 or (wz and used["c"] <= 0)
    if need:
        st.info("Enter a, Eg, me, mhh (and c for wurtzite; mlh, so for more bands) to draw the diagram.")
        return
    x, k, cb, vb, info = bands(used, structure)
    labels = LABELS_WZ if wz else LABELS
    ticks = [(-1, labels[-1]), (0, labels[0]), (1, labels[1])]
    curves = {n: (y, "#c2410c" if n in cb else "#1a7f37", [f"{kk:.3f} 1/\u00c5" for kk in k])
              for n, y in {**cb, **vb}.items()}
    st.plotly_chart(plot_fig(go, x, curves, ticks), use_container_width=True, config=PLOT_CONFIG)
    r = analyze(x, k, cb, vb, labels)
    if r is None:
        return
    st.subheader("Results (computed from the plotted bands)")
    st.write(f"**Band gap:** {r['gap']:.3f} eV | **Type:** {r['type']}")
    st.write(f"**VBM:** {r['vbm'][0]:.3f} eV at {r['vbm'][1]} | **CBM:** {r['cbm'][0]:.3f} eV at {r['cbm'][1]}")
    st.write(f"**Direct gap at \u0393:** {r['direct_gap_G']:.3f} eV")
    st.write("**Valley separation above CBM (eV):** " + ", ".join(f"{n}: {v:.3f}" for n, v in r["valley"].items()))
    st.write("**Effective masses (m0):** " + ", ".join(f"{n}: {v:.3f}" for n, v in r["mass"].items() if v))
    if info["gamma1"]:
        so_txt = f"m_so = {info['mso']:.3f}" if info["mso"] else "m_so n/a"
        ep_txt = f", Ep = {info['Ep']:.1f} eV ({info['Ep_src']})" if info["Ep"] else ""
        st.write(f"**Luttinger (from mhh, mlh):** \u03b31 = {info['gamma1']:.2f}, \u03b32 = {info['gamma2']:.2f}{ep_txt}, {so_txt}")
    if info["note"]:
        st.caption(info["note"])
    if "split" in r:
        st.write("**Valence splittings at \u0393 (eV):** " + ", ".join(f"{n}: {v:.4f}" for n, v in r["split"].items()))
    if "delta_so" in r:
        st.write(f"**\u0394so:** {r['delta_so']:.3f} eV")
    st.write("**Vertical transitions at \u0393 (eV):** " + ", ".join(f"CB\u2192{n}: {v:.3f}" for n, v in r["trans"].items()))


if __name__ == "__main__":
    main()
