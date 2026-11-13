"""E-k Theoretical mode: parameters -> bands -> calculations FROM the bands.

Flow
1. Reference values (your table) are stored here, fixed, hidden from the input fields.
2. ALL input boxes start BLANK for every material. Nothing is filled automatically.
3. Press "Generate E-k diagram":
   - nothing entered      -> red "Enter parameters first" message, no diagram
   - some fields missing  -> red "Enter value first" message listing them, no diagram
   - wrong value / sign   -> red error under the field + error box, no diagram
   - all valid            -> diagram + results use ONLY the values you typed
4. For a known material, if a typed value differs from the table a short
   "Not ideal value" toast appears; the diagram still follows YOUR value.
5. Both models (full-band EPM and simple parabolic) run when their required fields
   are filled; the conclusion is built from the band arrays.

Run:  streamlit run ek_parameters_module.py
Needs a recent Streamlit (number_input(value=None)):  pip install -U streamlit
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
# Kane energy Ep (eV): only values I am fairly sure about. Verify before citing.
EP_REF = {"Gallium Arsenide": 28.8, "Germanium": 26.3}
for _n, _v in REFERENCE.items():
    _v["Ep"] = EP_REF.get(_n, 0.0)
# Crystal-field splitting (eV), wurtzite only. ZnO value is uncertain; verify before citing.
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

# ---------------------------------------------------------------------------
# Input limits: (label, min, max, min_inclusive)
# min_inclusive False -> value must be strictly greater than min (0 not allowed)
# ---------------------------------------------------------------------------
LIMITS = {
    "a":    ("Lattice constant a (\u00c5)", 0.0, 15.0, False),
    "c":    ("Lattice constant c (\u00c5)", 0.0, 15.0, False),
    "Eg":   ("Band gap Eg (eV)", 0.0, 12.0, False),
    "me":   ("Electron mass me (m0)", 0.0, 5.0, False),
    "mhh":  ("Heavy-hole mass mhh (m0)", 0.0, 10.0, False),
    "mlh":  ("Light-hole mass mlh (m0)", 0.0, 10.0, False),
    "so":   ("Spin-orbit split \u0394so (eV)", 0.0, 2.0, True),
    "dG":   ("\u0393 valley offset dG (eV)", 0.0, 6.0, True),
    "dL":   ("L valley offset dL (eV)", 0.0, 6.0, True),
    "dX":   ("X valley offset dX (eV)", 0.0, 6.0, True),
    "Ep":   ("Kane energy Ep (eV)", 0.0, 60.0, False),
    "dcr":  ("Crystal-field \u0394cr (eV)", -0.5, 0.5, True),
    "V3S":  ("V3S (Ry)", -1.0, 1.0, True),
    "V8S":  ("V8S (Ry)", -1.0, 1.0, True),
    "V11S": ("V11S (Ry)", -1.0, 1.0, True),
    "V3A":  ("V3A (Ry)", -1.0, 1.0, True),
    "V4A":  ("V4A (Ry)", -1.0, 1.0, True),
    "V11A": ("V11A (Ry)", -1.0, 1.0, True),
}


def check(p, v):
    """Return a red error message for a wrong value, else None. Blank (None) is not an error."""
    if v is None:
        return None
    label, lo, hi, incl = LIMITS[p]
    short = label.split(" (")[0]
    if not np.isfinite(v):
        return f"{short}: invalid number"
    if lo == 0.0 and v < 0:
        return f"{short} can never be negative"
    if lo == 0.0 and not incl and v == 0:
        return f"{short} can never be zero"
    if v < lo or v > hi or (not incl and v == lo):
        return f"{short} must be between {lo:g} and {hi:g}"
    return None


def so_mass(p, EG):
    """Split-off mass from the 3-band Kane model.
    gamma1 = (1/mhh + 1/mlh)/2 (Luttinger, [001]);
    m0/m_so = gamma1 - Ep*D / (3*EG*(EG+D)),  EG = energy of the Gamma CB minimum.
    Ep = entered value, else derived from me (only if Gamma is the CBM)."""
    D = p["so"]
    info = dict(gamma1=None, gamma2=None, Ep=None, Ep_src="", mso=None, note="")
    if p["mhh"] <= 0 or p["mlh"] <= 0:
        info["note"] = "needs mhh and mlh"
        return info
    g1 = 0.5 * (1 / p["mhh"] + 1 / p["mlh"])
    g2 = 0.25 * (1 / p["mlh"] - 1 / p["mhh"])
    info.update(gamma1=g1, gamma2=g2)
    Ep, src = p.get("Ep", 0.0), "entered"
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
    for (a, b, name), ln in zip(SEGS, lens):
        n = max(int(round(npts * ln / sum(lens))), 8)
        t = np.linspace(0, 1, n)
        ks.append(np.array(HS[a])[None] + t[:, None] * (np.array(HS[b]) - np.array(HS[a]))[None])
        xs.append(x0 + t * ln)
        seg += [name] * n
        for lab, i in ((a, n_tot), (b, n_tot + n - 1)):
            hs_idx.setdefault(lab, i)
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

EPM_FIELDS = ["V3S", "V8S", "V11S", "V3A", "V4A", "V11A"]


def _f(v, nd=3):
    return "\u2014" if v is None else f"{v:.{nd}f}"


def _masses(d):
    t = ", ".join(f"{n}: {v:.3f}" for n, v in d.items() if v)
    return t or "\u2014"


def conclusion(material, epm, simp):
    """Build the table and the verdict from the two analysis dicts (either may be None)."""
    cols = {}
    if epm:
        r = epm
        cols["EPM (full band)"] = [
            r["type"], _f(r["gap"]), r["vbm_at"], r["cbm_at"], _f(r["direct"]["\u0393"]),
            f"{r['vert'][0]:.3f} at {r['vert'][1]}",
            _f(r["valley"]["\u0393"]), _f(r["valley"]["L"]), _f(r["valley"]["X"]),
            "n/a (no spin-orbit)", _masses(r["mass"])]
    if simp:
        r = simp
        v = r["valley"]
        so = (f"{r['delta_so']:.3f}" if "delta_so" in r else
              ("A-B {:.4f}, B-C {:.4f}".format(r["split"]["A-B"], r["split"]["B-C"]) if "split" in r else "\u2014"))
        cols["Simple (parabolic)"] = [
            r["type"], _f(r["gap"]), r["vbm"][1], r["cbm"][1], _f(r["direct_gap_G"]), "\u2014",
            _f(v.get("CB \u0393")), _f(v.get("CB L")), _f(v.get("CB X")), so, _masses(r["mass"])]
    ref = REFERENCE.get(material)
    if ref:
        cols["Table (reference)"] = ["\u2014", _f(ref["Eg"]), "\u2014", "\u2014", "\u2014", "\u2014",
                                     "\u2014", "\u2014", "\u2014", _f(ref["so"], 3), "\u2014"]
    rows = ["Gap type", "Band gap (eV)", "VBM at", "CBM at", "Direct gap at \u0393 (eV)",
            "Min. vertical gap (eV)", "\u0393 valley above CBM (eV)", "L valley above CBM (eV)",
            "X valley above CBM (eV)", "\u0394so / valence splitting (eV)", "Effective masses (m0)"]
    verdict = []
    types = {n: d["type"] for n, d in (("EPM", epm), ("Simple", simp)) if d}
    name = material if material != NEW else "This material"
    if len(set(types.values())) == 1:
        t = next(iter(types.values()))
        if epm:
            loc_v, loc_c, direct_min = epm["vbm_at"], epm["cbm_at"], epm["vert"][0]
            gap = epm["gap"]
        else:
            loc_v, loc_c, direct_min = simp["vbm"][1], simp["cbm"][1], simp["direct_gap_G"]
            gap = simp["gap"]
        agree = " (both models agree)" if len(types) == 2 else f" ({next(iter(types))} model only)"
        if t == "Direct":
            verdict.append(f"**{name}: DIRECT band gap{agree}.** VBM and CBM are both at {loc_v}; "
                           f"the gap is {gap:.3f} eV and optical transitions need no phonon.")
        else:
            verdict.append(f"**{name}: INDIRECT band gap{agree}.** VBM at {loc_v}, CBM at {loc_c}; "
                           f"gap {gap:.3f} eV. The lowest direct transition is {direct_min:.3f} eV, "
                           f"i.e. {direct_min - gap:.3f} eV above the gap, so absorption near the edge is phonon-assisted.")
    else:
        verdict.append(f"**Models disagree on the gap type** ({', '.join(f'{k}: {v}' for k, v in types.items())}). "
                       "Check the parameters; the full-band EPM is the more reliable of the two for valley ordering.")
    if epm and simp:
        verdict.append(f"EPM vs simple gap: {epm['gap']:.3f} vs {simp['gap']:.3f} eV "
                       f"(difference {abs(epm['gap'] - simp['gap']):.3f} eV).")
    if ref and epm:
        verdict.append(f"EPM gap differs from the table Eg ({ref['Eg']:.2f} eV) by {epm['gap'] - ref['Eg']:+.3f} eV; "
                       "the EPM is fitted to optical transitions, not to the fundamental gap.")
    return rows, cols, verdict


def main():
    import streamlit as st
    import plotly.graph_objects as go
    import pandas as pd
    st.title("E-k diagram and direct / indirect band-gap analysis")
    material = st.selectbox("Material", list(REFERENCE) + [NEW])
    if material in STRUCTURE:
        structure = STRUCTURE[material]
        st.caption(f"Structure: {structure}")
    else:
        structure = st.selectbox("Structure", STRUCTURES, index=1)
    wz = structure == "Wurtzite"
    if wz:
        st.info("Full-band (EPM) parameters for wurtzite are not available here, so only the simple "
                "quasi-cubic model near \u0393 is run. For full bands use DFT mode.")

    entered, errs = {}, {}

    def field(col, p):
        v = col.number_input(p, value=None, format="%.4f", key=f"{material}_{p}",
                             help=LIMITS[p][0])
        msg = check(p, v)
        if msg:
            errs[p] = msg
            col.markdown(f":red[{msg}]")
        entered[p] = v

    simple_fields = (["a", "c", "Eg", "me", "mhh", "mlh", "so", "dcr"] if wz else
                     ["a", "Eg", "me", "mhh", "mlh", "so", "dG", "dL", "dX", "Ep"])
    st.markdown("**Simple-model parameters** (a is shared with the EPM)")
    cols = st.columns(5)
    for i, p in enumerate(simple_fields):
        field(cols[i % 5], p)
    if not wz:
        st.markdown("**EPM form factors (Ry)**")
        cols = st.columns(6)
        for i, p in enumerate(EPM_FIELDS):
            field(cols[i], p)

    # values used for calculation = ONLY what the user typed (blank optional -> 0)
    used_s = {p: (entered.get(p) if entered.get(p) is not None else 0.0) for p in PARAMS}
    if not wz:
        used_s["c"] = used_s["a"]
    used_e = {p: (entered.get(p) if entered.get(p) is not None else 0.0) for p in EPM_PARAMS}

    # "Not ideal value" toast: compare typed values with the reference tables
    notes = []
    ref = REFERENCE.get(material)
    if ref:
        for p in simple_fields:
            v = entered.get(p)
            if v is not None and abs(v - ref[p]) > 1e-9 and not (p in ("Ep", "dcr") and ref[p] == 0):
                notes.append(f"{p} = {v} (reference {ref[p]})")
    eref = EPM_REF.get(material)
    if eref and not wz:
        for p in EPM_PARAMS:
            v = entered.get(p)
            if v is not None and abs(v - eref[p]) > 1e-9:
                notes.append(f"{p} = {v} (reference {eref[p]})")
    if notes and not errs:
        st.toast("Not ideal value: " + "; ".join(dict.fromkeys(notes)))

    sig = (material, structure)
    if st.session_state.get("gen_sig") != sig:
        st.session_state["gen_sig"] = None            # material changed -> back to empty
    if st.button("Generate E-k diagram", type="primary"):
        st.session_state["gen_sig"] = sig
    if st.session_state.get("gen_sig") != sig:
        st.info("Enter the parameters and press 'Generate E-k diagram'. "
                "No diagram or results are shown until a diagram is generated.")
        return

    # ---- validation: wrong value / sign -> red error, no diagram ----
    if errs:
        st.error("Wrong value(s) entered. Fix the red fields first:\n\n" +
                 "\n".join(f"- {m}" for m in errs.values()))
        return

    simple_req = [p for p in simple_fields if p != "Ep"]
    epm_req = ["a"] + EPM_FIELDS
    miss_s = [p for p in simple_req if entered.get(p) is None]
    miss_e = [p for p in epm_req if entered.get(p) is None]
    simp_ok = not miss_s
    epm_ok = (not wz) and not miss_e
    if not epm_ok and not simp_ok:
        if all(entered.get(p) is None for p in entered):
            st.error("Enter parameters first. Enter values, then press 'Generate E-k diagram'.")
        else:
            st.error("Enter value first. Missing for simple model: " + ", ".join(miss_s) +
                     ("" if wz else "  |  missing for EPM: " + ", ".join(miss_e)))
        return
    st.caption("Scroll to zoom, drag to pan, double-click to reset; hover a band for E and k.")

    figs, epm_r, simp_r = {}, None, None
    if epm_ok:
        ks, x, seg, hs_idx, ticks, E = compute_epm(*(round(used_e[p], 6) for p in EPM_PARAMS))
        epm_r = analyze_epm(used_e["a"], ks, x, seg, hs_idx, E)
        cd = [f"({k[0]:.2f}, {k[1]:.2f}, {k[2]:.2f}) 2\u03c0/a" for k in ks]
        curves = {f"Band {b + 1}": (E[:, b], "#1a7f37" if b < 4 else "#c2410c", cd) for b in range(NB)}
        figs["Full band (EPM)"] = plot_fig(go, x, curves, ticks,
                                           ylim=[E[:, 0].min() - 0.5, epm_r["cbm"] + 5.5],
                                           lines=[(epm_r["vbm"], ""), (epm_r["cbm"], "Energy gap")])
    if simp_ok:
        x, k, cb, vb, info = bands(used_s, structure)
        labels = LABELS_WZ if wz else LABELS
        simp_r = analyze(x, k, cb, vb, labels)
        curves = {n: (y, "#c2410c" if n in cb else "#1a7f37", [f"{kk:.3f} 1/\u00c5" for kk in k])
                  for n, y in {**cb, **vb}.items()}
        figs["Simple (parabolic)"] = plot_fig(go, x, curves, [(-1, labels[-1]), (0, labels[0]), (1, labels[1])])
    for tab, (name, fig) in zip(st.tabs(list(figs)), figs.items()):
        with tab:
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    st.subheader("Conclusion")
    rows, cols_, verdict = conclusion(material, epm_r, simp_r)
    for v in verdict:
        st.markdown(v)
    st.table(pd.DataFrame(cols_, index=rows))
    notes_txt = []
    if epm_r:
        notes_txt.append("EPM: local pseudopotential, no spin-orbit; all numbers are read from the plotted bands.")
    if simp_r:
        notes_txt.append("Simple: parabolic bands near \u0393 with the entered masses; L and X valleys use the same "
                         "m_e and are indicative only. Eg is an input here, so its agreement with the table is by construction.")
    st.caption(" ".join(notes_txt))


if __name__ == "__main__":
    main()
