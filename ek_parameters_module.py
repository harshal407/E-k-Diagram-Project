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


def resolve(material, entered):
    ref = REFERENCE.get(material)
    used, notes = {}, []
    for p in PARAMS:
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


def main():
    import streamlit as st
    import matplotlib.pyplot as plt
    st.title("E-k diagram (Theoretical mode)")
    material = st.selectbox("Material", list(REFERENCE) + [NEW])
    if material in STRUCTURE:
        structure = STRUCTURE[material]
        st.caption(f"Structure: {structure}")
    else:
        structure = st.selectbox("Structure", STRUCTURES, index=1)
    wz = structure == "Wurtzite"
    cols = st.columns(6)
    entered = {p: cols[i % 6].number_input(p, value=0.0, format="%.4f", key=f"{material}_{p}")
               for i, p in enumerate(PARAMS)}
    used, notes = resolve(material, entered)
    if notes:
        st.toast("Not ideal value: " + "; ".join(notes))
    need = used["a"] <= 0 or used["Eg"] <= 0 or used["me"] <= 0 or used["mhh"] <= 0 or (wz and used["c"] <= 0)
    if need:
        st.info("Enter a, Eg, me, mhh (and c for wurtzite; mlh, so for more bands) to draw the diagram.")
        return
    x, k, cb, vb, info = bands(used, structure)
    labels = LABELS_WZ if wz else LABELS
    fig, ax = plt.subplots(figsize=(6, 5))
    for n, y in {**cb, **vb}.items():
        ax.plot(x, y, label=n)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xticks([-1, 0, 1], [labels[-1], labels[0], labels[1]])
    ax.set_ylabel("E (eV)")
    ax.legend(fontsize=7)
    st.pyplot(fig)
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
