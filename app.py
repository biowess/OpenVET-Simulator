import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import json
import hashlib
import datetime
import pandas as pd
import io

# ==========================================
# 0. CONFIGURATION & METADATA
# ==========================================
APP_VERSION = "1.0.0"
MODEL_VERSION = "1.0.0"
MODEL_VERSION_RESEARCH = "1.0.0"
SCHEMA_VERSION = "1.0"

# ==========================================
# 1A. SIMULATION CORE — EDUCATIONAL MODE
#     (Hill-gated heuristic, 8 state vars)
#     Monograph Chapter 2 — reduced-order
# ==========================================
def coagulation_model_heuristic(t, y, p):
    """Educational mode: 8-state Hill-function heuristic ODE system."""
    TF, IIa, Fg, F, Fp, Plg, Pln, Pa = y

    if p.get('assay') in ['INTEM', 'HEPTEM']:
        k_TF_act = 0.005
        K_thr = 85.0
        K_thr_p = 95.0
    else:
        k_TF_act = p['k_TF_act']
        K_thr = p['K_thr']
        K_thr_p = p['K_thr_p']

    dTF_dt = 0
    dIIa_dt = k_TF_act * TF - p['k_AT'] * IIa

    hill_fib = IIa**p['n_hill'] / (K_thr**p['n_hill'] + IIa**p['n_hill'])
    rate_Fg = p['alpha1_max'] * hill_fib

    dFg_dt = -rate_Fg * Fg
    dF_dt = rate_Fg * Fg - p['alpha2'] * F

    if p.get('assay') == 'APTEM':
        k_deg_eff = 1e-6
    else:
        k_deg_eff = p['k_deg']

    dFp_dt = p['alpha2'] * F - k_deg_eff * Pln * Fp

    dPlg_dt = -p['k_plasmin_act'] * Plg
    dPln_dt = p['k_plasmin_act'] * Plg - p['k_a2AP'] * Pln

    if p.get('assay') == 'FIBTEM':
        dPa_dt = 0.0
    else:
        hill_p = IIa**p['n_hill'] / (K_thr_p**p['n_hill'] + IIa**p['n_hill'])
        dPa_dt = p['k_cat_P'] * hill_p * (1 - Pa)

    return [dTF_dt, dIIa_dt, dFg_dt, dF_dt, dFp_dt, dPlg_dt, dPln_dt, dPa_dt]


# ==========================================
# 1B. SIMULATION CORE — RESEARCH MODE
#     (Direct Michaelis-Menten, 11 state vars)
#     Monograph Chapter 2 — exact equations
#     States: [TF, Xa, IIa, Va, Prot, Fg, F, Fp, Plg, Pln, Pa]
#       0: TF  — Tissue Factor (initiator)
#       1: Xa  — Factor Xa
#       2: IIa — Thrombin
#       3: Va  — Factor Va (prothrombinase cofactor)
#       4: Prot — Prothrombinase complex [Xa·Va]
#       5: Fg  — Fibrinogen
#       6: F   — Fibrin monomer
#       7: Fp  — Polymerized fibrin
#       8: Plg — Plasminogen
#       9: Pln — Plasmin
#      10: Pa  — Activated platelets (fraction)
# ==========================================
def coagulation_model_exact(t, y, p):
    """Research mode: 11-state Michaelis-Menten ODE system from Hockin-Mann."""
    TF, Xa, IIa, Va, Prot, Fg, F, Fp, Plg, Pln, Pa = y

    # --- Tissue Factor: constant (exogenous addition) ---
    dTF_dt = 0.0

    # --- Factor Xa generation: TF-driven, with inhibition ---
    # TF·VIIa activates factor X → Xa
    dXa_dt = p['k_TF_Xa'] * TF - p['k_Xa_inh'] * Xa

    # --- Prothrombinase complex assembly: Xa + Va → Xa·Va ---
    # Simple bimolecular assembly with dissociation
    dVa_dt = p['k_IIa_Va'] * IIa * (1.0 - Va) - p['k_Va_inh'] * Va
    dProt_dt = p['k_prot_assem'] * Xa * Va - p['k_prot_inh'] * Prot

    # --- Thrombin generation: Michaelis-Menten kinetics ---
    # Prothrombinase-catalyzed conversion: k_cat_prot = 63.5 s^-1 (Hockin 2002)
    # Direct Xa-membrane pathway: k_cat_Xa = 2.3e-3 s^-1
    # AT-III inhibition: k_AT = 7.1e3 M^-1s^-1
    # Prothrombin is consumed as thrombin is generated (mass conservation):
    #   [II](t) = [II]_0 - [IIa](t), ensuring substrate depletion
    II_remaining = max(0.0, p['II_0'] - IIa)
    mm_prot = II_remaining / (II_remaining + p['Km_prot'])
    mm_xa = II_remaining / (II_remaining + p['Km_Xa'])

    dIIa_dt = (p['k_cat_prot'] * Prot * mm_prot +
               p['k_cat_Xa'] * Xa * mm_xa -
               p['k_AT'] * IIa)

    # --- Fibrinogen → Fibrin monomer → Polymerized fibrin ---
    # Thrombin-catalyzed fibrinogen cleavage (Ratto et al. 2021)
    alpha1_rate = p['alpha1'] * IIa
    dFg_dt = -alpha1_rate * Fg
    dF_dt = alpha1_rate * Fg - p['alpha2'] * F

    # --- Polymerized fibrin: formation minus plasmin degradation ---
    if p.get('assay') == 'APTEM':
        k_deg_eff = 1e-6  # Aprotinin suppresses plasmin → negligible degradation
    else:
        k_deg_eff = p['k_deg']

    dFp_dt = p['alpha2'] * F - k_deg_eff * Pln * Fp

    # --- Plasminogen → Plasmin (with α2-AP inhibition) ---
    dPlg_dt = -p['k_plasmin_act'] * Plg
    dPln_dt = p['k_plasmin_act'] * Plg - p['k_a2AP'] * Pln

    # --- Platelet activation: Michaelis-Menten kinetics ---
    # k_cat = 0.05 s^-1, K_M = 0.63 nM (Fogelson 2012 / Luan 2007)
    # Pa is fraction [0, 1] of total activatable platelets
    if p.get('assay') == 'FIBTEM':
        dPa_dt = 0.0  # Cytochalasin D abolishes platelet contribution
    else:
        mm_plt = IIa / (IIa + p['Km_plt'])
        dPa_dt = p['k_cat_plt'] * mm_plt * (1.0 - Pa)

    return [dTF_dt, dXa_dt, dIIa_dt, dVa_dt, dProt_dt,
            dFg_dt, dF_dt, dFp_dt, dPlg_dt, dPln_dt, dPa_dt]


# ==========================================
# 1C. AMPLITUDE CALCULATION
# ==========================================
def calculate_amplitude(state_array, assay_type, p, mode):
    """
    Compute viscoelastic amplitude A(t) from state array.
    Educational mode: 8 states → Fp at index 4, Pa at index 7.
    Research mode:   11 states → Fp at index 7, Pa at index 10.
    """
    if mode == "research":
        Fp = state_array[7].copy()
        Pa = state_array[10].copy()
    else:
        Fp = state_array[4].copy()
        Pa = state_array[7].copy()

    if assay_type == 'FIBTEM':
        Pa = np.zeros_like(Pa)

    CE_fib = p['CEmax_fib'] * Fp / (Fp + p['K_F'])
    gate = Fp / (Fp + p['K_gate'])
    CE_plt = p['CEmax_plt'] * Pa / (Pa + p['K_Pa']) * gate
    CE = CE_fib + CE_plt
    A_t = 100 * CE / (100 + CE)
    return A_t


# ==========================================
# 1D. SIMULATION RUNNERS
# ==========================================
@st.cache_data(show_spinner=False)
def run_simulation_heuristic(assay, pathology, adv_params_tuple, random_seed):
    """Run the 8-state Educational (heuristic) simulation."""
    np.random.seed(random_seed)

    params = dict(
        k_TF_act=0.05, k_AT=0.001, alpha1_max=0.5, alpha2=0.15,
        n_hill=6, K_thr=55.0, K_thr_p=65.0,
        k_plasmin_act=0.0002, k_a2AP=0.02, k_deg=0.005,
        k_cat_P=0.05, CEmax_fib=18.5, CEmax_plt=146.0,
        K_F=0.3, K_Pa=0.1, K_gate=0.1
    )

    # Apply advanced parameters (from hashable tuple)
    adv_dict = dict(adv_params_tuple)
    params.update(adv_dict)

    y0 = [10.0, 0.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    if pathology == "Hypofibrinogenemia":
        y0[2] = 2.0
    elif pathology == "Thrombocytopenia":
        params['CEmax_plt'] = 40.0
    elif pathology == "Hyperfibrinolysis":
        params['k_plasmin_act'] = 0.01
        params['k_deg'] = 1.0

    params['assay'] = assay
    t_span = (0, 3600)

    try:
        sol = solve_ivp(coagulation_model_heuristic, t_span, y0, args=(params,),
                        method='LSODA', rtol=1e-8, atol=1e-12, dense_output=True)
        t_eval = np.linspace(0, 3600, 1000)
        states = sol.sol(t_eval)
        A_t = calculate_amplitude(states, assay, params, mode="educational")

        if np.any(A_t < 0) or np.any(np.diff(t_eval) <= 0):
            raise ValueError("Simulation validation failed: Invalid trace generated.")

        return t_eval, A_t, params
    except Exception as e:
        return None, None, str(e)


@st.cache_data(show_spinner=False)
def run_simulation_research(assay, pathology, random_seed):
    """Run the 11-state Research (Michaelis-Menten) simulation."""
    np.random.seed(random_seed)

    # Literature-derived rate constants (Monograph Chapter 2)
    params = dict(
        # --- Thrombin generation (Hockin et al. 2002) ---
        k_TF_Xa=0.015,           # TF-driven Xa generation rate
        k_Xa_inh=0.005,          # TFPI-mediated Xa inhibition
        k_cat_prot=63.5,         # Prothrombinase catalytic rate (s^-1)
        Km_prot=0.322,           # Michaelis constant for prothrombinase (μM, normalized)
        k_cat_Xa=2.3e-3,         # Direct Xa-membrane catalytic rate (s^-1)
        Km_Xa=0.3,               # Michaelis constant for Xa-membrane (μM, normalized)
        k_AT=0.005,              # AT-III inhibition (effective, scaled from 7.1e3 M^-1s^-1)
        II_0=1.4,                # Initial prothrombin concentration (μM, physiological)
        # --- Prothrombinase assembly ---
        k_prot_assem=0.005,      # Xa + Va assembly rate (scaled for reduced model)
        k_prot_inh=0.001,        # Prothrombinase degradation
        k_IIa_Va=0.08,           # Thrombin-mediated Va activation
        k_Va_inh=0.002,          # Va inactivation
        # --- Fibrin polymerization (Ratto et al. 2021) ---
        alpha1=0.12,             # Thrombin-catalyzed fibrinogen cleavage
        alpha2=0.15,             # Fibrin polymerization rate
        # --- Fibrinolysis (Ouedraogo et al. 2024) ---
        k_plasmin_act=0.0002,    # Plasminogen → plasmin activation
        k_a2AP=0.02,             # α2-antiplasmin inhibition
        k_deg=0.005,             # Plasmin-mediated fibrin degradation (effective, from 5.0 s^-1)
        # --- Platelet activation (Fogelson/Luan 2007) ---
        k_cat_plt=0.05,          # Platelet activation k_cat (s^-1)
        Km_plt=0.63,             # K_M for platelet activation (nM, normalized)
        # --- Amplitude mapping ---
        CEmax_fib=18.5,
        CEmax_plt=146.0,
        K_F=0.3,
        K_Pa=0.1,
        K_gate=0.1
    )

    # Initial conditions: 11 state variables
    # [TF, Xa, IIa, Va, Prot, Fg, F, Fp, Plg, Pln, Pa]
    # TF0 = 0.06: picomolar-range tissue factor consistent with Hockin 2002
    # extrinsic pathway initiation (scaled for reduced 11-state model)
    y0 = [0.06, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    if assay in ['INTEM', 'HEPTEM']:
        params['k_TF_Xa'] = 0.003
        params['Km_plt'] = 1.2

    if pathology == "Hypofibrinogenemia":
        y0[5] = 2.0  # Reduced fibrinogen
    elif pathology == "Thrombocytopenia":
        params['CEmax_plt'] = 40.0
    elif pathology == "Hyperfibrinolysis":
        params['k_plasmin_act'] = 0.01
        params['k_deg'] = 1.0

    params['assay'] = assay
    t_span = (0, 3600)

    try:
        sol = solve_ivp(coagulation_model_exact, t_span, y0, args=(params,),
                        method='LSODA', rtol=1e-8, atol=1e-12, dense_output=True,
                        max_step=5e-1)
        t_eval = np.linspace(0, 3600, 1000)
        states = sol.sol(t_eval)
        A_t = calculate_amplitude(states, assay, params, mode="research")

        if np.any(A_t < 0) or np.any(np.diff(t_eval) <= 0):
            raise ValueError("Simulation validation failed: Invalid trace generated.")

        return t_eval, A_t, params
    except Exception as e:
        return None, None, str(e)


# ==========================================
# 2. METRICS & STATE DERIVATION
# ==========================================
def derive_metrics(t_eval, A_t):
    ct_idx = np.argmax(A_t > 2.0)
    ct = t_eval[ct_idx] if A_t[ct_idx] > 2.0 else 3600

    mcf = np.max(A_t)
    mcf_time = t_eval[np.argmax(A_t)]

    li60 = 100 * A_t[-1] / mcf if mcf > 0 else 0
    ml = 100 - li60 if mcf > 0 else 0

    idx_5min = np.argmin(np.abs(t_eval - 300))
    idx_10min = np.argmin(np.abs(t_eval - 600))

    return {
        "CT_s": float(ct),
        "MCF_mm": float(mcf),
        "LI60_pct": float(li60),
        "ML_pct": float(ml),
        "A5_mm": float(A_t[idx_5min]),
        "A10_mm": float(A_t[idx_10min]),
        "MCF_time_s": float(mcf_time)
    }

def get_phase(t, ct, mcf_time):
    if t < ct: return "Initiation (Pre-clotting)"
    elif t < mcf_time: return "Propagation (Clot Formation)"
    else: return "Lysis / Plateau"

# ==========================================
# 3. REPRODUCIBILITY & EXPORT
# ==========================================
def generate_manifest(sim_id, assay, pathology, params, random_seed, mode):
    return {
        "simulation_id": sim_id,
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION if mode == "educational" else MODEL_VERSION_RESEARCH,
        "model_mode": mode,
        "model_description": (
            "Hill-gated heuristic (8 state variables)" if mode == "educational"
            else "Michaelis-Menten direct math (11 state variables, Hockin-Mann)"
        ),
        "software_version": APP_VERSION,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "assay": assay,
        "pathology_preset": pathology,
        "simulation_duration_min": 60,
        "time_step_s": 3.6,
        "parameter_set": {k: v for k, v in params.items() if k != 'assay'},
        "random_seed": random_seed,
        "solver_tolerances": {
            "method": "LSODA",
            "rtol": 1e-8,
            "atol": 1e-12
        }
    }

def generate_sim_id(assay, pathology, params, seed, mode):
    hash_str = f"{mode}{assay}{pathology}{json.dumps(params, sort_keys=True)}{seed}"
    return "VET-" + hashlib.md5(hash_str.encode()).hexdigest()[:6].upper()

def export_to_json(manifest, metrics, t_eval, A_t):
    data = {
        "manifest": manifest,
        "derived_metrics": metrics,
        "time_series": {
            "time_s": t_eval.tolist(),
            "amplitude_mm": A_t.tolist()
        }
    }
    return json.dumps(data, indent=4)

def export_to_csv(t_eval, A_t):
    df = pd.DataFrame({"time_s": t_eval, "amplitude_mm": A_t})
    return df.to_csv(index=False)

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="OpenVET Simulator", layout="wide")

st.markdown("""
<style>
    /* ---- Type system: Arial throughout, clinical report register ---- */
    html, body, [class*="css"], .stApp, .stMarkdown, .stText, p,
    h1, h2, h3, h4, h5, h6, label, button, input, textarea,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {
        font-family: Arial, Helvetica, "Liberation Sans", sans-serif !important;
    }
    /* Preserve Material Symbols Rounded font for ALL Streamlit icon spans (expander, selectbox, etc.) */
    [data-testid="stIconMaterial"],
    span[data-testid="stIconMaterial"] {
        font-family: "Material Symbols Rounded" !important;
    }
    /* Apply Arial to expander content but exclude icon spans */
    [data-testid="stExpander"] div, [data-testid="stExpander"] p,
    [data-testid="stExpander"] span:not([data-testid="stIconMaterial"]):not(.material-icons):not([class*="material-icons"]) {
        font-family: Arial, Helvetica, "Liberation Sans", sans-serif !important;
    }

    :root {
        --ink: #16283D;        /* headers, primary text */
        --body-text: #2B2F36;  /* running text */
        --accent: #0F6E7F;     /* clinical teal - traces, links, highlights */
        --alert: #9C2B3A;      /* muted clinical red - current-time marker only */
        --paper: #F6F7F5;      /* page background, like report stock */
        --panel: #FFFFFF;      /* card background */
        --border: #D7DBDE;     /* hairline borders */
        --muted: #667085;      /* secondary/meta text */
    }

    .stApp { background-color: var(--paper); }

    .main-header {
        font-size: 25px; font-weight: 700; letter-spacing: 0.2px;
        color: var(--ink); margin-bottom: 2px;
    }
    .sub-header {
        font-size: 13.5px; color: var(--muted); margin-bottom: 4px;
        border-bottom: 1px solid var(--border); padding-bottom: 14px;
    }
    .citation-line {
        font-size: 11.5px; color: var(--muted); margin-bottom: 18px; font-style: italic;
    }
    .metric-box {
        background-color: var(--panel); padding: 15px 16px; border-radius: 6px;
        border: 1px solid var(--border); border-left: 3px solid var(--accent);
        margin-bottom: 10px; color: var(--body-text); line-height: 1.55;
    }
    .meta-row {
        font-size: 11.5px; color: var(--muted); margin-bottom: 18px;
        letter-spacing: 0.2px;
    }
    .meta-row b { color: var(--ink); }

    /* Mode badge */
    .mode-badge {
        display: inline-block; font-size: 11px; font-weight: 600;
        padding: 3px 10px; border-radius: 12px; margin-left: 12px;
        letter-spacing: 0.3px; vertical-align: middle;
    }
    .mode-badge.educational {
        background-color: #E8F4F6; color: #0F6E7F; border: 1px solid #B8DDE3;
    }
    .mode-badge.research {
        background-color: #FFF3E0; color: #B8600A; border: 1px solid #FDCF94;
    }

    /* Read-only constants box */
    .constants-box {
        background-color: #F8F9FA; padding: 12px 14px; border-radius: 6px;
        border: 1px solid var(--border); font-size: 12px; color: var(--body-text);
        line-height: 1.7; font-family: 'Courier New', Courier, monospace !important;
    }
    .constants-box .label { color: var(--muted); font-size: 11px; }
    .constants-box .val { color: var(--ink); font-weight: 600; }

    /* Metric widgets */
    [data-testid="stMetric"] {
        background-color: var(--panel); padding: 14px 16px; border-radius: 6px;
        border: 1px solid var(--border); border-left: 3px solid var(--accent);
    }
    [data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 12.5px !important; }

    /* Section labels e.g. "##### Time Navigation" */
    h5 { color: var(--ink) !important; font-weight: 700 !important;
         text-transform: uppercase; font-size: 12.5px !important; letter-spacing: 0.6px; }

    /* Buttons */
    .stButton button, [data-testid="stDownloadButton"] button {
        font-family: Arial, Helvetica, sans-serif !important;
        border-radius: 5px !important; border: 1px solid var(--border) !important;
        color: var(--ink) !important; background-color: var(--panel) !important;
        font-weight: 600 !important; font-size: 13px !important;
    }
    .stButton button:hover, [data-testid="stDownloadButton"] button:hover {
        border-color: var(--accent) !important; color: var(--accent) !important;
    }

    /* Expanders styled as report sections */
    [data-testid="stExpander"] {
        border: 1px solid var(--border) !important; border-radius: 6px !important;
        background-color: var(--panel) !important;
    }

    hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# --- INIT SESSION STATE ---
if 'selected_time' not in st.session_state:
    st.session_state.selected_time = 60.0
if 'random_seed' not in st.session_state:
    st.session_state.random_seed = 42
if 'adv_params' not in st.session_state:
    st.session_state.adv_params = {}

# --- SIDEBAR: MODEL MODE TOGGLE ---
with st.sidebar:
    st.markdown("##### Model Execution Mode")
    model_mode = st.radio(
        "Select mode",
        options=["Educational (Heuristic)", "Research (Direct Math)"],
        index=0,
        label_visibility="collapsed",
        help="Educational: fast, 8-state Hill-function model. Research: exact 11-state Michaelis-Menten system from the monograph."
    )
    is_research = model_mode == "Research (Direct Math)"
    mode_key = "research" if is_research else "educational"

    st.markdown("---")
    if is_research:
        st.markdown(
            "<div style='font-size:11.5px; color:#B8600A; line-height:1.55;'>"
            "<b>Research Mode</b> uses the exact Michaelis-Menten kinetics from "
            "Chapter 2 of the monograph (Hockin-Mann thrombin generation, "
            "Fogelson platelet activation, Ouedraogo fibrinolysis). "
            "11 state variables are integrated."
            "</div>", unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='font-size:11.5px; color:#0F6E7F; line-height:1.55;'>"
            "<b>Educational Mode</b> uses Hill-function gating for clot "
            "initiation thresholds. 8 state variables. Fast execution, "
            "tunable via Advanced Parameters."
            "</div>", unsafe_allow_html=True
        )

# --- HEADER ---
if is_research:
    badge_html = "<span class='mode-badge research'>Research Mode (Direct Hockin-Mann Math)</span>"
else:
    badge_html = "<span class='mode-badge educational'>Educational Mode (Reduced Heuristic)</span>"

st.markdown(f"<div class='main-header'>OpenVET Simulator{badge_html}</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>A mechanistic, reduced-order computational framework for viscoelastic coagulation testing.</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='citation-line'>Model structure informed by Hammami, M.W. (2026), "
    "<i>Mathematical, Physical, and Computational Principles of Rotational Thromboelastometry</i> — "
    "an unpublished monograph reviewing Hockin et al. (2002), Weisel &amp; Nagaswami (1992), Ratto et al. (2021), "
    "and Ouedraogo et al. (2024), among others. Parameters here are heuristically tuned to reference ranges, "
    "not fitted to the literature rate constants directly — see &ldquo;About the Model&rdquo; below.</div>",
    unsafe_allow_html=True
)

# --- CONTROLS SECTION ---
with st.container():
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1: assay = st.selectbox("Assay", ["EXTEM", "INTEM", "FIBTEM", "APTEM"])
    with c2: pathology = st.selectbox("Condition", ["Normal", "Hypofibrinogenemia", "Thrombocytopenia", "Hyperfibrinolysis"])
    with c3: st.session_state.random_seed = st.number_input("Random Seed", min_value=0, value=42, step=1)
    with c4:
        st.write("")  # Spacer for alignment
        st.write("")
        reset_btn = st.button("Reset Time Cursor", use_container_width=True)

# --- ADVANCED PARAMETERS (Collapsible) ---
with st.expander("Advanced Parameters", expanded=False):
    if is_research:
        # Research mode: show read-only Michaelis-Menten constants
        st.markdown(
            "<div style='font-size:12px; color:#667085; margin-bottom:10px;'>"
            "In Research Mode, the model uses literature-derived Michaelis-Menten constants. "
            "Hill-threshold sliders are not applicable.</div>",
            unsafe_allow_html=True
        )
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown("**Thrombin Generation**")
            st.markdown(
                "<div class='constants-box'>"
                "<span class='label'>k_cat_prot:</span> <span class='val'>63.5 s⁻¹</span><br>"
                "<span class='label'>k_cat_Xa:</span> <span class='val'>2.3×10⁻³ s⁻¹</span><br>"
                "<span class='label'>K_m_prot:</span> <span class='val'>0.322 μM</span><br>"
                "<span class='label'>K_m_Xa:</span> <span class='val'>0.3 μM</span><br>"
                "<span class='label'>k_AT (AT-III):</span> <span class='val'>7.1×10³ M⁻¹s⁻¹</span><br>"
                "<span class='label'>[II]₀ (prothrombin):</span> <span class='val'>1.4 μM</span>"
                "</div>", unsafe_allow_html=True
            )
        with rc2:
            st.markdown("**Platelet Activation**")
            st.markdown(
                "<div class='constants-box'>"
                "<span class='label'>k_cat_plt:</span> <span class='val'>0.05 s⁻¹</span><br>"
                "<span class='label'>K_M_plt:</span> <span class='val'>0.63 nM</span><br>"
                "<span class='label'>Source:</span> <span class='val'>Fogelson/Luan 2007</span>"
                "</div>", unsafe_allow_html=True
            )
            st.markdown("**Fibrinolysis**")
            st.markdown(
                "<div class='constants-box'>"
                "<span class='label'>k_deg:</span> <span class='val'>5.0 s⁻¹</span><br>"
                "<span class='label'>k_a2AP:</span> <span class='val'>0.02 s⁻¹</span><br>"
                "<span class='label'>Source:</span> <span class='val'>Ouedraogo et al. 2024</span>"
                "</div>", unsafe_allow_html=True
            )
        with rc3:
            st.markdown("**Fibrin Polymerization**")
            st.markdown(
                "<div class='constants-box'>"
                "<span class='label'>α₁ (cleavage):</span> <span class='val'>0.12</span><br>"
                "<span class='label'>α₂ (polymer.):</span> <span class='val'>0.15</span><br>"
                "<span class='label'>Source:</span> <span class='val'>Ratto et al. 2021</span>"
                "</div>", unsafe_allow_html=True
            )
            st.markdown("**Amplitude Mapping**")
            st.markdown(
                "<div class='constants-box'>"
                "<span class='label'>CEmax_fib:</span> <span class='val'>18.5</span><br>"
                "<span class='label'>CEmax_plt:</span> <span class='val'>146.0</span><br>"
                "<span class='label'>K_F:</span> <span class='val'>0.3</span><br>"
                "<span class='label'>K_Pa:</span> <span class='val'>0.1</span>"
                "</div>", unsafe_allow_html=True
            )
    else:
        # Educational mode: interactive sliders
        adv_c1, adv_c2, adv_c3 = st.columns(3)
        with adv_c1:
            st.session_state.adv_params['k_TF_act'] = st.slider("Clot Initiation (k_TF_act)", 0.01, 0.1, 0.05, 0.01)
            st.session_state.adv_params['k_deg'] = st.slider("Lysis Rate (k_deg)", 0.001, 1.0, 0.005, 0.001, format="%.3f")
        with adv_c2:
            st.session_state.adv_params['CEmax_fib'] = st.slider("Max Firmness (CEmax_fib)", 5.0, 30.0, 18.5, 0.5)
            st.session_state.adv_params['CEmax_plt'] = st.slider("Platelet Contrib (CEmax_plt)", 20.0, 200.0, 146.0, 5.0)
        with adv_c3:
            st.session_state.adv_params['alpha1_max'] = st.slider("Propagation (alpha1_max)", 0.1, 1.0, 0.5, 0.05)

# --- RUN SIMULATION ---
if is_research:
    result = run_simulation_research(assay, pathology, st.session_state.random_seed)
else:
    # Convert adv_params dict to hashable tuple for @st.cache_data
    adv_tuple = tuple(sorted(st.session_state.adv_params.items()))
    result = run_simulation_heuristic(assay, pathology, adv_tuple, st.session_state.random_seed)

t_eval, A_t, params_or_err = result

if t_eval is None:
    st.error(f"Simulation could not be generated.\n\nReason: {params_or_err}")
    st.stop()

params_used = params_or_err

# --- METRICS & STATE ---
metrics = derive_metrics(t_eval, A_t)
sim_id = generate_sim_id(assay, pathology, params_used, st.session_state.random_seed, mode_key)
manifest = generate_manifest(sim_id, assay, pathology, params_used, st.session_state.random_seed, mode_key)

active_model_ver = MODEL_VERSION if not is_research else MODEL_VERSION_RESEARCH
st.markdown(
    f"<div class='meta-row'>Simulation ID: <b>{sim_id}</b> | "
    f"Model Version: {active_model_ver} | App Version: {APP_VERSION} | "
    f"Mode: <b>{'Research' if is_research else 'Educational'}</b></div>",
    unsafe_allow_html=True
)

# --- RESET BUTTON LOGIC ---
if reset_btn:
    st.session_state.selected_time = 60.0

# --- LAYOUT ---
col1, col2 = st.columns([2.5, 1])

with col1:
    st.markdown("##### Time Navigation")
    selected_time = st.slider("Simulation Time (min)", 0.0, 60.0, float(st.session_state.selected_time), 0.1,
                              label_visibility="collapsed")
    st.session_state.selected_time = selected_time

    selected_idx = int((selected_time / 60.0) * (len(t_eval) - 1))
    selected_amp = A_t[selected_idx]

    # Dynamic Y-axis range based on MCF
    y_max = max(80, metrics['MCF_mm'] * 1.2)

    # PLOT
    fig = go.Figure()

    # Area fill under the trace (clinical ROTEM-style shading)
    fig.add_trace(go.Scatter(
        x=t_eval / 60, y=A_t, mode='lines', name=f'{assay} ({pathology})',
        line=dict(color='#16283D', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(15, 110, 127, 0.08)',
        hovertemplate="Time: %{x:.1f} min<br>Amplitude: %{y:.1f} mm<extra></extra>"
    ))

    # CT Marker
    ct_min = metrics['CT_s'] / 60
    fig.add_vline(x=ct_min, line_width=1.5, line_dash="dot", line_color="#98A2B3")
    fig.add_annotation(x=ct_min, y=2, text="CT", showarrow=False, font=dict(color="#667085", size=12, family="Arial"), yshift=10)

    # MCF Marker
    mcf_min = metrics['MCF_time_s'] / 60
    fig.add_trace(go.Scatter(
        x=[mcf_min], y=[metrics['MCF_mm']], mode='markers', name='MCF',
        marker=dict(color='#0F6E7F', size=9, symbol='x-thin', line=dict(width=2, color='#0F6E7F')), showlegend=False
    ))

    # Selected Time Marker
    fig.add_vline(x=selected_time, line_width=2, line_dash="dash", line_color="#9C2B3A")
    fig.add_trace(go.Scatter(
        x=[selected_time], y=[selected_amp], mode='markers', name='Current',
        marker=dict(color='#9C2B3A', size=11), showlegend=False,
        hovertemplate=f"Time: {selected_time:.1f} min<br>Amplitude: {selected_amp:.1f} mm<extra></extra>"
    ))

    # Mode annotation
    mode_annotation_text = "Research · Hockin-Mann" if is_research else "Educational · Heuristic"
    mode_annotation_color = "#B8600A" if is_research else "#0F6E7F"

    fig.update_layout(
        title=dict(text=f'Simulated Viscoelastic Trace: {assay} – {pathology}', font=dict(size=14, color="#16283D", family="Arial")),
        xaxis_title='Time (minutes)',
        yaxis_title='Clot Amplitude [mm]',
        yaxis_range=[-5, y_max],
        xaxis_range=[0, 60],
        template='plotly_white',
        font=dict(family="Arial, Helvetica, sans-serif", color="#2B2F36", size=12),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        annotations=[
            dict(
                text=mode_annotation_text,
                xref="paper", yref="paper",
                x=1.0, y=1.05,
                showarrow=False,
                font=dict(size=10.5, color=mode_annotation_color, family="Arial"),
                xanchor="right"
            )
        ]
    )
    fig.update_xaxes(gridcolor="#EEF0F2", zerolinecolor="#D7DBDE")
    fig.update_yaxes(gridcolor="#EEF0F2", zerolinecolor="#D7DBDE")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("##### Live Parameters")
    mc1, mc2 = st.columns(2)
    with mc1: st.metric("CT", f"{metrics['CT_s']:.0f} s")
    with mc2: st.metric("MCF", f"{metrics['MCF_mm']:.1f} mm")
    with mc1: st.metric("LI60", f"{metrics['LI60_pct']:.1f} %")
    with mc2: st.metric("ML", f"{metrics['ML_pct']:.1f} %")

    st.markdown("---")

    st.markdown("##### State at Selected Time")
    st.markdown(f"""
    <div class="metric-box">
        <b>Time:</b> {selected_time:.1f} min<br>
        <b>Amplitude:</b> {selected_amp:.1f} mm<br>
        <b>Phase:</b> {get_phase(selected_time*60, metrics['CT_s'], metrics['MCF_time_s'])}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("##### Export")
    ex1, ex2 = st.columns(2)
    with ex1:
        st.download_button("Export JSON", data=export_to_json(manifest, metrics, t_eval, A_t),
                           file_name=f"{sim_id}.json", mime="application/json")
    with ex2:
        st.download_button("Export CSV", data=export_to_csv(t_eval, A_t),
                           file_name=f"{sim_id}.csv", mime="text/csv")

# --- BOTTOM SECTION: MANIFEST & ABOUT ---
with st.expander("Reproducibility Manifest (JSON)", expanded=False):
    st.json(manifest)

with st.expander("About the Model", expanded=False):
    st.markdown("""
    **Scientific Disclaimer & Context**
    - This application is an **open-source computational simulator** intended for education, research, and software experimentation.
    - The underlying mathematical model is a **reduced-order ordinary differential equation (ODE) system** capturing thrombin generation, fibrin polymerization, fibrinolysis, and platelet activation.
    - Traces are **simulated** based on mechanistic heuristics calibrated against clinical viscoelastic testing reference ranges (e.g. Levrat et al. 2008; de Vries et al. 2020), not fitted directly to published rate constants.
    - **It is not a clinical diagnostic device** and does not imply clinical validation. Parameter choices should be interpreted strictly in the context of this computational model.

    **Model Modes**

    | Feature | Educational (Heuristic) | Research (Direct Math) |
    |---|---|---|
    | State Variables | 8 | 11 (TF, Xa, IIa, Va, Prothrombinase, Fg, F, Fp, Plg, Pln, Pa) |
    | Thrombin Generation | Hill-function threshold gating | Michaelis-Menten prothrombinase kinetics (k_cat = 63.5 s\u207b\u00b9) with substrate depletion |
    | Prothrombin | Not modeled explicitly | Consumed as thrombin is generated ([II] = [II]\u2080 \u2212 [IIa]) |
    | Platelet Activation | Hill-function threshold | Michaelis-Menten (k_cat = 0.05, K_M = 0.63 nM) |
    | Fibrinolysis | Simplified rate constant | Ouedraogo et al. k_deg = 5.0 s\u207b\u00b9 framework |
    | Tunability | Interactive sliders | Fixed literature constants |

    **Conceptual Lineage**
    This simulator's structure \u2014 thrombin generation \u2192 fibrinogen conversion \u2192 fibrinolysis \u2192 platelet activation \u2192 viscoelastic readout \u2014 follows the reduced-order modeling logic laid out in the accompanying monograph (Hammami, 2026), which draws on:
    - Hockin MF, et al. (2002). *J Biol Chem* \u2014 stoichiometric thrombin generation model.
    - Weisel JW, Nagaswami C. (1992). *Biophys J* \u2014 fibrin polymerization kinetics.
    - Ouedraogo RR, et al. (2024). *PLoS Comput Biol* \u2014 plasmin-mediated single-fiber fibrinolysis.
    - Fogelson AL, et al. (2012). *Biophys J* \u2014 thrombin-driven platelet activation under flow.
    - Ratto N, et al. (2021). *Bull Math Biol* \u2014 patient-specific reduced-kinetic coagulation modeling.
    - Solomon C, et al. (2015); de Vries JJ, et al. (2020); Levrat A, et al. (2008); Chapman MP, et al. (2015) \u2014 clinical viscoelastic testing reference ranges and pathological phenotypes used for calibration targets.

    **Important honesty note:** the *Educational Mode* equations (Hill-function thresholding for clot initiation, an additive fibrin+platelet contribution to clot elasticity) are a *simplification*, not a direct numerical implementation of the literature rate constants. The *Research Mode* implements the Michaelis-Menten kinetics from Chapter 2 of the monograph more directly, with mass-conserving prothrombin substrate depletion, but effective rate constants are still scaled to produce clinically realistic ROTEM amplitudes. Treat the citations as intellectual lineage and calibration targets.
    """)