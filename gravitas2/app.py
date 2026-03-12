"""
Gravitas AI 2.0 — Pregnancy Drug Safety Intelligence Platform
The Menon Laboratory · Perinatal Research
"It's About Saving Babies"
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import anthropic

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Gravitas AI 2.0",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] {
    background-color: #080C14;
    color: #E2E8F0;
    font-family: 'Segoe UI', sans-serif;
}
.main { background-color: #080C14; }
section[data-testid="stSidebar"] {
    background-color: #0D1421;
    border-right: 1px solid #1A2744;
}

/* ── Title ── */
.g-title { font-size: 2.8rem; font-weight: 900; color: #E8707A;
           font-family: Georgia, serif; letter-spacing: -1px; margin: 0; }
.g-subtitle { font-size: 1.05rem; color: #63B3ED; margin-top: 2px; }
.g-tagline { font-size: 0.88rem; color: #F6AD55; font-style: italic; margin-top: 4px; }
.g-version { background: linear-gradient(135deg,#E8707A22,#63B3ED22);
             border: 1px solid #E8707A44; border-radius: 20px;
             padding: 2px 12px; font-size: 0.75rem; color: #E8707A;
             display: inline-block; font-weight: 700; letter-spacing: 1px; }

/* ── Cards ── */
.g-card { background: #0D1421; border: 1px solid #1E3A5F;
          border-radius: 12px; padding: 18px 20px; margin-bottom: 14px;
          box-shadow: 0 4px 20px #00000044; }
.g-card-pink  { border-left: 4px solid #E8707A; }
.g-card-blue  { border-left: 4px solid #63B3ED; }
.g-card-green { border-left: 4px solid #68D391; }
.g-card-gold  { border-left: 4px solid #F6AD55; }
.g-card-purple{ border-left: 4px solid #B794F4; }
.g-card-teal  { border-left: 4px solid #4FD1C5; }
.g-card-red   { border-left: 4px solid #FC8181; }

/* ── Risk badges ── */
.badge { display: inline-block; padding: 3px 12px; border-radius: 20px;
         font-size: 0.78rem; font-weight: 700; letter-spacing: 1px; }
.badge-high     { background: #FC818144; color: #FC8181; border: 1px solid #FC818166; }
.badge-moderate { background: #F6AD5544; color: #F6AD55; border: 1px solid #F6AD5566; }
.badge-low      { background: #68D39144; color: #68D391; border: 1px solid #68D39166; }
.badge-unknown  { background: #A0AEC044; color: #A0AEC0; border: 1px solid #A0AEC066; }
.badge-positive { background: #FC818144; color: #FC8181; border: 1px solid #FC818166; }
.badge-negative { background: #68D39144; color: #68D391; border: 1px solid #68D39166; }
.badge-equivocal{ background: #F6AD5544; color: #F6AD55; border: 1px solid #F6AD5566; }

/* ── Metric tiles ── */
.metric-tile { background: #111827; border: 1px solid #1E3A5F;
               border-radius: 10px; padding: 12px 16px; text-align: center; }
.metric-val { font-size: 1.5rem; font-weight: 700; }
.metric-lbl { font-size: 0.72rem; color: #718096; text-transform: uppercase;
              letter-spacing: 1px; margin-top: 2px; }

/* ── Section headers ── */
.sec-header { font-size: 1.1rem; font-weight: 700; color: #E2E8F0;
              padding: 8px 0 6px 0; border-bottom: 1px solid #1E3A5F;
              margin-bottom: 14px; }

/* ── Progress bars ── */
.prog-row { display: flex; align-items: center; gap: 10px; margin-bottom: 7px; }
.prog-label { width: 130px; font-size: 0.8rem; color: #A0AEC0; text-align: right;
              white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.prog-bar-bg { flex: 1; background: #1A2744; border-radius: 6px; height: 10px; }
.prog-bar-fill { height: 10px; border-radius: 6px; }
.prog-val { width: 44px; font-size: 0.78rem; color: #E2E8F0; text-align: right; }

/* ── Table ── */
.g-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.g-table th { background: #1A2744; color: #63B3ED; padding: 8px 12px;
              text-align: left; font-weight: 600; font-size: 0.78rem;
              text-transform: uppercase; letter-spacing: 0.5px; }
.g-table td { padding: 7px 12px; border-bottom: 1px solid #1A2744; color: #E2E8F0; }
.g-table tr:hover td { background: #111827; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #0D1421; border-bottom: 1px solid #1E3A5F; gap: 4px; }
.stTabs [data-baseweb="tab"] { color: #718096; background: transparent;
    border-radius: 8px 8px 0 0; padding: 8px 18px; font-size: 0.88rem; }
.stTabs [aria-selected="true"] { color: #E8707A !important; border-bottom: 2px solid #E8707A !important;
    background: #0D142188 !important; }

/* ── Sidebar ── */
.stSelectbox label, .stRadio label { color: #A0AEC0 !important; font-size: 0.85rem !important; }
div[data-testid="stSelectbox"] > div { background: #111827; border-color: #1E3A5F; }

/* ── PHI gauge ── */
.phi-gauge { text-align: center; padding: 20px; }
.phi-number { font-size: 3.5rem; font-weight: 900; font-family: Georgia, serif; }
.phi-label { font-size: 0.85rem; color: #A0AEC0; text-transform: uppercase; letter-spacing: 2px; }

/* ── AI chat ── */
.ai-bubble { background: #111827; border: 1px solid #1E3A5F; border-radius: 12px;
             padding: 16px 18px; margin: 10px 0; font-size: 0.92rem; line-height: 1.7; }
.ai-icon { font-size: 1.2rem; margin-right: 8px; }

/* ── Docking card ── */
.dock-card { background: #111827; border: 1px solid #1E3A5F; border-radius: 8px;
             padding: 10px 14px; text-align: center; }
.dock-val { font-size: 1.3rem; font-weight: 700; }
.dock-lbl { font-size: 0.72rem; color: #718096; margin-top: 2px; }

/* ── Disclaimer ── */
.disclaimer { background: #1A1200; border: 1px solid #F6AD5544; border-radius: 8px;
              padding: 10px 14px; font-size: 0.78rem; color: #F6AD55; margin-top: 16px; }

/* ── DART endpoint cell ── */
.dart-pos { color: #FC8181; font-weight: 600; }
.dart-neg { color: #68D391; }
.dart-na  { color: #4A5568; }
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING ──────────────────────────────────────────────
@st.cache_data
def load_all_data():
    base = os.path.dirname(os.path.abspath(__file__))
    
    # ADME + Tox + Docking
    adme_path = os.path.join(base, "data", "ADME_Toxicity_Moleculardocking_Parameters.xlsx")
    adme = pd.read_excel(adme_path).set_index("Drug names")
    
    # DART
    dart_path = os.path.join(base, "data", "DART_AI_ready_schema.xlsx")
    dart_summary = pd.read_excel(dart_path, sheet_name="Drug_Summary").set_index("Drug_name")
    dart_evidence = pd.read_excel(dart_path, sheet_name="Raw_Evidence")
    dart_ontology = pd.read_excel(dart_path, sheet_name="Endpoint_Ontology")
    
    # PBPK
    pbpk_path = os.path.join(base, "data", "Pregnancy_PBPK_AI_Prototype.xlsx")
    pbpk_scenarios = pd.read_excel(pbpk_path, sheet_name="Scenario_Calculator")
    pbpk_params    = pd.read_excel(pbpk_path, sheet_name="Drug_PBPK_Parameters").set_index("Drug_name")
    pbpk_physio    = pd.read_excel(pbpk_path, sheet_name="Pregnancy_Physiology_Defaults")
    pbpk_lit       = pd.read_excel(pbpk_path, sheet_name="Literature_Status").set_index("Drug_name")
    
    return {
        "adme": adme,
        "dart_summary": dart_summary,
        "dart_evidence": dart_evidence,
        "dart_ontology": dart_ontology,
        "pbpk_scenarios": pbpk_scenarios,
        "pbpk_params": pbpk_params,
        "pbpk_physio": pbpk_physio,
        "pbpk_lit": pbpk_lit,
        "drugs": sorted(adme.index.tolist()),
    }

# ── HELPERS ───────────────────────────────────────────────────
def safe(val, decimals=3, pct=False, suffix=""):
    try:
        if pd.isna(val): return "N/A"
        v = float(val)
        if pct: return f"{v*100:.1f}%"
        return f"{v:.{decimals}f}{suffix}"
    except: return str(val)

def risk_badge(val, thresholds, labels, colors):
    """thresholds: list of cutoffs ascending, labels/colors: matching list"""
    try:
        v = float(val)
        for i, t in enumerate(thresholds):
            if v <= t: return f'<span class="badge badge-{colors[i]}">{labels[i]}</span>'
        return f'<span class="badge badge-{colors[-1]}">{labels[-1]}</span>'
    except: return '<span class="badge badge-unknown">N/A</span>'

def color_val(val, low=0.3, high=0.7, invert=False):
    """Return colored HTML for a probability value"""
    try:
        v = float(val)
        if invert: v = 1 - v
        if v >= high: color = "#FC8181"
        elif v >= low: color = "#F6AD55"
        else: color = "#68D391"
        return f'<span style="color:{color};font-weight:600">{float(val):.3f}</span>'
    except: return f'<span style="color:#718096">N/A</span>'

def docking_color(val):
    """Color docking scores (more negative = better binding = higher concern)"""
    try:
        v = float(val)
        if v <= -9.0: color = "#FC8181"
        elif v <= -7.0: color = "#F6AD55"
        elif v <= -5.0: color = "#68D391"
        else: color = "#A0AEC0"
        return f'<span style="color:{color};font-weight:600">{v:.1f}</span>'
    except: return '<span style="color:#718096">N/A</span>'

def prog_bar(label, val, max_val=1.0, color="#E8707A", pct=False, invert_color=False):
    try:
        v = float(val)
        pct_fill = min(100, max(0, (v / max_val) * 100))
        display = f"{v*100:.1f}%" if pct else f"{v:.3f}"
        if invert_color:
            bar_color = "#68D391" if v < 0.3 else ("#F6AD55" if v < 0.7 else "#FC8181")
        else:
            bar_color = color
        return f"""
        <div class="prog-row">
          <div class="prog-label">{label}</div>
          <div class="prog-bar-bg"><div class="prog-bar-fill" style="width:{pct_fill:.1f}%;background:{bar_color}"></div></div>
          <div class="prog-val">{display}</div>
        </div>"""
    except: return f'<div class="prog-row"><div class="prog-label">{label}</div><div class="prog-val">N/A</div></div>'

def compute_phi(row):
    """Pregnancy Hazard Index"""
    try:
        dili  = float(row.get("DILI", 0) or 0)
        herg  = float(row.get("hERG", 0) or 0)
        ames  = float(row.get("Ames", 0) or 0)
        nr_er = float(row.get("NR-ER", 0) or 0)
        nr_ar = float(row.get("NR-AR", 0) or 0)
        srp53 = float(row.get("SR-p53", 0) or 0)
        bbb   = float(row.get("BBB", 0) or 0)
        phi = (dili*0.25 + herg*0.15 + ames*0.10 +
               nr_er*0.15 + nr_ar*0.10 + srp53*0.10 + bbb*0.15)
        return round(phi * 100, 1)
    except: return None

def phi_level(phi):
    if phi is None: return "Unknown", "unknown"
    if phi >= 65: return "HIGH RISK", "high"
    if phi >= 40: return "MODERATE", "moderate"
    return "LOW RISK", "low"

def get_api_key():
    try: return st.secrets.get("ANTHROPIC_API_KEY", "")
    except: return os.environ.get("ANTHROPIC_API_KEY", "")

# ── AI ANALYSIS ───────────────────────────────────────────────
def ai_analyze(drug_name, data, user_type, question=""):
    key = get_api_key()
    if not key: return "⚠️ AI analysis unavailable — API key not configured."
    
    adme_row = data["adme"].loc[drug_name] if drug_name in data["adme"].index else {}
    dart_row = data["dart_summary"].loc[drug_name] if drug_name in data["dart_summary"].index else {}
    pbpk_drug = data["pbpk_scenarios"][data["pbpk_scenarios"]["Drug_name"]==drug_name]
    lit_row = data["pbpk_lit"].loc[drug_name] if drug_name in data["pbpk_lit"].index else {}
    
    phi = compute_phi(adme_row)
    phi_lev, _ = phi_level(phi)
    
    # Build context
    context = f"""
DRUG: {drug_name}
Pregnancy Hazard Index (PHI): {phi}/100 — {phi_lev}

ADME KEY VALUES:
- MW: {safe(adme_row.get('MW'))}, LogP: {safe(adme_row.get('logP'))}, TPSA: {safe(adme_row.get('TPSA'))} Å²
- BBB penetration: {safe(adme_row.get('BBB'))}, PPB: {safe(adme_row.get('PPB'))}%, Fu: {safe(adme_row.get('Fu'))}%
- HIA: {safe(adme_row.get('hia'))}, Caco-2: {safe(adme_row.get('caco2'))}, P-gp substrate: {safe(adme_row.get('pgp_sub'))}

TOXICITY:
- DILI: {safe(adme_row.get('DILI'))}, hERG: {safe(adme_row.get('hERG'))}, Ames: {safe(adme_row.get('Ames'))}
- NR-ER: {safe(adme_row.get('NR-ER'))}, NR-AR: {safe(adme_row.get('NR-AR'))}, NR-Aromatase: {safe(adme_row.get('NR-Aromatase'))}
- SR-p53: {safe(adme_row.get('SR-p53'))}, SR-ARE: {safe(adme_row.get('SR-ARE'))}

MOLECULAR DOCKING (kcal/mol — more negative = stronger binding):
- P38-MAPK: {safe(adme_row.get('P-38'),1)}, NF-κB: {safe(adme_row.get('NFKB'),1)}, MAPK: {safe(adme_row.get('MAPK'),1)}
- JAK2: {safe(adme_row.get('JAK2'),1)}, TGFβR1: {safe(adme_row.get('TGFBR1'),1)}, HIF1α: {safe(adme_row.get('HIF1A'),1)}

DART STUDIES:
- Overall signal: {dart_row.get('Overall_DART_signal','N/A')}
- Evidence confidence: {dart_row.get('Evidence_confidence','N/A')}
- EFD available: {dart_row.get('EFD_available','N/A')}
- Summary: {dart_row.get('Summary_text','N/A')}

PREGNANCY PBPK (T1/T2/T3):"""
    for _, row in pbpk_drug.iterrows():
        context += f"\n  {row['Trimester']}: Maternal Cavg={safe(row.get('Maternal_Cavg_mg_L'),4)} mg/L, Fetal Cavg={safe(row.get('Fetal_plasma_Cavg_mg_L'),4)} mg/L, Flag={row.get('Exposure_flag','N/A')}"
    
    context += f"\nPBPK Literature: {lit_row.get('Status_bucket','N/A')} — {lit_row.get('Key_quantitative_note','N/A')}"
    
    # User-type specific prompt
    if user_type == "👩‍⚕️ Clinician":
        system = """You are a clinical pharmacology expert specializing in obstetric pharmacotherapy. 
Provide evidence-based clinical guidance. Use medical terminology. Focus on: recommended dosing during pregnancy, 
trimester-specific risks, fetal exposure concerns, monitoring parameters, and clinical decision framework.
Be direct and clinically actionable. Reference DART data and PBPK predictions explicitly."""
        user_q = question or f"Provide a comprehensive clinical assessment of {drug_name} for use during pregnancy, including dosing recommendations and monitoring."
    
    elif user_type == "🤰 Patient":
        system = """You are a patient-friendly pregnancy drug safety advisor. 
Explain everything in plain, clear language a non-medical person can understand. 
Avoid jargon. Use analogies. Focus on: is this drug safe in pregnancy, what trimester concerns exist, 
what to discuss with their doctor, breastfeeding considerations. Be warm, reassuring but honest."""
        user_q = question or f"Can you explain whether {drug_name} is safe to take during pregnancy, in simple terms?"
    
    else:  # Pharma researcher
        system = """You are a pharmaceutical scientist expert in reproductive toxicology and DMPK.
Provide technical, mechanistic analysis. Focus on: ADME-PK interpretation, molecular docking significance,
PBPK model outputs, DART endpoints, network pharmacology implications, structure-activity considerations.
Use technical terminology. Reference specific values. Discuss mechanistic basis for observed parameters."""
        user_q = question or f"Provide a comprehensive pharmacological and toxicological analysis of {drug_name} for pregnancy research purposes."
    
    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            messages=[{"role": "user", "content": f"Context data:\n{context}\n\nQuestion: {user_q}"}],
            system=system
        )
        return response.content[0].text
    except Exception as e:
        return f"⚠️ AI analysis error: {str(e)}"

# ── NETWORK PHARMACOLOGY LOADER ────────────────────────────────
def load_network_data(drug_name):
    """Load network pharmacology files for a drug.
    
    Folder structure:
        {drug_name}/Network_Pharmacology/{drug_name}_KEGG.xlsx
    e.g.:
        Anastrozole/Network_Pharmacology/Anastrozole_KEGG.xlsx
    """
    base = os.path.dirname(os.path.abspath(__file__))
    drug_folder = os.path.join(base, drug_name, "Network_Pharmacology")
    result = {}

    file_types = {
        "kegg":       f"{drug_name}_KEGG.xlsx",
        "reactome":   f"{drug_name}_Reactome.xlsx",
        "go_bp":      f"{drug_name}_GO_BP.xlsx",
        "go_cc":      f"{drug_name}_GO_CC.xlsx",
        "go_mf":      f"{drug_name}_GO_MF.xlsx",
        "hub_genes":  f"{drug_name}_Hubgenes_STRING_centrality.xlsx",
        "common_genes": "Commongenes_Venn.xlsx",
    }

    for key, filename in file_types.items():
        path = os.path.join(drug_folder, filename)
        if os.path.exists(path):
            try:
                result[key] = pd.read_excel(path)
            except:
                pass

    # Also scan for figure files (.png / .jpg / .svg)
    figures = {}
    fig_types = {
        "fig_dotplot_bp":       f"{drug_name}_Dotplot_GO_BP",
        "fig_dotplot_cc":       f"{drug_name}_Dotplot_GO_CC",
        "fig_dotplot_mf":       f"{drug_name}_Dotplot_GO_MF",
        "fig_enrichmap_kegg":   f"{drug_name}_EnrichmentMap_KEGG",
        "fig_enrichmap_reactome": f"{drug_name}_EnrichmentMap_Reactome",
    }
    for key, stem in fig_types.items():
        for ext in [".png", ".jpg", ".jpeg", ".svg"]:
            path = os.path.join(drug_folder, stem + ext)
            if os.path.exists(path):
                figures[key] = path
                break
    result["figures"] = figures

    return result

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 16px 0">
        <div class="g-title" style="font-size:1.8rem">Gravitas AI</div>
        <div style="font-size:0.72rem;color:#63B3ED;margin-top:2px">Pregnancy Drug Safety Platform</div>
        <div style="margin-top:8px"><span class="g-version">VERSION 2.0</span></div>
        <div style="font-size:0.72rem;color:#F6AD55;font-style:italic;margin-top:6px">"It's About Saving Babies"</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Page navigation
    page = st.radio("Navigation", [
        "🏠 Home",
        "🔍 Drug Analysis",
        "📊 Database Explorer",
        "🕸️ Network Pharmacology",
        "🤖 AI Consultation",
        "ℹ️ About",
    ])
    
    st.markdown("---")
    st.markdown('<div style="font-size:0.72rem;color:#4A5568;text-align:center">The Menon Laboratory<br>Perinatal Research</div>', unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────
try:
    data = load_all_data()
except Exception as e:
    st.error(f"⚠️ Data loading error: {e}")
    st.info("Ensure data files are in the `data/` folder.")
    st.stop()

# ══════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div style="padding: 10px 0 20px 0">
            <div class="g-title">Gravitas AI <span style="font-size:1.4rem;color:#63B3ED">2.0</span></div>
            <div class="g-subtitle">Pregnancy Drug Safety Intelligence Platform</div>
            <div class="g-tagline">"It's About Saving Babies" · The Menon Laboratory</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="text-align:right;padding-top:10px">
            <span class="g-version">VALIDATED DATASET</span><br>
            <span style="font-size:0.72rem;color:#4A5568">16 Drugs · Multi-Modal</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Pipeline banner
    st.markdown("""
    <div style="background:#0D1421;border:1px solid #1E3A5F;border-radius:12px;padding:16px 20px;margin-bottom:20px">
        <div style="font-size:0.72rem;color:#718096;text-transform:uppercase;letter-spacing:2px;margin-bottom:10px">Integrated Computational Pipeline</div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
            <span style="background:#2B6CB022;border:1px solid #2B6CB066;border-radius:6px;padding:4px 12px;font-size:0.82rem;color:#63B3ED">🗄️ FDA Library</span>
            <span style="color:#4A5568">→</span>
            <span style="background:#27674922;border:1px solid #27674966;border-radius:6px;padding:4px 12px;font-size:0.82rem;color:#68D391">🧬 ADMET Lab 3.0</span>
            <span style="color:#4A5568">→</span>
            <span style="background:#C5303022;border:1px solid #C5303066;border-radius:6px;padding:4px 12px;font-size:0.82rem;color:#FC8181">☢️ ProTox 3.0</span>
            <span style="color:#4A5568">→</span>
            <span style="background:#B7791F22;border:1px solid #B7791F66;border-radius:6px;padding:4px 12px;font-size:0.82rem;color:#F6AD55">⚗️ AutoDock Vina</span>
            <span style="color:#4A5568">→</span>
            <span style="background:#55309A22;border:1px solid #55309A66;border-radius:6px;padding:4px 12px;font-size:0.82rem;color:#B794F4">💊 GastroPlus PBPK</span>
            <span style="color:#4A5568">→</span>
            <span style="background:#28615E22;border:1px solid #28615E66;border-radius:6px;padding:4px 12px;font-size:0.82rem;color:#4FD1C5">🕸️ Network Pharmacology</span>
            <span style="color:#4A5568">→</span>
            <span style="background:#E8707A22;border:1px solid #E8707A66;border-radius:6px;padding:4px 12px;font-size:0.82rem;color:#E8707A;font-weight:700">🤖 Gravitas AI 2.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    stats = [
        ("16", "Validated Drugs", "#E8707A"),
        ("175", "ADME/Tox Parameters", "#63B3ED"),
        ("6", "Docking Targets", "#F6AD55"),
        ("3", "Trimesters Modelled", "#B794F4"),
        ("27", "DART Endpoints", "#FC8181"),
        ("48", "PBPK Scenarios", "#68D391"),
    ]
    for col, (val, lbl, color) in zip([c1,c2,c3,c4,c5,c6], stats):
        with col:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-val" style="color:{color}">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Audience cards
    st.markdown('<div class="sec-header">🎯 Choose Your Profile</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    for col, (icon, title, color, cls, pts) in zip([col1, col2, col3], [
        ("👩‍⚕️", "Clinician", "#68D391", "green", [
            "Trimester-specific dosing guidance",
            "PBPK-derived fetal exposure",
            "DART reproductive toxicology",
            "Monitoring parameters",
            "Clinical decision framework",
        ]),
        ("🤰", "Patient", "#E8707A", "pink", [
            "Plain-language safety summary",
            "Is it safe in my trimester?",
            "What to discuss with your doctor",
            "Breastfeeding safety",
            "Neonatal risk overview",
        ]),
        ("🔬", "Pharma Researcher", "#63B3ED", "blue", [
            "Full ADME/PK parameter set",
            "Molecular docking affinities",
            "Pregnancy PBPK simulation",
            "Network pharmacology pathways",
            "DART signal confidence",
        ]),
    ]):
        with col:
            st.markdown(f"""
            <div class="g-card g-card-{cls}">
                <div style="font-size:2rem;margin-bottom:8px">{icon}</div>
                <div style="font-size:1.05rem;font-weight:700;color:{color};margin-bottom:10px">{title}</div>
                {"".join(f'<div style="font-size:0.82rem;color:#A0AEC0;padding:3px 0">• {p}</div>' for p in pts)}
            </div>""", unsafe_allow_html=True)
    
    # Quick search
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">⚡ Quick Drug Lookup</div>', unsafe_allow_html=True)
    
    col_s, col_b = st.columns([4, 1])
    with col_s:
        quick_drug = st.selectbox("Select a drug", [""] + data["drugs"], label_visibility="collapsed")
    with col_b:
        quick_go = st.button("🔍 Analyze", use_container_width=True)
    
    if quick_drug and quick_go:
        st.session_state["selected_drug"] = quick_drug
        st.session_state["goto_page"] = "🔍 Drug Analysis"
        st.rerun()
    
    # Drug tiles
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">💊 Drug Library</div>', unsafe_allow_html=True)
    
    cols = st.columns(4)
    for i, drug in enumerate(data["drugs"]):
        row = data["adme"].loc[drug]
        phi = compute_phi(row)
        phi_lev, phi_cls = phi_level(phi)
        dart_sig = data["dart_summary"].loc[drug, "Overall_DART_signal"] if drug in data["dart_summary"].index else "N/A"
        
        color_map = {"high": "#FC8181", "moderate": "#F6AD55", "low": "#68D391"}
        card_color = color_map.get(phi_cls, "#718096")
        
        with cols[i % 4]:
            st.markdown(f"""
            <div class="g-card" style="border-left:4px solid {card_color};cursor:pointer">
                <div style="font-size:0.92rem;font-weight:700;color:#E2E8F0">{drug}</div>
                <div style="display:flex;gap:6px;margin-top:6px;flex-wrap:wrap">
                    <span class="badge badge-{phi_cls}">PHI {phi}</span>
                    <span class="badge badge-{'positive' if dart_sig=='Positive' else 'negative' if dart_sig=='Negative' else 'unknown'}">{dart_sig}</span>
                </div>
                <div style="font-size:0.75rem;color:#4A5568;margin-top:4px">
                    MW {safe(row.get('MW'),1)} · LogP {safe(row.get('logP'),2)}
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: DRUG ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Drug Analysis":
    
    col_sel, col_user = st.columns([3, 2])
    with col_sel:
        default_drug = st.session_state.get("selected_drug", data["drugs"][0])
        drug = st.selectbox("Select Drug", data["drugs"],
            index=data["drugs"].index(default_drug) if default_drug in data["drugs"] else 0)
    with col_user:
        user_type = st.selectbox("View As", ["👩‍⚕️ Clinician", "🤰 Patient", "🔬 Pharma Researcher"])
    
    if drug:
        row = data["adme"].loc[drug]
        phi = compute_phi(row)
        phi_lev, phi_cls = phi_level(phi)
        dart_row = data["dart_summary"].loc[drug] if drug in data["dart_summary"].index else {}
        pbpk_drug = data["pbpk_scenarios"][data["pbpk_scenarios"]["Drug_name"]==drug]
        lit_row = data["pbpk_lit"].loc[drug] if drug in data["pbpk_lit"].index else {}
        
        phi_colors = {"high": "#FC8181", "moderate": "#F6AD55", "low": "#68D391"}
        phi_color = phi_colors.get(phi_cls, "#718096")
        dart_sig = dart_row.get("Overall_DART_signal", "N/A") if len(dart_row) else "N/A"
        
        # Drug header
        st.markdown(f"""
        <div style="background:#0D1421;border:1px solid #1E3A5F;border-radius:12px;padding:20px 24px;margin-bottom:16px">
            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
                <div>
                    <div style="font-size:1.8rem;font-weight:900;color:#E2E8F0;font-family:Georgia">{drug}</div>
                    <div style="font-size:0.82rem;color:#718096;margin-top:2px">
                        MW {safe(row.get('MW'),2)} g/mol · SMILES available · {user_type}
                    </div>
                </div>
                <div style="display:flex;gap:16px;align-items:center">
                    <div class="phi-gauge">
                        <div class="phi-number" style="color:{phi_color}">{phi}</div>
                        <div class="phi-label">Pregnancy<br>Hazard Index</div>
                    </div>
                    <div>
                        <div style="margin-bottom:6px"><span class="badge badge-{phi_cls}" style="font-size:0.88rem;padding:5px 16px">{phi_lev}</span></div>
                        <div><span class="badge badge-{'positive' if dart_sig=='Positive' else 'negative' if dart_sig=='Negative' else 'unknown'}">DART: {dart_sig}</span></div>
                        <div style="margin-top:6px"><span class="badge badge-unknown" style="color:#63B3ED;border-color:#63B3ED44;background:#63B3ED11">{lit_row.get('Status_bucket','N/A') if len(lit_row) else 'N/A'}</span></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 8 TABS
        tabs = st.tabs([
            "📋 Summary", "🧬 ADME", "☢️ Toxicity",
            "⚗️ Docking", "💊 PBPK", "🧪 DART",
            "🕸️ Pathways", "🤖 AI Analysis"
        ])
        
        # ── TAB 1: SUMMARY ────────────────────────────────────
        with tabs[0]:
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown('<div class="sec-header">⚗️ Physicochemical</div>', unsafe_allow_html=True)
                phys_data = [
                    ("Mol. Weight", safe(row.get('MW'),2), "g/mol"),
                    ("LogP", safe(row.get('logP'),3), ""),
                    ("LogS", safe(row.get('logS'),3), ""),
                    ("LogD", safe(row.get('logD'),3), ""),
                    ("TPSA", safe(row.get('TPSA'),1), "Å²"),
                    ("QED", safe(row.get('QED'),3), ""),
                    ("HBD", safe(row.get('nHD'),0), ""),
                    ("HBA", safe(row.get('nHA'),0), ""),
                ]
                html = '<table class="g-table">'
                for lbl, val, unit in phys_data:
                    html += f'<tr><td style="color:#718096">{lbl}</td><td style="color:#E2E8F0;font-weight:600">{val} <span style="color:#4A5568;font-size:0.78rem">{unit}</span></td></tr>'
                html += '</table>'
                st.markdown(html, unsafe_allow_html=True)
            
            with c2:
                st.markdown('<div class="sec-header">💊 Key ADME</div>', unsafe_allow_html=True)
                adme_data = [
                    ("BBB Penetration", safe(row.get('BBB'),3), "high" if float(row.get('BBB',0) or 0) > 0.7 else "low"),
                    ("Plasma Protein Binding", f"{safe(row.get('PPB'),1)}%", ""),
                    ("Fraction Unbound", f"{safe(row.get('Fu'),1)}%", ""),
                    ("P-gp Substrate", safe(row.get('pgp_sub'),3), ""),
                    ("P-gp Inhibitor", safe(row.get('pgp_inh'),3), ""),
                    ("HIA", safe(row.get('hia'),3), ""),
                    ("Oral Bioavail (F30%)", safe(row.get('f30'),3), ""),
                    ("t½", safe(row.get('t0.5'),2), "hr"),
                ]
                html = '<table class="g-table">'
                for lbl, val, cls in adme_data:
                    html += f'<tr><td style="color:#718096">{lbl}</td><td style="color:#E2E8F0;font-weight:600">{val}</td></tr>'
                html += '</table>'
                st.markdown(html, unsafe_allow_html=True)
            
            with c3:
                st.markdown('<div class="sec-header">🚨 Risk Summary</div>', unsafe_allow_html=True)
                risk_items = [
                    ("DILI Risk", row.get('DILI'), 0.3, 0.7),
                    ("hERG Cardiotox", row.get('hERG'), 0.3, 0.7),
                    ("Ames Mutagenicity", row.get('Ames'), 0.3, 0.7),
                    ("NR-ER Activity", row.get('NR-ER'), 0.3, 0.7),
                    ("NR-AR Activity", row.get('NR-AR'), 0.3, 0.7),
                    ("SR-p53 Pathway", row.get('SR-p53'), 0.3, 0.7),
                    ("Carcinogenicity", row.get('Carcinogenicity'), 0.3, 0.7),
                    ("Neurotoxicity", row.get('Neurotoxicity-DI'), 0.3, 0.7),
                ]
                bars_html = ""
                for lbl, val, lo, hi in risk_items:
                    bars_html += prog_bar(lbl, val, 1.0, "#E8707A", invert_color=True)
                st.markdown(bars_html, unsafe_allow_html=True)
            
            # DART + PBPK quick summary
            st.markdown("<br>", unsafe_allow_html=True)
            c4, c5 = st.columns(2)
            
            with c4:
                st.markdown('<div class="sec-header">🧪 DART Signal</div>', unsafe_allow_html=True)
                if len(dart_row):
                    dart_endpoints = {
                        "EFD Available": dart_row.get("EFD_available"),
                        "Embryo-fetal death": dart_row.get("Embryo_fetal_death_or_loss"),
                        "Fetal growth reduction": dart_row.get("Fetal_growth_reduction"),
                        "Skeletal malformation": dart_row.get("Skeletal_malformation"),
                        "Neurobehavioral effect": dart_row.get("Neurobehavioral_or_IQ_effect"),
                        "Neonatal toxicity": dart_row.get("Neonatal_toxicity"),
                    }
                    html = '<table class="g-table">'
                    for lbl, val in dart_endpoints.items():
                        if pd.notna(val) and val == 1:
                            indicator = '<span class="dart-pos">⚠ Positive</span>'
                        elif pd.isna(val):
                            indicator = '<span class="dart-na">—</span>'
                        else:
                            indicator = '<span class="dart-neg">✓ Not reported</span>'
                        html += f'<tr><td style="color:#718096">{lbl}</td><td>{indicator}</td></tr>'
                    html += '</table>'
                    st.markdown(html, unsafe_allow_html=True)
                    summary = dart_row.get("Summary_text","")
                    if summary and str(summary) != "nan":
                        st.markdown(f'<div class="g-card g-card-red" style="font-size:0.82rem;color:#A0AEC0;margin-top:10px">{summary}</div>', unsafe_allow_html=True)
            
            with c5:
                st.markdown('<div class="sec-header">💊 PBPK Exposure</div>', unsafe_allow_html=True)
                if not pbpk_drug.empty:
                    html = '<table class="g-table"><tr><th>Trimester</th><th>Maternal Cavg (mg/L)</th><th>Fetal Cavg (mg/L)</th><th>Flag</th></tr>'
                    for _, pr in pbpk_drug.iterrows():
                        flag = pr.get('Exposure_flag', 'N/A')
                        flag_color = "#FC8181" if "Above" in str(flag) else "#68D391"
                        html += f"""<tr>
                            <td style="color:#B794F4;font-weight:600">{pr['Trimester']}</td>
                            <td>{safe(pr.get('Maternal_Cavg_mg_L'),5)}</td>
                            <td style="color:#63B3ED">{safe(pr.get('Fetal_plasma_Cavg_mg_L'),5)}</td>
                            <td style="color:{flag_color};font-size:0.78rem">{flag}</td>
                        </tr>"""
                    html += '</table>'
                    st.markdown(html, unsafe_allow_html=True)
        
        # ── TAB 2: ADME ───────────────────────────────────────
        with tabs[1]:
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown('<div class="sec-header">🔬 Absorption & Distribution</div>', unsafe_allow_html=True)
                abs_bars = [
                    ("HIA", row.get('hia'), 1.0),
                    ("Oral F (20%)", row.get('f20'), 1.0),
                    ("Oral F (30%)", row.get('f30'), 1.0),
                    ("Oral F (50%)", row.get('f50'), 1.0),
                    ("Caco-2 (norm)", None, 1.0),
                    ("PAMPA", row.get('PAMPA'), 1.0),
                    ("BBB Penetration", row.get('BBB'), 1.0),
                    ("PPB / 100", row.get('PPB'), 100.0),
                ]
                html = ""
                for lbl, val, mx in abs_bars:
                    if val is not None:
                        html += prog_bar(lbl, val, mx, "#63B3ED", invert_color=False)
                st.markdown(html, unsafe_allow_html=True)
                
                st.markdown('<div class="sec-header" style="margin-top:16px">🔄 Transporters</div>', unsafe_allow_html=True)
                trans_data = [
                    ("P-gp Substrate", row.get('pgp_sub')),
                    ("P-gp Inhibitor", row.get('pgp_inh')),
                    ("OATP1B1", row.get('OATP1B1')),
                    ("OATP1B3", row.get('OATP1B3')),
                    ("BCRP", row.get('BCRP')),
                    ("MRP1", row.get('MRP1')),
                    ("BSEP", row.get('BSEP')),
                ]
                html = '<table class="g-table">'
                for lbl, val in trans_data:
                    html += f'<tr><td style="color:#718096">{lbl}</td><td>{color_val(val)}</td></tr>'
                html += '</table>'
                st.markdown(html, unsafe_allow_html=True)
            
            with c2:
                st.markdown('<div class="sec-header">🔁 CYP Metabolism (Inhibition probability)</div>', unsafe_allow_html=True)
                cyp_bars = [
                    ("CYP1A2-inh", row.get('CYP1A2-inh')),
                    ("CYP2C19-inh", row.get('CYP2C19-inh')),
                    ("CYP2C9-inh", row.get('CYP2C9-inh')),
                    ("CYP2D6-inh", row.get('CYP2D6-inh')),
                    ("CYP3A4-inh", row.get('CYP3A4-inh')),
                    ("CYP2B6-inh", row.get('CYP2B6-inh')),
                    ("CYP2C8-inh", row.get('CYP2C8-inh')),
                ]
                html = ""
                for lbl, val in cyp_bars:
                    html += prog_bar(lbl, val, 1.0, "#B794F4", invert_color=True)
                st.markdown(html, unsafe_allow_html=True)
                
                st.markdown('<div class="sec-header" style="margin-top:16px">📊 CYP Substrate Status</div>', unsafe_allow_html=True)
                cyp_sub = [
                    ("CYP1A2-sub", row.get('CYP1A2-sub')),
                    ("CYP2C19-sub", row.get('CYP2C19-sub')),
                    ("CYP2C9-sub", row.get('CYP2C9-sub')),
                    ("CYP2D6-sub", row.get('CYP2D6-sub')),
                    ("CYP3A4-sub", row.get('CYP3A4-sub')),
                    ("CYP2B6-sub", row.get('CYP2B6-sub')),
                ]
                html = '<table class="g-table">'
                for lbl, val in cyp_sub:
                    html += f'<tr><td style="color:#718096">{lbl}</td><td>{color_val(val)}</td></tr>'
                html += '</table>'
                st.markdown(html, unsafe_allow_html=True)
        
        # ── TAB 3: TOXICITY ───────────────────────────────────
        with tabs[2]:
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown('<div class="sec-header">🔴 Core Toxicity</div>', unsafe_allow_html=True)
                tox_items = [
                    ("DILI", row.get('DILI')), ("hERG Cardiac", row.get('hERG')),
                    ("Ames Mutagenicity", row.get('Ames')), ("H-HT", row.get('H-HT')),
                    ("Carcinogenicity", row.get('Carcinogenicity')),
                    ("Genotoxicity", row.get('Genotoxicity')),
                    ("Respiratory", row.get('Respiratory')),
                    ("Skin Sensitization", row.get('SkinSen')),
                ]
                html = ""
                for lbl, val in tox_items:
                    html += prog_bar(lbl, val, 1.0, "#FC8181", invert_color=True)
                st.markdown(html, unsafe_allow_html=True)
            
            with c2:
                st.markdown('<div class="sec-header">🟡 Organ Toxicity</div>', unsafe_allow_html=True)
                organ_tox = [
                    ("Neurotoxicity", row.get('Neurotoxicity-DI')),
                    ("Nephrotoxicity", row.get('Nephrotoxicity-DI')),
                    ("Hematotoxicity", row.get('Hematotoxicity')),
                    ("Ototoxicity", row.get('Ototoxicity')),
                    ("dili (PTox)", row.get('dili')),
                    ("neuro (PTox)", row.get('neuro')),
                    ("nephro (PTox)", row.get('nephro')),
                    ("cardio (PTox)", row.get('cardio')),
                    ("immuno (PTox)", row.get('immuno')),
                ]
                html = ""
                for lbl, val in organ_tox:
                    html += prog_bar(lbl, val, 1.0, "#F6AD55", invert_color=True)
                st.markdown(html, unsafe_allow_html=True)
            
            with c3:
                st.markdown('<div class="sec-header">🟣 Nuclear Receptors & Stress Response</div>', unsafe_allow_html=True)
                nr_items = [
                    ("NR-AhR", row.get('NR-AhR')), ("NR-AR", row.get('NR-AR')),
                    ("NR-AR-LBD", row.get('NR-AR-LBD')), ("NR-Aromatase", row.get('NR-Aromatase')),
                    ("NR-ER", row.get('NR-ER')), ("NR-ER-LBD", row.get('NR-ER-LBD')),
                    ("NR-PPAR-γ", row.get('NR-PPAR-gamma')),
                    ("SR-ARE", row.get('SR-ARE')), ("SR-p53", row.get('SR-p53')),
                    ("SR-HSE", row.get('SR-HSE')), ("SR-MMP", row.get('SR-MMP')),
                    ("SR-ATAD5", row.get('SR-ATAD5')),
                ]
                html = ""
                for lbl, val in nr_items:
                    html += prog_bar(lbl, val, 1.0, "#B794F4", invert_color=True)
                st.markdown(html, unsafe_allow_html=True)
        
        # ── TAB 4: DOCKING ────────────────────────────────────
        with tabs[3]:
            st.markdown('<div class="sec-header">⚗️ Molecular Docking — PTB Inflammatory Targets (kcal/mol)</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.82rem;color:#718096;margin-bottom:16px">More negative = stronger binding affinity. Threshold: strong &lt;-9.0 · moderate -7.0 to -9.0 · weak &gt;-7.0</div>', unsafe_allow_html=True)
            
            dock_targets = {
                "P38-MAPK": ("P-38", "P38 mitogen-activated protein kinase — key mediator of inflammatory cytokine production in preterm birth"),
                "NF-κB": ("NFKB", "Nuclear factor kappa-B — master regulator of inflammatory gene expression, critical in PTB pathology"),
                "MAPK": ("MAPK", "Mitogen-activated protein kinase cascade — regulates prostaglandin synthesis and uterine contractility"),
                "JAK2": ("JAK2", "Janus kinase 2 — mediates IL-6 and cytokine signaling at the feto-maternal interface"),
                "TGFβR1": ("TGFBR1", "TGF-beta receptor 1 — regulates placental development and immune tolerance in pregnancy"),
                "HIF1α": ("HIF1A", "Hypoxia-inducible factor 1-alpha — regulates placental angiogenesis and oxygen sensing"),
            }
            
            cols = st.columns(3)
            for i, (name, (col_key, desc)) in enumerate(dock_targets.items()):
                val = row.get(col_key)
                try:
                    v = float(val)
                    if v <= -9.0: color, strength = "#FC8181", "Strong"
                    elif v <= -7.0: color, strength = "#F6AD55", "Moderate"
                    elif v <= -5.0: color, strength = "#68D391", "Weak"
                    else: color, strength = "#718096", "Very Weak"
                    display = f"{v:.1f}"
                except:
                    color, strength, display = "#4A5568", "N/A", "N/A"
                
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="g-card" style="border-left:4px solid {color};margin-bottom:12px">
                        <div style="font-size:1.1rem;font-weight:700;color:{color}">{name}</div>
                        <div style="font-size:2rem;font-weight:900;color:{color};margin:6px 0">{display} <span style="font-size:1rem">kcal/mol</span></div>
                        <div style="font-size:0.78rem;color:#718096;margin-bottom:6px">{strength} binding</div>
                        <div style="font-size:0.75rem;color:#4A5568;line-height:1.4">{desc}</div>
                    </div>""", unsafe_allow_html=True)
            
            # CYP docking
            st.markdown('<div class="sec-header" style="margin-top:20px">🔁 CYP Enzyme Docking Scores</div>', unsafe_allow_html=True)
            cyp_dock = [("CYP1A2","CYP1A2"),("CYP2C19","CYP2C19"),("CYP2C9","CYP2C9"),
                        ("CYP2D6","CYP2D6"),("CYP3A4","CYP3A4"),("CYP2E1","CYP2E1")]
            cols = st.columns(6)
            for col, (name, key) in zip(cols, cyp_dock):
                val = row.get(key)
                with col:
                    try:
                        v = float(val)
                        color = "#FC8181" if v <= -9 else "#F6AD55" if v <= -7 else "#68D391"
                        st.markdown(f'<div class="dock-card"><div class="dock-val" style="color:{color}">{v:.2f}</div><div class="dock-lbl">{name}</div></div>', unsafe_allow_html=True)
                    except:
                        st.markdown(f'<div class="dock-card"><div class="dock-val" style="color:#4A5568">N/A</div><div class="dock-lbl">{name}</div></div>', unsafe_allow_html=True)
        
        # ── TAB 5: PBPK ───────────────────────────────────────
        with tabs[4]:
            st.markdown('<div class="sec-header">💊 Pregnancy PBPK — Trimester Exposure Modelling</div>', unsafe_allow_html=True)
            
            if lit_row is not None and len(lit_row):
                status = lit_row.get('Status_bucket','')
                note = lit_row.get('Key_quantitative_note','')
                color_map_s = {
                    "Pregnancy PBPK available": "#68D391",
                    "Pregnancy maternal-fetal PBPK available": "#68D391",
                    "Observed pregnancy popPK available": "#63B3ED",
                    "Developmental tox PK/IVIVE only": "#F6AD55",
                    "Scaffold only": "#FC8181",
                }
                s_color = color_map_s.get(str(status), "#718096")
                st.markdown(f"""
                <div class="g-card" style="border-left:4px solid {s_color};margin-bottom:16px">
                    <span class="badge" style="color:{s_color};border-color:{s_color}44;background:{s_color}11">{status}</span>
                    <div style="font-size:0.85rem;color:#A0AEC0;margin-top:8px">{note}</div>
                </div>""", unsafe_allow_html=True)
            
            if not pbpk_drug.empty:
                # Trimester comparison table
                html = '<table class="g-table"><tr><th>Trimester</th><th>Wk</th><th>Dose (mg/d)</th><th>CL preg (L/hr)</th><th>Vd preg (L)</th><th>Maternal AUC24</th><th>Maternal Cavg</th><th>Maternal Unbound</th><th>Fetal Plasma</th><th>Fetal Tissue</th><th>Exposure Flag</th><th>DART Signal</th></tr>'
                tri_colors = {"T1":"#B794F4","T2":"#63B3ED","T3":"#68D391"}
                for _, pr in pbpk_drug.iterrows():
                    tri = pr['Trimester']
                    tc = tri_colors.get(tri,"#718096")
                    flag = str(pr.get('Exposure_flag',''))
                    flag_c = "#FC8181" if "Above" in flag else "#68D391"
                    dart_s = str(pr.get('DART_signal',''))
                    dart_c = "#FC8181" if dart_s=="Positive" else "#68D391"
                    html += f"""<tr>
                        <td style="color:{tc};font-weight:700">{tri}</td>
                        <td style="color:#718096">{data['pbpk_physio'].loc[data['pbpk_physio']['Trimester']==tri,'Gestational_week_midpoint'].values[0] if tri in data['pbpk_physio']['Trimester'].values else '—'}</td>
                        <td>{safe(pr.get('Dose_used_mg_day'),1)}</td>
                        <td>{safe(pr.get('CL_preg_L_hr'),3)}</td>
                        <td>{safe(pr.get('Vd_preg_L'),2)}</td>
                        <td style="color:#E2E8F0">{safe(pr.get('Maternal_AUC24_mg*h_L'),4)}</td>
                        <td style="color:#E2E8F0">{safe(pr.get('Maternal_Cavg_mg_L'),5)}</td>
                        <td>{safe(pr.get('Maternal_unbound_Cavg_mg_L'),5)}</td>
                        <td style="color:#63B3ED;font-weight:600">{safe(pr.get('Fetal_plasma_Cavg_mg_L'),5)}</td>
                        <td style="color:#4FD1C5">{safe(pr.get('Fetal_tissue_Cavg_mg_L'),5)}</td>
                        <td style="color:{flag_c};font-size:0.78rem">{flag}</td>
                        <td style="color:{dart_c}">{dart_s}</td>
                    </tr>"""
                html += '</table>'
                st.markdown(html, unsafe_allow_html=True)
                
                # Physiology defaults
                st.markdown('<div class="sec-header" style="margin-top:20px">🤰 Pregnancy Physiological Parameters Used</div>', unsafe_allow_html=True)
                phys_df = data["pbpk_physio"][["Trimester","Gestational_week_midpoint","Body_weight_kg",
                    "Plasma_volume_multiplier","Albumin_multiplier","GFR_multiplier",
                    "Hepatic_clearance_multiplier","Placental_transfer_multiplier","Fetal_weight_kg"]]
                html = '<table class="g-table"><tr>'
                for c in phys_df.columns:
                    html += f'<th>{c}</th>'
                html += '</tr>'
                for _, pr in phys_df.iterrows():
                    tc = tri_colors.get(pr['Trimester'],"#718096")
                    html += f'<tr><td style="color:{tc};font-weight:700">{pr["Trimester"]}</td>'
                    for c in phys_df.columns[1:]:
                        html += f'<td>{safe(pr[c],2)}</td>'
                    html += '</tr>'
                html += '</table>'
                st.markdown(html, unsafe_allow_html=True)
        
        # ── TAB 6: DART ───────────────────────────────────────
        with tabs[5]:
            st.markdown('<div class="sec-header">🧪 Developmental & Reproductive Toxicology (DART)</div>', unsafe_allow_html=True)
            
            if len(dart_row):
                # Summary card
                overall = dart_row.get("Overall_DART_signal","N/A")
                conf = dart_row.get("Evidence_confidence","N/A")
                overall_c = "#FC8181" if overall=="Positive" else "#68D391" if overall=="Negative" else "#F6AD55"
                
                st.markdown(f"""
                <div class="g-card g-card-red" style="margin-bottom:16px">
                    <div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap">
                        <div>
                            <div style="font-size:0.72rem;color:#718096;text-transform:uppercase">Overall DART Signal</div>
                            <div style="font-size:1.5rem;font-weight:700;color:{overall_c}">{overall}</div>
                        </div>
                        <div>
                            <div style="font-size:0.72rem;color:#718096;text-transform:uppercase">Evidence Confidence</div>
                            <div style="font-size:1.2rem;font-weight:600;color:#F6AD55">{conf}</div>
                        </div>
                        <div style="flex:1">
                            <div style="font-size:0.72rem;color:#718096;text-transform:uppercase;margin-bottom:4px">Summary</div>
                            <div style="font-size:0.85rem;color:#A0AEC0">{dart_row.get('Summary_text','N/A')}</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
                
                # Endpoint grid
                endpoint_groups = {
                    "Fertility": ["Male_fertility_signal","Female_fertility_signal"],
                    "Embryo-Fetal": ["Embryo_fetal_death_or_loss","Implantation_loss","Resorptions","Fetal_growth_reduction"],
                    "Structural": ["Skeletal_variation_or_delayed_ossification","Skeletal_malformation","Visceral_malformation","External_malformation"],
                    "Postnatal": ["Postnatal_survival_decrease","Developmental_delay","Neurobehavioral_or_IQ_effect","Neonatal_toxicity"],
                    "Maternal": ["Maternal_toxicity_reported"],
                }
                
                cols = st.columns(len(endpoint_groups))
                for col, (group, endpoints) in zip(cols, endpoint_groups.items()):
                    with col:
                        st.markdown(f'<div style="font-size:0.78rem;font-weight:700;color:#63B3ED;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">{group}</div>', unsafe_allow_html=True)
                        for ep in endpoints:
                            val = dart_row.get(ep)
                            clean = ep.replace("_", " ").replace(" signal","").replace(" available","").replace(" or ", "/")
                            if pd.notna(val) and val == 1:
                                icon, cls = "⚠", "dart-pos"
                            elif pd.isna(val):
                                icon, cls = "—", "dart-na"
                            else:
                                icon, cls = "✓", "dart-neg"
                            st.markdown(f'<div style="font-size:0.8rem;padding:3px 0"><span class="{cls}">{icon}</span> <span style="color:#A0AEC0">{clean}</span></div>', unsafe_allow_html=True)
                
                # Raw evidence
                st.markdown('<div class="sec-header" style="margin-top:20px">📋 Raw Study Evidence</div>', unsafe_allow_html=True)
                drug_evidence = data["dart_evidence"][data["dart_evidence"]["Drug_name"]==drug]
                if not drug_evidence.empty:
                    show_cols = ["Study_type","Species","Route","Dose","Endpoint_group","Endpoint_term","Result_code","NOAEL","LOAEL"]
                    html = '<table class="g-table"><tr>'
                    for c in show_cols: html += f'<th>{c}</th>'
                    html += '</tr>'
                    for _, er in drug_evidence.iterrows():
                        rc = str(er.get("Result_code",""))
                        rc_color = "#FC8181" if rc in ["Positive","Adverse"] else "#68D391" if rc=="Negative" else "#F6AD55"
                        html += '<tr>'
                        for c in show_cols:
                            v = er.get(c,"—")
                            v = "—" if pd.isna(v) else str(v)
                            color_attr = f'style="color:{rc_color}"' if c=="Result_code" else ""
                            html += f'<td {color_attr}>{v}</td>'
                        html += '</tr>'
                    html += '</table>'
                    st.markdown(html, unsafe_allow_html=True)
        
        # ── TAB 7: PATHWAYS ───────────────────────────────────
        with tabs[6]:
            st.markdown('<div class="sec-header">🕸️ Network Pharmacology — Upload Drug Data</div>', unsafe_allow_html=True)
            
            net_data = load_network_data(drug)
            figures = net_data.pop('figures', {})
            
            if net_data or figures:
                avail = [k for k in net_data.keys() if k != "figures"]
                st.success(f"✅ Network pharmacology data found: {', '.join(avail)}")
                
                sub_tabs = st.tabs(["KEGG Pathways","Reactome","GO-BP","GO-CC","GO-MF","Hub Genes"])
                
                for sub_tab, key, title in zip(sub_tabs,
                    ["kegg","reactome","go_bp","go_cc","go_mf","hub_genes"],
                    ["KEGG","Reactome","GO Biological Process","GO Cellular Component","GO Molecular Function","Hub Genes"]):
                    with sub_tab:
                        if key in net_data:
                            df_net = net_data[key]
                            st.dataframe(df_net.head(20), use_container_width=True)
                        else:
                            st.info(f"Place {drug}_{title.replace(' ','_')}.xlsx in {drug}/Network_Pharmacology/")
            else:
                st.markdown(f"""
                <div class="g-card g-card-teal">
                    <div style="font-size:1rem;font-weight:700;color:#4FD1C5;margin-bottom:10px">📂 Upload Network Pharmacology Files for {drug}</div>
                    <div style="font-size:0.85rem;color:#A0AEC0;margin-bottom:12px">
                        Place files in: <code style="background:#111827;padding:2px 8px;border-radius:4px">data/network_pharmacology/{drug}/</code>
                    </div>
                    <div style="font-size:0.82rem;color:#718096">
                        Expected files:<br>
                        • {drug}_KEGG.xlsx &nbsp;•&nbsp; {drug}_Reactome.xlsx<br>
                        • {drug}_GO_BP.xlsx &nbsp;•&nbsp; {drug}_GO_CC.xlsx &nbsp;•&nbsp; {drug}_GO_MF.xlsx<br>
                        • {drug}_Hubgenes_STRING_centrality.xlsx &nbsp;•&nbsp; Commongenes_Venn.xlsx
                    </div>
                </div>""", unsafe_allow_html=True)
                
                # Show docking as proxy for pathway relevance
                st.markdown('<div class="sec-header" style="margin-top:20px">🔗 PTB Pathway Target Binding (from Docking)</div>', unsafe_allow_html=True)
                pathway_targets = {
                    "Inflammatory Cascade": [("NF-κB",row.get("NFKB")),("P38-MAPK",row.get("P-38")),("MAPK",row.get("MAPK"))],
                    "Cytokine Signaling": [("JAK2",row.get("JAK2")),("TGFβR1",row.get("TGFBR1"))],
                    "Hypoxia Response": [("HIF1α",row.get("HIF1A"))],
                    "CYP Metabolism": [("CYP3A4",row.get("CYP3A4")),("CYP2C9",row.get("CYP2C9")),("CYP1A2",row.get("CYP1A2"))],
                }
                cols = st.columns(2)
                for i, (pathway, targets) in enumerate(pathway_targets.items()):
                    with cols[i % 2]:
                        st.markdown(f'<div class="g-card g-card-teal"><div style="font-size:0.88rem;font-weight:700;color:#4FD1C5;margin-bottom:8px">{pathway}</div>', unsafe_allow_html=True)
                        for tname, tval in targets:
                            try:
                                v = float(tval)
                                color = "#FC8181" if v<=-9 else "#F6AD55" if v<=-7 else "#68D391"
                                st.markdown(f'<div style="font-size:0.82rem;padding:2px 0"><span style="color:{color};font-weight:600">{tname}: {v:.1f} kcal/mol</span></div>', unsafe_allow_html=True)
                            except:
                                st.markdown(f'<div style="font-size:0.82rem;padding:2px 0;color:#4A5568">{tname}: N/A</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # ── TAB 8: AI ANALYSIS ────────────────────────────────
        with tabs[7]:
            st.markdown('<div class="sec-header">🤖 AI Clinical Intelligence</div>', unsafe_allow_html=True)
            
            col_q, col_b = st.columns([4, 1])
            with col_q:
                custom_q = st.text_input(
                    "Ask a specific question (optional)",
                    placeholder=f"e.g. What is the recommended dose of {drug} in the third trimester?",
                    key=f"q_{drug}"
                )
            with col_b:
                st.markdown("<br>", unsafe_allow_html=True)
                run_ai = st.button("🔍 Analyze", key=f"ai_{drug}", use_container_width=True)
            
            if run_ai or custom_q:
                with st.spinner(f"Analyzing {drug} for {user_type}..."):
                    response = ai_analyze(drug, data, user_type, custom_q)
                st.markdown(f'<div class="ai-bubble"><span class="ai-icon">🤖</span>{response}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="g-card g-card-gold">
                    <div style="font-size:0.92rem;color:#F6AD55;font-weight:600;margin-bottom:8px">AI Analysis Ready</div>
                    <div style="font-size:0.85rem;color:#A0AEC0">
                        Click "Analyze" to get a {user_type}-specific assessment of <strong>{drug}</strong> integrating all pipeline data:
                        ADME parameters, toxicity profile, molecular docking scores, DART studies, and PBPK exposure predictions.
                    </div>
                </div>""", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="disclaimer">
            ⚠️ <strong>Disclaimer:</strong> Gravitas AI 2.0 is a research and information tool only. 
            All medication decisions during pregnancy must be made in consultation with a qualified healthcare provider. 
            This platform does not provide clinical recommendations and is not a substitute for professional medical judgment.
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: DATABASE EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "📊 Database Explorer":
    st.markdown('<div class="g-title" style="margin-bottom:16px">📊 Database Explorer</div>', unsafe_allow_html=True)
    
    # Build summary table
    rows = []
    for drug in data["drugs"]:
        row = data["adme"].loc[drug]
        dart = data["dart_summary"].loc[drug] if drug in data["dart_summary"].index else {}
        phi = compute_phi(row)
        phi_lev, _ = phi_level(phi)
        pbpk_d = data["pbpk_scenarios"][data["pbpk_scenarios"]["Drug_name"]==drug]
        fetal_t1 = pbpk_d[pbpk_d["Trimester"]=="T1"]["Fetal_plasma_Cavg_mg_L"].values
        
        rows.append({
            "Drug": drug,
            "PHI": phi,
            "Risk Level": phi_lev,
            "DART Signal": dart.get("Overall_DART_signal","N/A") if len(dart) else "N/A",
            "MW": round(float(row.get("MW",0) or 0),2),
            "LogP": round(float(row.get("logP",0) or 0),3),
            "TPSA": round(float(row.get("TPSA",0) or 0),1),
            "DILI": round(float(row.get("DILI",0) or 0),3),
            "hERG": round(float(row.get("hERG",0) or 0),3),
            "NR-ER": round(float(row.get("NR-ER",0) or 0),4),
            "NF-κB dock": row.get("NFKB"),
            "TGFβR1 dock": row.get("TGFBR1"),
            "Fetal Cavg T1": round(float(fetal_t1[0]),5) if len(fetal_t1) else None,
            "BBB": round(float(row.get("BBB",0) or 0),3),
            "PPB%": round(float(row.get("PPB",0) or 0),1),
        })
    
    summary_df = pd.DataFrame(rows)
    
    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        risk_filter = st.multiselect("Risk Level", ["HIGH RISK","MODERATE","LOW RISK"], default=["HIGH RISK","MODERATE","LOW RISK"])
    with c2:
        dart_filter = st.multiselect("DART Signal", ["Positive","Negative","Equivocal","N/A"], default=["Positive","Negative","Equivocal","N/A"])
    with c3:
        phi_range = st.slider("PHI Range", 0, 100, (0, 100))
    
    filtered = summary_df[
        (summary_df["Risk Level"].isin(risk_filter)) &
        (summary_df["DART Signal"].isin(dart_filter)) &
        (summary_df["PHI"] >= phi_range[0]) & (summary_df["PHI"] <= phi_range[1])
    ]
    
    st.markdown(f'<div style="font-size:0.82rem;color:#718096;margin-bottom:8px">Showing {len(filtered)} of {len(summary_df)} drugs</div>', unsafe_allow_html=True)
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    
    csv = filtered.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "gravitas2_database.csv", "text/csv")

# ══════════════════════════════════════════════════════════════
# PAGE: NETWORK PHARMACOLOGY
# ══════════════════════════════════════════════════════════════
elif page == "🕸️ Network Pharmacology":
    st.markdown('<div class="g-title" style="margin-bottom:6px">🕸️ Network Pharmacology</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#718096;font-size:0.88rem;margin-bottom:16px">KEGG · Reactome · GO Analysis · Hub Genes · STRING centrality</div>', unsafe_allow_html=True)
    
    drug = st.selectbox("Select Drug", data["drugs"])
    
    net_data = load_network_data(drug)
    figures = net_data.pop("figures", {})
    table_keys = ["kegg","reactome","go_bp","go_cc","go_mf","hub_genes","common_genes"]
    has_tables = any(k in net_data for k in table_keys)

    if has_tables or figures:
        sub_tabs = st.tabs(["📊 KEGG","🔬 Reactome","🟢 GO-BP","🔵 GO-CC","🟡 GO-MF","🔗 Hub Genes","🔶 Common Genes","🖼️ Figures"])
        tab_keys = [("kegg","KEGG"),("reactome","Reactome"),("go_bp","GO Biological Process"),
                    ("go_cc","GO Cellular Component"),("go_mf","GO Molecular Function"),
                    ("hub_genes","Hub Genes (STRING)"),("common_genes","Common Genes (Venn)")]
        for tab, (key, title) in zip(sub_tabs[:-1], tab_keys):
            with tab:
                if key in net_data:
                    df_n = net_data[key]
                    st.markdown(f'<div class="sec-header">{title} — {drug} ({len(df_n)} entries)</div>', unsafe_allow_html=True)
                    st.dataframe(df_n, use_container_width=True)
                    csv = df_n.to_csv(index=False)
                    st.download_button(f"📥 Download {title}", csv, f"{drug}_{key}.csv", key=f"dl_{key}")
                else:
                    st.info(f"📂 Place {drug}_{key}.xlsx in {drug}/Network_Pharmacology/")
        with sub_tabs[-1]:
            fig_labels = {
                "fig_dotplot_bp": "GO-BP Dot Plot", "fig_dotplot_cc": "GO-CC Dot Plot",
                "fig_dotplot_mf": "GO-MF Dot Plot", "fig_enrichmap_kegg": "KEGG Enrichment Map",
                "fig_enrichmap_reactome": "Reactome Enrichment Map",
            }
            if figures:
                cols = st.columns(2)
                for i, (key, label) in enumerate(fig_labels.items()):
                    if key in figures:
                        with cols[i % 2]:
                            st.markdown(f'<div class="sec-header">{label}</div>', unsafe_allow_html=True)
                            st.image(figures[key], use_container_width=True)
            else:
                st.info(f"📂 Place PNG/JPG figures in {drug}/Network_Pharmacology/")
    else:
        st.markdown(f"""
        <div class="g-card g-card-teal">
            <div style="font-size:1rem;font-weight:700;color:#4FD1C5;margin-bottom:12px">📂 Expected Folder Structure</div>
            <div style="font-size:0.88rem;color:#A0AEC0;line-height:2.0">
                <code style="background:#111827;padding:2px 8px;border-radius:4px">gravitas2/{drug}/Network_Pharmacology/</code><br><br>
                Tables: <code>{drug}_KEGG.xlsx</code>, <code>{drug}_Reactome.xlsx</code>,
                <code>{drug}_GO_BP.xlsx</code>, <code>{drug}_GO_CC.xlsx</code>, <code>{drug}_GO_MF.xlsx</code>,
                <code>{drug}_Hubgenes_STRING_centrality.xlsx</code>, <code>Commongenes_Venn.xlsx</code><br><br>
                Figures (PNG/JPG): <code>{drug}_EnrichmentMap_KEGG.png</code>,
                <code>{drug}_EnrichmentMap_Reactome.png</code>, <code>{drug}_Dotplot_GO_BP.png</code>,
                <code>{drug}_Dotplot_GO_CC.png</code>, <code>{drug}_Dotplot_GO_MF.png</code>
            </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: AI CONSULTATION
# ══════════════════════════════════════════════════════════════
elif page == "🤖 AI Consultation":
    st.markdown('<div class="g-title" style="margin-bottom:6px">🤖 AI Consultation</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#718096;font-size:0.88rem;margin-bottom:20px">Multi-drug comparison · Custom questions · Profile-specific analysis</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns([2, 1])
    with c1:
        selected_drugs = st.multiselect("Select drugs to analyze", data["drugs"], max_selections=4)
    with c2:
        user_type = st.selectbox("View As", ["👩‍⚕️ Clinician", "🤰 Patient", "🔬 Pharma Researcher"])
    
    question = st.text_area("Your question", 
        placeholder="e.g. Compare fetal exposure risk between Aspirin and Indomethacin in T3. Which is safer?",
        height=90)
    
    if st.button("🔍 Run AI Analysis", use_container_width=True) and selected_drugs:
        if len(selected_drugs) == 1:
            with st.spinner("Analyzing..."):
                resp = ai_analyze(selected_drugs[0], data, user_type, question)
            st.markdown(f'<div class="ai-bubble">{resp}</div>', unsafe_allow_html=True)
        else:
            for drug in selected_drugs:
                st.markdown(f'<div class="sec-header">📋 {drug}</div>', unsafe_allow_html=True)
                with st.spinner(f"Analyzing {drug}..."):
                    resp = ai_analyze(drug, data, user_type, question or f"Provide a {user_type} analysis focusing on pregnancy safety.")
                st.markdown(f'<div class="ai-bubble">{resp}</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="disclaimer">
    ⚠️ Gravitas AI 2.0 is a research tool only. Not for clinical decision-making without professional consultation.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("""
    <div style="text-align:center;padding:30px 0">
        <div class="g-title">Gravitas AI <span style="font-size:1.4rem;color:#63B3ED">2.0</span></div>
        <div class="g-subtitle">Pregnancy Drug Safety Intelligence Platform</div>
        <div class="g-tagline" style="font-size:1rem;margin-top:8px">"It's About Saving Babies"</div>
        <div style="margin-top:12px"><span class="g-version">VERSION 2.0 · VALIDATED MULTI-MODAL DATASET</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="g-card g-card-blue">
            <div class="sec-header">🔬 The Pipeline</div>
            <div style="font-size:0.85rem;color:#A0AEC0;line-height:1.8">
                <strong style="color:#63B3ED">1. FDA Drug Library</strong> — 16 curated compounds<br>
                <strong style="color:#68D391">2. ADMET Lab 3.0</strong> — 175 ADME parameters per compound<br>
                <strong style="color:#FC8181">3. ProTox 3.0</strong> — 50+ toxicity endpoints<br>
                <strong style="color:#F6AD55">4. AutoDock Vina</strong> — Docking vs 6 PTB inflammatory proteins<br>
                <strong style="color:#B794F4">5. GastroPlus</strong> — Pregnancy PBPK (T1/T2/T3)<br>
                <strong style="color:#4FD1C5">6. Network Pharmacology</strong> — KEGG, Reactome, GO, STRING hubs<br>
                <strong style="color:#E8707A">7. Gravitas AI 2.0</strong> — Integrated RAG-powered platform
            </div>
        </div>""", unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div class="g-card g-card-pink">
            <div class="sec-header">🎯 Designed For</div>
            <div style="font-size:0.85rem;color:#A0AEC0;line-height:2.0">
                <strong style="color:#E8707A">👩‍⚕️ Clinicians</strong> — Dosing guidance, fetal exposure, monitoring<br>
                <strong style="color:#E8707A">🤰 Patients</strong> — Plain-language safety, trimester risk<br>
                <strong style="color:#E8707A">🔬 Pharma Researchers</strong> — Full ADME/PK, docking, PBPK, pathways<br><br>
                <strong style="color:#F6AD55">PHI Formula:</strong><br>
                <code style="background:#111827;padding:4px 8px;border-radius:4px;font-size:0.8rem">
                DILI(0.25) + hERG(0.15) + Ames(0.10) +<br>NR-ER(0.15) + NR-AR(0.10) + SR-p53(0.10) + BBB(0.15)
                </code>
            </div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="disclaimer" style="text-align:center;margin-top:20px">
    ⚠️ For research purposes only. Not a clinical decision support system. 
    Always consult a qualified healthcare provider for medical decisions during pregnancy.
    </div>""", unsafe_allow_html=True)
