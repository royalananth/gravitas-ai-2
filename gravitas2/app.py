"""
Gravitas AI 2.0 — The Menon Laboratory
Pregnancy Drug Safety Intelligence Platform
"It's About Saving Babies"
"""

import streamlit as st
import anthropic
import json
import pandas as pd
import os
import base64

st.set_page_config(
    page_title="Gravitas AI · The Menon Laboratory",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── LOGO ──────────────────────────────────────────────────────────
# Place your logo as "logo.png" (or logo.jpg / logo.jpeg) next to this
# file in the repo. If none is found, the app still runs without a logo.
def _load_logo():
    base = os.path.dirname(os.path.abspath(__file__))
    for name in ["logo.png", "logo.jpg", "logo.jpeg", "logo.webp"]:
        p = os.path.join(base, name)
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            except Exception:
                pass
    return ""

LOGO_B64 = _load_logo()

# Model used for all Anthropic API calls (single place to update)
CLAUDE_MODEL = "claude-sonnet-4-6"

# ── CSS — V1 visual style ─────────────────────────────────────────
st.markdown("""
<style>
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}

html, body, [class*="css"] {
    background-color: #080c14; color: #e2e8f0;
    font-family: 'Segoe UI', sans-serif;
}
.main { background-color: #080c14; }
section[data-testid="stSidebar"] {
    background-color: #0d1421;
    border-right: 1px solid rgba(99,179,237,0.15);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Hero (v1 style) ── */
.hero-wrap {
    text-align: center;
    padding: 64px 40px 56px;
    background: radial-gradient(ellipse at 50% 0%, #0d1f3c 0%, #080c14 65%);
    border-bottom: 1px solid rgba(99,179,237,0.08);
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 50% 100%, rgba(99,179,237,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.hero-logo {
    width: 100px; border-radius: 14px;
    filter: drop-shadow(0 4px 24px rgba(255,140,120,0.5));
    margin-bottom: 18px;
}
.hero-lab {
    font-size: 0.68rem; color: rgba(99,179,237,0.7);
    letter-spacing: 0.28em; text-transform: uppercase; margin-bottom: 20px;
}
.hero-title-wrap { margin-bottom: 16px; line-height: 1.0; }
.hero-gravitas {
    font-size: 5.5rem; font-weight: 900;
    font-family: Georgia, "Times New Roman", serif;
    color: #63b3ed;
    display: inline;
}
.hero-ai {
    font-size: 5.5rem; font-weight: 900;
    font-family: Georgia, "Times New Roman", serif;
    color: #f6ad55;
    font-style: italic;
    display: inline;
}
.hero-sub {
    font-size: 2.8rem; font-weight: 900;
    font-family: Georgia, "Times New Roman", serif;
    color: #e2e8f0; margin-bottom: 14px; line-height: 1.1;
}
.hero-desc {
    font-size: 0.92rem; color: #718096;
    line-height: 1.7; margin-bottom: 28px;
    max-width: 520px; margin-left: auto; margin-right: auto;
}
.hero-chips {
    display: flex; flex-wrap: wrap; gap: 10px;
    justify-content: center; margin-bottom: 28px;
}
.hero-chip {
    padding: 7px 16px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
    border: 1px solid rgba(99,179,237,0.3);
    color: #a0aec0; background: rgba(99,179,237,0.06);
    letter-spacing: 0.3px;
}
.hero-cta {
    font-size: 0.8rem; color: #4a5568;
    margin-top: 8px;
}
.hero-arrow {
    font-size: 1.2rem; color: #4a5568; margin-top: 4px;
    animation: bounce 2s infinite;
}
@keyframes bounce {
    0%,100% { transform: translateY(0); }
    50% { transform: translateY(6px); }
}

/* ── Search inputs (v1 style) ── */
.stTextInput > div > div > input {
    background: #111827 !important;
    border: 1px solid rgba(99,179,237,0.3) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 1.0rem !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: #63b3ed !important;
    box-shadow: 0 0 0 2px rgba(99,179,237,0.15) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #63b3ed, #4299e1) !important;
    color: #0a0f1a !important; font-weight: 700 !important;
    border: none !important; border-radius: 8px !important;
    padding: 12px 24px !important; font-size: 0.95rem !important;
    transition: all 0.2s; letter-spacing: 0.3px;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(99,179,237,0.35) !important;
}

/* ── Risk banners ── */
.found-banner {
    background: rgba(104,211,145,0.08); border: 1px solid rgba(104,211,145,0.3);
    border-radius: 8px; padding: 12px 18px; margin: 12px 0;
    font-size: 0.9rem; color: #68d391;
}
.risk-banner {
    border-radius: 10px; padding: 16px 22px; margin: 16px 0;
    display: flex; align-items: center; gap: 16px;
}
.risk-high { background: rgba(252,129,129,0.1); border: 1px solid rgba(252,129,129,0.35); }
.risk-mod  { background: rgba(246,173,85,0.1);  border: 1px solid rgba(246,173,85,0.35);  }
.risk-low  { background: rgba(104,211,145,0.1); border: 1px solid rgba(104,211,145,0.35); }
.risk-icon { font-size: 2rem; }
.risk-label { font-size: 1.2rem; font-weight: 800; letter-spacing: 0.5px; }
.risk-sub   { font-size: 0.78rem; color: #718096; margin-top: 3px; }
.phi-number { font-size: 2.2rem; font-weight: 900; }
.phi-label  { font-size: 0.7rem; color: #718096; }

/* ── Metric tiles ── */
[data-testid="metric-container"] {
    background: #0d1421; border: 1px solid rgba(99,179,237,0.15);
    border-radius: 10px; padding: 1rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: #0d1421; border: 1px solid rgba(99,179,237,0.15);
    border-radius: 7px; color: #718096; font-size: 0.82rem; padding: 7px 14px;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,179,237,0.12) !important;
    border-color: rgba(99,179,237,0.4) !important;
    color: #63b3ed !important;
}

/* ── Cards ── */
.g-card {
    background: #0d1421; border: 1px solid rgba(99,179,237,0.12);
    border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
}
.g-card-teal  { border-color: rgba(79,209,197,0.25); background: rgba(79,209,197,0.04); }
.g-card-red   { border-color: rgba(252,129,129,0.2); background: rgba(252,129,129,0.04); }
.g-card-gold  { border-color: rgba(246,173,85,0.25); background: rgba(246,173,85,0.04); }
.g-card-blue  { border-color: rgba(99,179,237,0.25); background: rgba(99,179,237,0.04); }

/* ── Tables ── */
.g-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
.g-table th { color:#718096; font-weight:600; border-bottom:1px solid rgba(99,179,237,0.15);
              padding:6px 8px; text-align:left; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.5px; }
.g-table td { padding:5px 8px; border-bottom:1px solid rgba(255,255,255,0.04); }
.g-table tr:last-child td { border-bottom:none; }

/* ── Progress bars ── */
.prog-row { display:flex; align-items:center; gap:8px; margin-bottom:5px; }
.prog-lbl { font-size:0.78rem; color:#a0aec0; width:160px; flex-shrink:0; }
.prog-bg  { flex:1; height:6px; background:rgba(255,255,255,0.06); border-radius:3px; overflow:hidden; }
.prog-fill { height:100%; border-radius:3px; transition:width 0.3s; }
.prog-val { font-size:0.75rem; color:#718096; width:44px; text-align:right; }

/* ── DART indicators ── */
.dart-pos { color:#fc8181; font-weight:700; }
.dart-neg { color:#68d391; }
.dart-na  { color:#4a5568; }

/* ── Badges ── */
.badge {
    display:inline-block; padding:3px 10px; border-radius:5px;
    font-size:0.74rem; font-weight:700; letter-spacing:0.5px;
    border: 1px solid rgba(99,179,237,0.3); color:#63b3ed;
    background: rgba(99,179,237,0.08);
}
.b-high { color:#fc8181 !important; border-color:rgba(252,129,129,0.4) !important; background:rgba(252,129,129,0.08) !important; }
.b-mod  { color:#f6ad55 !important; border-color:rgba(246,173,85,0.4) !important;  background:rgba(246,173,85,0.08) !important;  }
.b-low  { color:#68d391 !important; border-color:rgba(104,211,145,0.4) !important; background:rgba(104,211,145,0.08) !important; }

/* ── Misc ── */
.sec-hdr {
    font-size:0.72rem; font-weight:700; color:#63b3ed;
    text-transform:uppercase; letter-spacing:1.2px; margin-bottom:10px;
    padding-bottom:5px; border-bottom:1px solid rgba(99,179,237,0.15);
}
.metric-tile { background:#0d1421; border:1px solid rgba(99,179,237,0.12); border-radius:8px; padding:10px 12px; text-align:center; margin-bottom:8px; }
.metric-val  { font-size:1.25rem; font-weight:800; }
.metric-lbl  { font-size:0.7rem; color:#718096; margin-top:2px; }
.ai-bubble {
    background: rgba(99,179,237,0.06); border: 1px solid rgba(99,179,237,0.2);
    border-radius: 10px; padding: 18px 22px;
    font-size: 0.9rem; line-height: 1.75; color: #e2e8f0;
}
.ai-banner {
    background: rgba(246,173,85,0.08); border: 1px solid rgba(246,173,85,0.25);
    border-radius: 8px; padding: 12px 18px; margin-bottom: 14px;
    font-size: 0.88rem; color: #e2e8f0;
}
.disclaimer {
    font-size: 0.75rem; color: #4a5568; margin-top: 20px;
    padding: 10px 14px; border: 1px solid rgba(255,255,255,0.05);
    border-radius: 6px; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING ─────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    base = os.path.dirname(os.path.abspath(__file__))
    adme   = pd.read_excel(os.path.join(base,"data","ADME_Toxicity_Moleculardocking_Parameters.xlsx")).set_index("Drug names")
    dart_s = pd.read_excel(os.path.join(base,"data","DART_AI_ready_schema.xlsx"),sheet_name="Drug_Summary").set_index("Drug_name")
    dart_e = pd.read_excel(os.path.join(base,"data","DART_AI_ready_schema.xlsx"),sheet_name="Raw_Evidence")
    pbpk_s = pd.read_excel(os.path.join(base,"data","Pregnancy_PBPK_AI_Prototype.xlsx"),sheet_name="Scenario_Calculator")
    pbpk_p = pd.read_excel(os.path.join(base,"data","Pregnancy_PBPK_AI_Prototype.xlsx"),sheet_name="Drug_PBPK_Parameters").set_index("Drug_name")
    pbpk_l = pd.read_excel(os.path.join(base,"data","Pregnancy_PBPK_AI_Prototype.xlsx"),sheet_name="Literature_Status").set_index("Drug_name")
    return {"adme":adme,"dart_s":dart_s,"dart_e":dart_e,
            "pbpk_s":pbpk_s,"pbpk_p":pbpk_p,"pbpk_l":pbpk_l,
            "drugs":sorted(adme.index.tolist())}

def safe(v, d=3, suffix=""):
    try:
        if pd.isna(v): return "N/A"
        return f"{float(v):.{d}f}{suffix}"
    except: return str(v) if v else "N/A"

def compute_phi(row):
    try:
        return round((float(row.get("DILI",0) or 0)*0.25 +
                      float(row.get("hERG",0) or 0)*0.15 +
                      float(row.get("Ames",0) or 0)*0.10 +
                      float(row.get("NR-ER",0) or 0)*0.15 +
                      float(row.get("NR-AR",0) or 0)*0.10 +
                      float(row.get("SR-p53",0) or 0)*0.10 +
                      float(row.get("BBB",0) or 0)*0.15)*100, 1)
    except: return 0.0

def phi_cls(phi):
    if phi >= 65: return "HIGH RISK",  "high", "#fc8181", "b-high", "⚠️"
    if phi >= 40: return "MODERATE",   "mod",  "#f6ad55", "b-mod",  "⚡"
    return            "LOW RISK",   "low",  "#68d391", "b-low",  "✓"

def prog(lbl, val, mx=1.0, invert=True):
    try:
        v=float(val); p=min(100,max(0,(v/mx)*100))
        c=("#fc8181" if v>=0.7 else "#f6ad55" if v>=0.3 else "#68d391") if invert else "#63b3ed"
        return f'<div class="prog-row"><div class="prog-lbl">{lbl}</div><div class="prog-bg"><div class="prog-fill" style="width:{p:.0f}%;background:{c}"></div></div><div class="prog-val">{v:.3f}</div></div>'
    except: return f'<div class="prog-row"><div class="prog-lbl">{lbl}</div><div class="prog-val" style="color:#4a5568">N/A</div></div>'

def get_api_key():
    try: return st.secrets.get("ANTHROPIC_API_KEY","")
    except: return os.environ.get("ANTHROPIC_API_KEY","")

def load_network_data(drug_name):
    app_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(app_dir, "data", drug_name, "Network_Pharmacology"),
        os.path.join(app_dir, drug_name, "Network_Pharmacology"),
        os.path.join(os.path.dirname(app_dir), drug_name, "Network_Pharmacology"),
    ]
    folder = next((c for c in candidates if os.path.isdir(c)), candidates[0])
    result, figures = {}, {}
    file_stems = {"kegg":f"{drug_name}_KEGG","reactome":f"{drug_name}_Reactome",
                   "go_bp":f"{drug_name}_GO_BP","go_cc":f"{drug_name}_GO_CC",
                   "go_mf":f"{drug_name}_GO_MF","hub_genes":f"{drug_name}_Hubgenes_STRING_centrality",
                   "common_genes":"Commongenes_Venn"}
    for key, stem in file_stems.items():
        for ext, reader in [(".csv", pd.read_csv), (".xlsx", pd.read_excel)]:
            p = os.path.join(folder, stem + ext)
            if os.path.exists(p):
                try: result[key] = reader(p); break
                except: pass
    for key, stem in {"fig_dotplot_bp":f"{drug_name}_Dotplot_GO_BP","fig_dotplot_cc":f"{drug_name}_Dotplot_GO_CC",
                      "fig_dotplot_mf":f"{drug_name}_Dotplot_GO_MF","fig_enrichmap_kegg":f"{drug_name}_EnrichmentMap_KEGG",
                      "fig_enrichmap_reactome":f"{drug_name}_EnrichmentMap_Reactome"}.items():
        for ext in [".png",".jpg",".jpeg",".svg"]:
            p = os.path.join(folder,stem+ext)
            if os.path.exists(p): figures[key]=p; break
    result["figures"] = figures
    return result

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding:1.2rem 0 0.8rem">
        <img src="data:image/png;base64,{LOGO_B64}"
             style="width:100px; border-radius:10px;
                    filter:drop-shadow(0 2px 12px rgba(255,160,140,0.4))"/>
        <div style="font-size:0.65rem; color:rgba(255,160,140,0.85);
                    letter-spacing:0.15em; text-transform:uppercase;
                    margin-top:0.6rem">The Menon Laboratory</div>
        <div style="font-size:0.6rem; color:#718096; margin-top:0.2rem">
            Perinatal Research
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🔬 Gravitas AI 2.0")
    st.markdown("""
    <div style="font-size:0.8rem; color:#718096; line-height:1.7">
    Pregnancy drug safety intelligence.<br><br>
    🧬 16 validated compounds<br>
    📊 175 ADME parameters<br>
    ⚗️ 6 molecular docking targets<br>
    💊 T1·T2·T3 PBPK modelled<br>
    🕸️ Network pharmacology<br>
    🤖 ∞ novel compounds via AI
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", [
        "🔍 Drug Search",
        "🕸️ Network Pharmacology",
        "🤖 AI Consultation",
        "ℹ️ About"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.65rem; color:#4a5568; text-align:center; line-height:1.6">
    ⚠️ Research use only.<br>Not for clinical decisions.
    </div>
    """, unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────────────────
try:
    DATA = load_all_data()
except Exception as e:
    st.error(f"⚠️ Data loading error: {e}")
    st.stop()

# ══════════════════════════════════════════════════════════════════
# PAGE: DRUG SEARCH
# ══════════════════════════════════════════════════════════════════
if page == "🔍 Drug Search":

    # Hero — landmark style matching v1 screenshot
    st.markdown(f"""
    <div class="hero-wrap">
        <img src="data:image/png;base64,{{LOGO_B64}}" class="hero-logo"/>
        <div class="hero-lab">The Menon Laboratory &nbsp;·&nbsp; Perinatal Research &nbsp;·&nbsp; Pregnancy Drug Intelligence</div>
        <div class="hero-title-wrap">
            <span class="hero-gravitas">Gravitas </span><span class="hero-ai">AI</span>
        </div>
        <div class="hero-sub">Know Before You Prescribe</div>
        <div class="hero-desc">
            Instant prediction of pregnancy safety, toxicity risk, ADME properties,<br>
            molecular pathways, and recommended dosing — powered by 16<br>
            validated compounds and AI.
        </div>
        <div class="hero-chips">
            <span class="hero-chip">16 Drugs in DB</span>
            <span class="hero-chip">🤖 AI for Novel Compounds</span>
            <span class="hero-chip">175 ADME Parameters</span>
            <span class="hero-chip">Toxicity Profiling</span>
            <span class="hero-chip">Pregnancy Pathways</span>
            <span class="hero-chip">PBPK Modelling</span>
            <span class="hero-chip">DART Analysis</span>
        </div>
        <div class="hero-cta">Search below ↓</div>
    </div>
    """.replace("{{LOGO_B64}}", LOGO_B64), unsafe_allow_html=True)

    # Search box — v1 layout
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("", placeholder="Drug name (e.g. Aspirin, Indomethacin, Thalidomide...)",
                              label_visibility="collapsed")
    with col2:
        search = st.button("Analyze →", use_container_width=True)

    st.markdown("""
    <div style="font-size:0.78rem; color:#4a5568; margin-top:4px; margin-bottom:8px">
    Type any drug name · Found in database → full validated profile · Not found → AI-powered analysis
    </div>
    """, unsafe_allow_html=True)

    if (search or query) and query.strip():
        q = query.strip()
        drugs_lower = {d.lower(): d for d in DATA["drugs"]}
        match = None
        if q.lower() in drugs_lower:
            match = drugs_lower[q.lower()]
        else:
            for dl, dn in drugs_lower.items():
                if q.lower() in dl:
                    match = dn; break

        # ── DATABASE HIT ─────────────────────────────────────────
        if match:
            row    = DATA["adme"].loc[match]
            phi    = compute_phi(row)
            plabel, pcls, pcolor, pbadge, picon = phi_cls(phi)
            dart_r = DATA["dart_s"].loc[match] if match in DATA["dart_s"].index else {}
            pbpk_d = DATA["pbpk_s"][DATA["pbpk_s"]["Drug_name"]==match]
            lit_r  = DATA["pbpk_l"].loc[match] if match in DATA["pbpk_l"].index else {}
            dart_sig = dart_r.get("Overall_DART_signal","N/A") if len(dart_r) else "N/A"

            st.markdown(f'<div class="found-banner">✓ Found in database: <strong>{match}</strong></div>', unsafe_allow_html=True)

            # Risk banner
            st.markdown(f"""
            <div class="risk-banner risk-{pcls}">
                <div class="risk-icon">{picon}</div>
                <div style="flex:1">
                    <div class="risk-label" style="color:{pcolor}">{plabel}</div>
                    <div class="risk-sub">Pregnancy Hazard Index · Validated multi-modal data</div>
                </div>
                <div style="text-align:center">
                    <div class="phi-number" style="color:{pcolor}">{phi}</div>
                    <div class="phi-label">PHI / 100</div>
                </div>
                <div style="text-align:center;margin-left:16px">
                    <span class="badge {pbadge}">{plabel}</span><br>
                    <span style="font-size:0.74rem;color:#718096;margin-top:4px;display:block">
                        DART: <span style="color:{'#fc8181' if dart_sig=='Positive' else '#68d391' if dart_sig=='Negative' else '#718096'}">{dart_sig}</span>
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

            tabs = st.tabs(["📋 Summary","💊 PK / ADME","🔄 ADME Detail","☢️ Toxicity","⚗️ Docking","💉 PBPK","🧪 DART","🕸️ Pathways","🤖 AI Analysis"])

            # ── TAB 0: SUMMARY ───────────────────────────────────
            with tabs[0]:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown('<div class="sec-hdr">Physicochemical</div>', unsafe_allow_html=True)
                    rows = [("MW", safe(row.get("MW"),2), "g/mol"),("LogP", safe(row.get("logP"),3),""),
                            ("LogS", safe(row.get("logS"),3),""),("TPSA", safe(row.get("TPSA"),1),"Å²"),
                            ("QED", safe(row.get("QED"),3),""),("HBD", safe(row.get("nHD"),0),""),
                            ("HBA", safe(row.get("nHA"),0),""),("Fsp3", safe(row.get("Fsp3"),3),"")]
                    h = '<table class="g-table">'
                    for l,v,u in rows:
                        h += f'<tr><td style="color:#718096">{l}</td><td style="font-weight:600">{v} <span style="color:#4a5568;font-size:0.75rem">{u}</span></td></tr>'
                    st.markdown(h+'</table>', unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="sec-hdr">Key ADME</div>', unsafe_allow_html=True)
                    adme_k = [("BBB Penetration", safe(row.get("BBB"),3)),
                               ("Plasma Protein Binding", f'{safe(row.get("PPB"),1)}%'),
                               ("Fraction Unbound", f'{safe(row.get("Fu"),1)}%'),
                               ("HIA", safe(row.get("hia"),3)),
                               ("Caco-2", safe(row.get("caco2"),3)),
                               ("P-gp Substrate", safe(row.get("pgp_sub"),3)),
                               ("Oral F (30%)", safe(row.get("f30"),3)),
                               ("t½", safe(row.get("t0.5"),2)+" hr")]
                    h = '<table class="g-table">'
                    for l,v in adme_k:
                        h += f'<tr><td style="color:#718096">{l}</td><td style="font-weight:600">{v}</td></tr>'
                    st.markdown(h+'</table>', unsafe_allow_html=True)
                with c3:
                    st.markdown('<div class="sec-hdr">Risk Profile</div>', unsafe_allow_html=True)
                    risk_items = [("DILI Risk", row.get("DILI")),("hERG Cardiotox", row.get("hERG")),
                                  ("Ames Mutagenicity", row.get("Ames")),("NR-ER Activity", row.get("NR-ER")),
                                  ("NR-AR Activity", row.get("NR-AR")),("SR-p53", row.get("SR-p53")),
                                  ("Carcinogenicity", row.get("Carcinogenicity")),("Neurotoxicity", row.get("Neurotoxicity-DI"))]
                    h = ""
                    for l,v in risk_items: h += prog(l,v)
                    st.markdown(h, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                c4, c5 = st.columns(2)
                with c4:
                    st.markdown('<div class="sec-hdr">DART Quick View</div>', unsafe_allow_html=True)
                    if len(dart_r):
                        ep = {"EFD study": dart_r.get("EFD_available"),
                              "Embryo-fetal death": dart_r.get("Embryo_fetal_death_or_loss"),
                              "Fetal growth reduction": dart_r.get("Fetal_growth_reduction"),
                              "Skeletal malformation": dart_r.get("Skeletal_malformation"),
                              "Neurobehavioral effect": dart_r.get("Neurobehavioral_or_IQ_effect"),
                              "Neonatal toxicity": dart_r.get("Neonatal_toxicity")}
                        h = '<table class="g-table">'
                        for l,v in ep.items():
                            ind = '<span class="dart-pos">⚠ Positive</span>' if (pd.notna(v) and v==1) else '<span class="dart-na">—</span>' if pd.isna(v) else '<span class="dart-neg">✓ Not reported</span>'
                            h += f'<tr><td style="color:#718096">{l}</td><td>{ind}</td></tr>'
                        st.markdown(h+'</table>', unsafe_allow_html=True)
                        if dart_r.get("Summary_text"):
                            st.markdown(f'<div class="g-card g-card-red" style="font-size:0.81rem;color:#a0aec0;margin-top:8px">{dart_r.get("Summary_text","")}</div>', unsafe_allow_html=True)
                with c5:
                    st.markdown('<div class="sec-hdr">PBPK Fetal Exposure</div>', unsafe_allow_html=True)
                    if not pbpk_d.empty:
                        h = '<table class="g-table"><tr><th>Trimester</th><th>Maternal Cavg</th><th>Fetal Plasma</th><th>Flag</th></tr>'
                        tc = {"T1":"#b794f4","T2":"#63b3ed","T3":"#68d391"}
                        for _,pr in pbpk_d.iterrows():
                            flag = str(pr.get("Exposure_flag",""))
                            fc = "#fc8181" if "Above" in flag else "#68d391"
                            h += f'<tr><td style="color:{tc.get(pr["Trimester"],"#718096")};font-weight:700">{pr["Trimester"]}</td><td>{safe(pr.get("Maternal_Cavg_mg_L"),5)}</td><td style="color:#63b3ed">{safe(pr.get("Fetal_plasma_Cavg_mg_L"),5)}</td><td style="color:{fc};font-size:0.77rem">{flag}</td></tr>'
                        st.markdown(h+'</table>', unsafe_allow_html=True)

            # ── TAB 1: PK / ADME (v1-style metrics) ─────────────
            with tabs[1]:
                st.markdown('<div class="sec-hdr">Pharmacokinetics</div>', unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Half-Life (hr)", safe(row.get("t0.5"),2))
                col2.metric("VDss (L/kg)", safe(row.get("logVDss"),3))
                col3.metric("PPB %", f"{float(row.get('PPB') or 0):.0f}%")
                col4.metric("Fraction Unbound %", safe(row.get("Fu"),1))
                st.markdown('<div class="sec-hdr" style="margin-top:18px">ADME Properties</div>', unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MW (g/mol)", safe(row.get("MW"),2))
                col2.metric("LogP", safe(row.get("logP"),3))
                col3.metric("HIA", safe(row.get("hia"),3))
                col4.metric("TPSA (Å²)", safe(row.get("TPSA"),1))
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("BBB Penetration", safe(row.get("BBB"),3))
                col2.metric("QED", safe(row.get("QED"),3))
                col3.metric("Caco-2", safe(row.get("caco2"),3))
                col4.metric("P-gp Substrate", safe(row.get("pgp_sub"),3))

            # ── TAB 2: ADME DETAIL ───────────────────────────────
            with tabs[2]:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="sec-hdr">Absorption & Oral Bioavailability</div>', unsafe_allow_html=True)
                    for l,v in [("HIA",row.get("hia")),("Oral F 20%",row.get("f20")),
                                ("Oral F 30%",row.get("f30")),("Oral F 50%",row.get("f50")),
                                ("BBB",row.get("BBB")),("PAMPA",row.get("PAMPA"))]:
                        st.markdown(prog(l,v,invert=False), unsafe_allow_html=True)
                    st.markdown('<div class="sec-hdr" style="margin-top:14px">Transporters</div>', unsafe_allow_html=True)
                    h = '<table class="g-table">'
                    for l,v in [("P-gp Substrate",row.get("pgp_sub")),("P-gp Inhibitor",row.get("pgp_inh")),
                                ("OATP1B1",row.get("OATP1B1")),("OATP1B3",row.get("OATP1B3")),
                                ("BCRP",row.get("BCRP")),("MRP1",row.get("MRP1")),("BSEP",row.get("BSEP"))]:
                        try:
                            fv=float(v); col="#fc8181" if fv>0.7 else "#f6ad55" if fv>0.3 else "#68d391"
                            disp=f'<span style="color:{col};font-weight:600">{fv:.3f}</span>'
                        except: disp='<span style="color:#4a5568">N/A</span>'
                        h += f'<tr><td style="color:#718096">{l}</td><td>{disp}</td></tr>'
                    st.markdown(h+'</table>', unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="sec-hdr">CYP Inhibition Probability</div>', unsafe_allow_html=True)
                    for l,v in [("CYP1A2-inh",row.get("CYP1A2-inh")),("CYP2C19-inh",row.get("CYP2C19-inh")),
                                ("CYP2C9-inh",row.get("CYP2C9-inh")),("CYP2D6-inh",row.get("CYP2D6-inh")),
                                ("CYP3A4-inh",row.get("CYP3A4-inh")),("CYP2B6-inh",row.get("CYP2B6-inh"))]:
                        st.markdown(prog(l,v), unsafe_allow_html=True)
                    st.markdown('<div class="sec-hdr" style="margin-top:14px">Distribution</div>', unsafe_allow_html=True)
                    h = '<table class="g-table">'
                    for l,v in [("PPB %", f'{safe(row.get("PPB"),1)}%'),("Fu %", f'{safe(row.get("Fu"),1)}%'),
                                ("LogVDss", safe(row.get("logVDss"),3)),("t½ (hr)", safe(row.get("t0.5"),2)),
                                ("CL plasma", safe(row.get("cl-plasma"),3)),("LM human", safe(row.get("LM-human"),3))]:
                        h += f'<tr><td style="color:#718096">{l}</td><td style="font-weight:600">{v}</td></tr>'
                    st.markdown(h+'</table>', unsafe_allow_html=True)

            # ── TAB 3: TOXICITY ──────────────────────────────────
            with tabs[3]:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown('<div class="sec-hdr">Core Toxicity</div>', unsafe_allow_html=True)
                    for l,v in [("DILI",row.get("DILI")),("hERG Cardiac",row.get("hERG")),
                                ("Ames Mutagenicity",row.get("Ames")),("H-HT",row.get("H-HT")),
                                ("Carcinogenicity",row.get("Carcinogenicity")),("Genotoxicity",row.get("Genotoxicity")),
                                ("Respiratory",row.get("Respiratory")),("Skin Sensitization",row.get("SkinSen"))]:
                        st.markdown(prog(l,v), unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="sec-hdr">Organ Toxicity</div>', unsafe_allow_html=True)
                    for l,v in [("Neurotoxicity",row.get("Neurotoxicity-DI")),("Nephrotoxicity",row.get("Nephrotoxicity-DI")),
                                ("Hematotoxicity",row.get("Hematotoxicity")),("Ototoxicity",row.get("Ototoxicity")),
                                ("Cardiotoxicity",row.get("cardio")),("Immunotoxicity",row.get("immuno"))]:
                        st.markdown(prog(l,v), unsafe_allow_html=True)
                with c3:
                    st.markdown('<div class="sec-hdr">Nuclear Receptors & Stress</div>', unsafe_allow_html=True)
                    for l,v in [("NR-ER",row.get("NR-ER")),("NR-AR",row.get("NR-AR")),
                                ("NR-Aromatase",row.get("NR-Aromatase")),("NR-PPAR-γ",row.get("NR-PPAR-gamma")),
                                ("SR-p53",row.get("SR-p53")),("SR-ARE",row.get("SR-ARE")),
                                ("SR-HSE",row.get("SR-HSE")),("SR-MMP",row.get("SR-MMP"))]:
                        st.markdown(prog(l,v), unsafe_allow_html=True)

            # ── TAB 4: DOCKING ───────────────────────────────────
            with tabs[4]:
                st.markdown('<div class="sec-hdr">Molecular Docking — PTB Inflammatory Targets (kcal/mol)</div>', unsafe_allow_html=True)
                st.markdown('<div style="font-size:0.79rem;color:#718096;margin-bottom:14px">More negative = stronger binding · Strong &lt;−9.0 · Moderate −7.0 to −9.0 · Weak &gt;−7.0</div>', unsafe_allow_html=True)
                targets = [("P38-MAPK","P-38","Inflammatory cytokine production in PTB"),
                           ("NF-κB","NFKB","Master regulator of inflammatory gene expression"),
                           ("MAPK","MAPK","Prostaglandin synthesis & uterine contractility"),
                           ("JAK2","JAK2","IL-6 & cytokine signaling at feto-maternal interface"),
                           ("TGFβR1","TGFBR1","Placental development & immune tolerance"),
                           ("HIF1α","HIF1A","Placental angiogenesis & oxygen sensing")]
                cols = st.columns(3)
                for i,(name,key,desc) in enumerate(targets):
                    val = row.get(key)
                    try:
                        v=float(val); col="#fc8181" if v<=-9 else "#f6ad55" if v<=-7 else "#68d391" if v<=-5 else "#718096"
                        strength = "Strong" if v<=-9 else "Moderate" if v<=-7 else "Weak"; disp=f"{v:.1f}"
                    except: col,strength,disp = "#4a5568","N/A","N/A"
                    with cols[i%3]:
                        st.markdown(f'''<div class="g-card" style="border-left:4px solid {col}">
                        <div style="font-size:0.95rem;font-weight:700;color:{col}">{name}</div>
                        <div style="font-size:1.8rem;font-weight:900;color:{col};margin:4px 0">{disp} <span style="font-size:0.85rem">kcal/mol</span></div>
                        <div style="font-size:0.76rem;color:#718096;margin-bottom:4px">{strength}</div>
                        <div style="font-size:0.73rem;color:#4a5568;line-height:1.4">{desc}</div>
                        </div>''', unsafe_allow_html=True)
                st.markdown('<div class="sec-hdr" style="margin-top:18px">CYP Enzyme Docking</div>', unsafe_allow_html=True)
                cyp_cols = st.columns(6)
                for col,(n,k) in zip(cyp_cols,[("CYP1A2","CYP1A2"),("CYP2C19","CYP2C19"),("CYP2C9","CYP2C9"),("CYP2D6","CYP2D6"),("CYP3A4","CYP3A4"),("CYP2E1","CYP2E1")]):
                    with col:
                        try:
                            v=float(row.get(k)); c="#fc8181" if v<=-9 else "#f6ad55" if v<=-7 else "#68d391"
                            col.metric(n, f"{v:.2f}")
                        except: col.metric(n, "N/A")

            # ── TAB 5: PBPK ──────────────────────────────────────
            with tabs[5]:
                st.markdown('<div class="sec-hdr">Pregnancy PBPK — Trimester Exposure Modelling</div>', unsafe_allow_html=True)
                if len(lit_r):
                    sc = {"Pregnancy PBPK available":"#68d391","Pregnancy maternal-fetal PBPK available":"#68d391",
                          "Observed pregnancy popPK available":"#63b3ed","Developmental tox PK/IVIVE only":"#f6ad55","Scaffold only":"#fc8181"}
                    sc_col = sc.get(str(lit_r.get("Status_bucket","")), "#718096")
                    st.markdown(f'''<div class="g-card" style="border-left:4px solid {sc_col};margin-bottom:14px">
                    <span class="badge" style="color:{sc_col};border-color:{sc_col}44;background:{sc_col}11">{lit_r.get("Status_bucket","N/A")}</span>
                    <div style="font-size:0.83rem;color:#a0aec0;margin-top:7px">{lit_r.get("Key_quantitative_note","N/A")}</div>
                    </div>''', unsafe_allow_html=True)
                if not pbpk_d.empty:
                    h = '<table class="g-table"><tr><th>Trimester</th><th>Dose (mg/d)</th><th>CL preg</th><th>Vd preg</th><th>Maternal AUC24</th><th>Maternal Cavg</th><th>Maternal Unbound</th><th>Fetal Plasma</th><th>Fetal Tissue</th><th>Flag</th></tr>'
                    tc = {"T1":"#b794f4","T2":"#63b3ed","T3":"#68d391"}
                    for _,pr in pbpk_d.iterrows():
                        flag = str(pr.get("Exposure_flag",""))
                        fc = "#fc8181" if "Above" in flag else "#68d391"
                        h += f'<tr><td style="color:{tc.get(pr["Trimester"],"#718096")};font-weight:700">{pr["Trimester"]}</td><td>{safe(pr.get("Dose_used_mg_day"),1)}</td><td>{safe(pr.get("CL_preg_L_hr"),3)}</td><td>{safe(pr.get("Vd_preg_L"),2)}</td><td>{safe(pr.get("Maternal_AUC24_mg*h_L"),4)}</td><td style="font-weight:600">{safe(pr.get("Maternal_Cavg_mg_L"),5)}</td><td>{safe(pr.get("Maternal_unbound_Cavg_mg_L"),5)}</td><td style="color:#63b3ed;font-weight:600">{safe(pr.get("Fetal_plasma_Cavg_mg_L"),5)}</td><td style="color:#4fd1c5">{safe(pr.get("Fetal_tissue_Cavg_mg_L"),5)}</td><td style="color:{fc};font-size:0.77rem">{flag}</td></tr>'
                    st.markdown(h+'</table>', unsafe_allow_html=True)
                else:
                    st.info("PBPK data not available for this compound.")

            # ── TAB 6: DART ──────────────────────────────────────
            with tabs[6]:
                if len(dart_r):
                    od = dart_r.get("Overall_DART_signal","N/A"); conf = dart_r.get("Evidence_confidence","N/A")
                    oc = "#fc8181" if od=="Positive" else "#68d391" if od=="Negative" else "#f6ad55"
                    dart_pcls = "high" if od=="Positive" else "low" if od=="Negative" else "mod"
                    st.markdown(f'''<div class="risk-banner risk-{dart_pcls}">
                    <div>
                        <div style="font-size:0.7rem;color:#718096;text-transform:uppercase">Overall DART Signal</div>
                        <div style="font-size:1.35rem;font-weight:700;color:{oc}">{od}</div>
                    </div>
                    <div style="margin-left:24px">
                        <div style="font-size:0.7rem;color:#718096;text-transform:uppercase">Confidence</div>
                        <div style="font-size:1.05rem;font-weight:600;color:#f6ad55">{conf}</div>
                    </div>
                    <div style="flex:1;margin-left:24px;font-size:0.84rem;color:#a0aec0">{dart_r.get("Summary_text","")}</div>
                    </div>''', unsafe_allow_html=True)
                    grps = {"Fertility":["Male_fertility_signal","Female_fertility_signal"],
                            "Embryo-Fetal":["Embryo_fetal_death_or_loss","Implantation_loss","Resorptions","Fetal_growth_reduction"],
                            "Structural":["Skeletal_variation_or_delayed_ossification","Skeletal_malformation","Visceral_malformation","External_malformation"],
                            "Postnatal":["Postnatal_survival_decrease","Developmental_delay","Neurobehavioral_or_IQ_effect","Neonatal_toxicity"],
                            "Maternal":["Maternal_toxicity_reported"]}
                    gcols = st.columns(len(grps))
                    for col,(grp,eps) in zip(gcols,grps.items()):
                        with col:
                            st.markdown(f'<div style="font-size:0.73rem;font-weight:700;color:#63b3ed;text-transform:uppercase;letter-spacing:1px;margin-bottom:7px">{grp}</div>', unsafe_allow_html=True)
                            for ep in eps:
                                v = dart_r.get(ep); clean = ep.replace("_"," ").replace(" signal","")
                                icon,cls = ("⚠","dart-pos") if (pd.notna(v) and v==1) else ("—","dart-na") if pd.isna(v) else ("✓","dart-neg")
                                st.markdown(f'<div style="font-size:0.77rem;padding:2px 0"><span class="{cls}">{icon}</span> <span style="color:#a0aec0">{clean}</span></div>', unsafe_allow_html=True)
                    evid = DATA["dart_e"][DATA["dart_e"]["Drug_name"]==match]
                    if not evid.empty:
                        st.markdown('<div class="sec-hdr" style="margin-top:18px">Raw Study Evidence</div>', unsafe_allow_html=True)
                        h = '<table class="g-table"><tr>'
                        for c in ["Study_type","Species","Route","Dose","Endpoint_term","Result_code","NOAEL"]: h += f'<th>{c}</th>'
                        h += '</tr>'
                        for _,er in evid.iterrows():
                            rc = str(er.get("Result_code",""))
                            rc_c = "#fc8181" if rc in ["Positive","Adverse"] else "#68d391" if rc=="Negative" else "#f6ad55"
                            h += '<tr>'
                            for c in ["Study_type","Species","Route","Dose","Endpoint_term","Result_code","NOAEL"]:
                                v = er.get(c,"—"); v = "—" if pd.isna(v) else str(v)
                                attr = f'style="color:{rc_c}"' if c=="Result_code" else ""
                                h += f'<td {attr}>{v}</td>'
                            h += '</tr>'
                        st.markdown(h+'</table>', unsafe_allow_html=True)
                else:
                    st.info("DART data not available for this compound.")

            # ── TAB 7: PATHWAYS ──────────────────────────────────
            with tabs[7]:
                net = load_network_data(match); figs = net.pop("figures",{})
                has_tables = any(k in net for k in ["kegg","reactome","go_bp","go_cc","go_mf","hub_genes"])
                if has_tables or figs:
                    ntabs = st.tabs(["KEGG","Reactome","GO-BP","GO-CC","GO-MF","Hub Genes","Figures"])
                    nkeys = [("kegg","KEGG"),("reactome","Reactome"),("go_bp","GO BP"),("go_cc","GO CC"),("go_mf","GO MF"),("hub_genes","Hub Genes")]
                    for nt,(k,title) in zip(ntabs[:-1],nkeys):
                        with nt:
                            if k in net: st.dataframe(net[k],use_container_width=True)
                            else: st.info(f"📂 File not found for {match} — {k}. Upload .csv or .xlsx to: data/{match}/Network_Pharmacology/")
                    with ntabs[-1]:
                        fig_lbls = {"fig_dotplot_bp":"GO-BP Dot Plot","fig_dotplot_cc":"GO-CC Dot Plot",
                                    "fig_dotplot_mf":"GO-MF Dot Plot","fig_enrichmap_kegg":"KEGG Enrichment Map",
                                    "fig_enrichmap_reactome":"Reactome Enrichment Map"}
                        if figs:
                            fc = st.columns(2)
                            for i,(k,lbl) in enumerate(fig_lbls.items()):
                                if k in figs:
                                    with fc[i%2]:
                                        st.markdown(f'<div class="sec-hdr">{lbl}</div>', unsafe_allow_html=True)
                                        st.image(figs[k],use_container_width=True)
                        else: st.info("📂 Place PNG figures in the Network_Pharmacology folder.")
                else:
                    st.markdown('<div class="sec-hdr">PTB Pathway Target Binding (from Docking)</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="ai-banner">📂 Network pharmacology files not yet uploaded for <strong>{match}</strong>. Upload KEGG/Reactome/GO .xlsx files to <code>data/{match}/Network_Pharmacology/</code> to see full analysis. Showing docking-based pathway relevance below.</div>', unsafe_allow_html=True)
                    pcols = st.columns(2)
                    for i,(pw,tgts) in enumerate({"Inflammatory Cascade":[("NF-κB","NFKB"),("P38-MAPK","P-38"),("MAPK","MAPK")],
                                                   "Cytokine Signaling":[("JAK2","JAK2"),("TGFβR1","TGFBR1")],
                                                   "Hypoxia/Angiogenesis":[("HIF1α","HIF1A")],
                                                   "CYP Metabolism":[("CYP3A4","CYP3A4"),("CYP1A2","CYP1A2"),("CYP2C9","CYP2C9")]}.items()):
                        with pcols[i%2]:
                            h = f'<div class="g-card g-card-teal"><div style="font-size:0.83rem;font-weight:700;color:#4fd1c5;margin-bottom:8px">{pw}</div>'
                            for tn,tk in tgts:
                                try:
                                    v=float(row.get(tk)); c="#fc8181" if v<=-9 else "#f6ad55" if v<=-7 else "#68d391"
                                    h += f'<div style="font-size:0.81rem;padding:2px 0"><span style="color:{c};font-weight:600">{tn}: {v:.1f} kcal/mol</span></div>'
                                except: h += f'<div style="font-size:0.81rem;padding:2px 0;color:#4a5568">{tn}: N/A</div>'
                            st.markdown(h+'</div>', unsafe_allow_html=True)

            # ── TAB 8: AI ANALYSIS ───────────────────────────────
            with tabs[8]:
                user_type = st.selectbox("View As", ["👩‍⚕️ Clinician","🤰 Patient","🔬 Pharma Researcher"], key="utype")
                custom_q  = st.text_input("Specific question (optional)",
                    placeholder=f"e.g. Safe in T3? Breastfeeding risk? Placental transfer?", key=f"q_{match}")
                if st.button("🤖 Analyze with Gravitas AI", use_container_width=True, key=f"ai_{match}"):
                    key_a = get_api_key()
                    if not key_a: st.warning("API key not configured in Streamlit secrets.")
                    else:
                        ctx = f"""DRUG: {match}
PHI: {phi}/100 | DART: {dart_r.get("Overall_DART_signal","N/A") if len(dart_r) else "N/A"}
MW: {safe(row.get("MW"),2)} | LogP: {safe(row.get("logP"),3)} | TPSA: {safe(row.get("TPSA"),1)} Å²
BBB: {safe(row.get("BBB"),3)} | PPB: {safe(row.get("PPB"),1)}% | Fu: {safe(row.get("Fu"),1)}%
DILI: {safe(row.get("DILI"),3)} | hERG: {safe(row.get("hERG"),3)} | Ames: {safe(row.get("Ames"),3)}
NR-ER: {safe(row.get("NR-ER"),4)} | NR-AR: {safe(row.get("NR-AR"),4)} | NR-Aromatase: {safe(row.get("NR-Aromatase"),4)}
P38: {safe(row.get("P-38"),1)} | NF-κB: {safe(row.get("NFKB"),1)} | MAPK: {safe(row.get("MAPK"),1)} | JAK2: {safe(row.get("JAK2"),1)} | TGFβR1: {safe(row.get("TGFBR1"),1)} | HIF1α: {safe(row.get("HIF1A"),1)} kcal/mol
DART: {dart_r.get("Summary_text","N/A") if len(dart_r) else "N/A"}
PBPK: {lit_r.get("Status_bucket","N/A") if len(lit_r) else "N/A"} — {lit_r.get("Key_quantitative_note","N/A") if len(lit_r) else "N/A"}"""
                        for _,pr in pbpk_d.iterrows():
                            ctx += f"\n{pr['Trimester']}: Maternal={safe(pr.get('Maternal_Cavg_mg_L'),4)} mg/L, Fetal={safe(pr.get('Fetal_plasma_Cavg_mg_L'),4)} mg/L, {pr.get('Exposure_flag','')}"
                        sys_map = {
                            "👩‍⚕️ Clinician":"You are a clinical pharmacology expert in obstetric pharmacotherapy. Be direct, evidence-based, clinically actionable. Focus on dosing, monitoring, fetal exposure, and trimester-specific risks.",
                            "🤰 Patient":"You are a patient-friendly pregnancy safety advisor. Use plain, warm language. Focus on safety, trimester concerns, and what to discuss with their doctor.",
                            "🔬 Pharma Researcher":"You are a pharmaceutical scientist expert in reproductive toxicology and DMPK. Be technical and mechanistic. Reference specific values. Discuss ADME-PK, docking, PBPK, DART."
                        }
                        q_map = {
                            "👩‍⚕️ Clinician":f"Provide clinical assessment of {match}: dosing, fetal exposure, trimester risks, monitoring.",
                            "🤰 Patient":f"Is {match} safe during pregnancy? Explain simply.",
                            "🔬 Pharma Researcher":f"Provide full pharmacological and toxicological analysis of {match} for pregnancy research."
                        }
                        with st.spinner(f"🤖 Gravitas AI analyzing {match}..."):
                            try:
                                client = anthropic.Anthropic(api_key=key_a)
                                resp = client.messages.create(
                                    model=CLAUDE_MODEL, max_tokens=1200,
                                    system=sys_map.get(user_type, sys_map["👩‍⚕️ Clinician"]),
                                    messages=[{"role":"user","content":f"Data:\n{ctx}\n\nQuestion: {custom_q or q_map.get(user_type,'')}"}]
                                )
                                st.markdown(f'<div class="ai-bubble">🤖 {resp.content[0].text}</div>', unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"AI error: {e}")
                st.markdown('<div class="disclaimer">⚠️ Gravitas AI 2.0 is a research tool. All medication decisions during pregnancy must be made with a qualified healthcare provider.</div>', unsafe_allow_html=True)


        # ── NOT IN DATABASE — RAG AI ──────────────────────────────
        else:
            st.markdown(f'<div class="ai-banner">🤖 <strong>{q}</strong> is not in the validated database. Running Gravitas AI analysis calibrated against our 16-compound validated dataset...</div>', unsafe_allow_html=True)
            key_a = get_api_key()
            if not key_a:
                st.warning("API key not configured.")
            else:
                adme_df = DATA["adme"]; dart_df = DATA["dart_s"]
                phi_scores = [(d, compute_phi(adme_df.loc[d])) for d in DATA["drugs"]]
                phi_scores.sort(key=lambda x: x[1])
                n = len(phi_scores)
                ref_indices = [0, n//6, n//3, n//2, n*2//3, n*5//6, n-2, n-1]
                ref_drugs = [phi_scores[min(i,n-1)][0] for i in ref_indices]
                rag_ctx = "VALIDATED REFERENCE DATABASE (8 calibration compounds):\n"
                for rd in ref_drugs:
                    rr = adme_df.loc[rd]; rphi = compute_phi(rr)
                    dart_sig2 = dart_df.loc[rd,"Overall_DART_signal"] if rd in dart_df.index else "N/A"
                    rag_ctx += f"\nDrug: {rd} | PHI: {rphi} | DART: {dart_sig2}\n  DILI:{safe(rr.get('DILI'),3)} hERG:{safe(rr.get('hERG'),3)} Ames:{safe(rr.get('Ames'),3)} NR-ER:{safe(rr.get('NR-ER'),4)} NR-AR:{safe(rr.get('NR-AR'),4)}\n  BBB:{safe(rr.get('BBB'),3)} LogP:{safe(rr.get('logP'),3)} MW:{safe(rr.get('MW'),2)} TPSA:{safe(rr.get('TPSA'),1)}\n  P38:{safe(rr.get('P-38'),1)} NF-kB:{safe(rr.get('NFKB'),1)} TGFbR1:{safe(rr.get('TGFBR1'),1)} kcal/mol"
                with st.spinner(f"🤖 Gravitas AI analyzing {q}..."):
                    prompt = f"""{rag_ctx}\n\nNOVEL COMPOUND TO ANALYZE: {q}\n\nReturn ONLY valid JSON — no markdown fences:\n{{"name":"compound name","phi":0,"risk_label":"HIGH RISK/MODERATE/LOW RISK","summary":"3-4 sentence pregnancy safety summary","trimester_risk":{{"T1":"risk + mechanism","T2":"risk","T3":"risk"}},"recommendations":[{{"icon":"✅/⚠️/❌","text":"recommendation"}}],"adme":{{"MW":"x Da","LogP":"x","HIA":"x%","BBB":"x%","PPB":"x%","t_half":"x h"}},"toxicity":{{"DILI":"x","hERG":"x","Ames":"x","NR_ER":"x","NR_AR":"x"}},"docking":{{"strongest_target":"target","affinity":"x kcal/mol","pathway_relevance":"explanation"}},"pbpk_estimate":{{"T1_fetal_exposure":"estimate","T3_fetal_exposure":"estimate","placental_transfer":"Low/Moderate/High"}},"dart_prediction":"predicted DART signal","key_pathways":["pathway 1","pathway 2","pathway 3"],"hub_proteins":["protein 1","protein 2","protein 3"],"confidence":"Low/Medium/High — reasoning"}}"""
                    try:
                        client = anthropic.Anthropic(api_key=key_a)
                        resp = client.messages.create(
                            model=CLAUDE_MODEL, max_tokens=1400,
                            system="You are Gravitas AI, expert in obstetric pharmacology. Return ONLY valid JSON, no markdown backticks.",
                            messages=[{"role":"user","content":prompt}]
                        )
                        raw = resp.content[0].text.replace("```json","").replace("```","").strip()
                        result = json.loads(raw)
                        phi_ai = float(result.get("phi",0))
                        plabel2,pcls2,pcolor2,pbadge2,picon2 = phi_cls(phi_ai)
                        rl = result.get("risk_label", plabel2)
                        st.markdown(f"""<div class="risk-banner risk-{pcls2}">
                        <div class="risk-icon">{picon2}</div>
                        <div style="flex:1"><div class="risk-label" style="color:{pcolor2}">{rl}</div>
                        <div class="risk-sub">Gravitas AI prediction · Calibrated against validated database</div></div>
                        <div style="text-align:center"><div class="phi-number" style="color:{pcolor2}">{phi_ai:.0f}</div>
                        <div class="phi-label">PHI / 100</div></div>
                        </div>""", unsafe_allow_html=True)
                        ai_tabs = st.tabs(["📋 Summary","⏱️ Trimester Risk","💊 PK/ADME","☢️ Toxicity","⚗️ Docking","💉 PBPK","🕸️ Pathways"])
                        with ai_tabs[0]:
                            st.markdown(f'<div class="ai-bubble">{result.get("summary","")}</div>', unsafe_allow_html=True)
                            st.markdown('<div class="sec-hdr" style="margin-top:16px">Recommendations</div>', unsafe_allow_html=True)
                            for rec in result.get("recommendations",[]): st.markdown(f"> {rec.get('icon','')} {rec.get('text','')}")
                            conf = result.get("confidence","")
                            if conf: st.markdown(f'<div style="font-size:0.77rem;color:#718096;margin-top:12px;font-style:italic">🔬 Confidence: {conf}</div>', unsafe_allow_html=True)
                        with ai_tabs[1]:
                            tr = result.get("trimester_risk",{})
                            c1,c2,c3 = st.columns(3)
                            for col,tri,tc2 in [(c1,"T1","#b794f4"),(c2,"T2","#63b3ed"),(c3,"T3","#68d391")]:
                                with col: st.markdown(f'<div class="g-card" style="border-left:4px solid {tc2}"><div style="font-size:0.9rem;font-weight:700;color:{tc2};margin-bottom:8px">{tri}</div><div style="font-size:0.84rem;color:#a0aec0;line-height:1.6">{tr.get(tri,"N/A")}</div></div>', unsafe_allow_html=True)
                        with ai_tabs[2]:
                            adme_ai = result.get("adme",{})
                            c1,c2 = st.columns(2)
                            with c1:
                                for l,k in [("Mol. Weight","MW"),("LogP","LogP"),("Oral Absorption","HIA")]:
                                    st.metric(l, adme_ai.get(k,"N/A"))
                            with c2:
                                for l,k in [("BBB Penetration","BBB"),("Plasma Protein Binding","PPB"),("Half-Life","t_half")]:
                                    st.metric(l, adme_ai.get(k,"N/A"))
                        with ai_tabs[3]:
                            tox_ai = result.get("toxicity",{})
                            c1,c2 = st.columns(2)
                            with c1:
                                for l,k in [("DILI","DILI"),("hERG Cardiotox","hERG"),("Ames Mutagenicity","Ames")]:
                                    st.metric(l, tox_ai.get(k,"N/A"))
                            with c2:
                                for l,k in [("NR-ER (Estrogen)","NR_ER"),("NR-AR (Androgen)","NR_AR")]:
                                    st.metric(l, tox_ai.get(k,"N/A"))
                        with ai_tabs[4]:
                            dock_ai = result.get("docking",{})
                            st.markdown(f'''<div class="g-card g-card-gold">
                            <div class="sec-hdr">Strongest PTB Target</div>
                            <div style="font-size:1.3rem;font-weight:700;color:#f6ad55">{dock_ai.get("strongest_target","N/A")}</div>
                            <div style="font-size:1.1rem;color:#e2e8f0;margin:4px 0">{dock_ai.get("affinity","N/A")}</div>
                            <div style="font-size:0.84rem;color:#a0aec0;margin-top:8px">{dock_ai.get("pathway_relevance","")}</div>
                            </div>''', unsafe_allow_html=True)
                        with ai_tabs[5]:
                            pbpk_ai = result.get("pbpk_estimate",{})
                            c1,c2 = st.columns(2)
                            with c1:
                                st.metric("T1 Fetal Exposure (est.)", pbpk_ai.get("T1_fetal_exposure","N/A"))
                                st.metric("T3 Fetal Exposure (est.)", pbpk_ai.get("T3_fetal_exposure","N/A"))
                            with c2:
                                st.metric("Placental Transfer", pbpk_ai.get("placental_transfer","N/A"))
                                st.metric("DART Prediction", result.get("dart_prediction","N/A"))
                        with ai_tabs[6]:
                            c1,c2 = st.columns(2)
                            with c1:
                                st.markdown('<div class="sec-hdr">Key Pathways</div>', unsafe_allow_html=True)
                                for p in result.get("key_pathways",[]): st.markdown(f"• {p}")
                            with c2:
                                st.markdown('<div class="sec-hdr">Hub Proteins</div>', unsafe_allow_html=True)
                                for p in result.get("hub_proteins",[]): st.markdown(f"• {p}")
                        st.markdown('<div class="disclaimer">⚠️ AI-generated prediction calibrated against validated database. Not a substitute for experimental validation or clinical judgment.</div>', unsafe_allow_html=True)
                    except json.JSONDecodeError:
                        st.markdown(f'<div class="ai-bubble">🤖 {resp.content[0].text}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"AI error: {e}")


# ══════════════════════════════════════════════════════════════════
# PAGE: NETWORK PHARMACOLOGY
# ══════════════════════════════════════════════════════════════════
elif page == "🕸️ Network Pharmacology":
    st.markdown("## 🕸️ Network Pharmacology Explorer")
    st.markdown('<div style="font-size:0.85rem;color:#718096;margin-bottom:18px">KEGG · Reactome · GO enrichment · Hub gene STRING analysis</div>', unsafe_allow_html=True)
    drug_np = st.selectbox("Select Drug", DATA["drugs"])
    if drug_np:
        net = load_network_data(drug_np); figs = net.pop("figures",{})
        has_tables = any(k in net for k in ["kegg","reactome","go_bp","go_cc","go_mf","hub_genes"])
        if has_tables or figs:
            ntabs = st.tabs(["KEGG","Reactome","GO-BP","GO-CC","GO-MF","Hub Genes","Figures"])
            for nt,(k,title) in zip(ntabs[:-1],[("kegg","KEGG"),("reactome","Reactome"),("go_bp","GO BP"),("go_cc","GO CC"),("go_mf","GO MF"),("hub_genes","Hub Genes")]):
                with nt:
                    if k in net: st.dataframe(net[k],use_container_width=True)
                    else: st.info(f"📂 File not found — upload .csv or .xlsx to: data/{drug_np}/Network_Pharmacology/")
            with ntabs[-1]:
                if figs:
                    fc = st.columns(2)
                    for i,(k,lbl) in enumerate({"fig_dotplot_bp":"GO-BP Dot Plot","fig_dotplot_cc":"GO-CC Dot Plot","fig_dotplot_mf":"GO-MF Dot Plot","fig_enrichmap_kegg":"KEGG Enrichment Map","fig_enrichmap_reactome":"Reactome Enrichment Map"}.items()):
                        if k in figs:
                            with fc[i%2]:
                                st.markdown(f'<div class="sec-hdr">{lbl}</div>', unsafe_allow_html=True)
                                st.image(figs[k],use_container_width=True)
                else: st.info("📂 Place PNG figures in the Network_Pharmacology folder.")
        else:
            st.markdown(f'''<div class="g-card g-card-teal">
            <div class="sec-hdr">📂 Upload to repo: data/{drug_np}/Network_Pharmacology/</div>
            <div style="font-size:0.83rem;color:#a0aec0;line-height:2.0">
            <b style="color:#63b3ed">Tables (.csv or .xlsx):</b> <code>{drug_np}_KEGG</code> · <code>{drug_np}_Reactome.xlsx</code> · <code>{drug_np}_GO_BP.xlsx</code> · <code>{drug_np}_GO_CC.xlsx</code> · <code>{drug_np}_GO_MF.xlsx</code> · <code>{drug_np}_Hubgenes_STRING_centrality.xlsx</code> · <code>Commongenes_Venn.xlsx</code><br>
            <b style="color:#63b3ed">Figures:</b> <code>{drug_np}_EnrichmentMap_KEGG.png</code> · <code>{drug_np}_EnrichmentMap_Reactome.png</code> · <code>{drug_np}_Dotplot_GO_BP.png</code> · <code>{drug_np}_Dotplot_GO_CC.png</code> · <code>{drug_np}_Dotplot_GO_MF.png</code>
            </div></div>''', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: AI CONSULTATION
# ══════════════════════════════════════════════════════════════════
elif page == "🤖 AI Consultation":
    st.markdown("## 🤖 AI Consultation")
    st.markdown('<div style="font-size:0.85rem;color:#718096;margin-bottom:18px">Multi-drug comparison · Custom clinical questions · Clinician / Patient / Researcher views</div>', unsafe_allow_html=True)
    sel_drugs = st.multiselect("Select drugs to compare (up to 4)", DATA["drugs"], max_selections=4)
    user_type = st.selectbox("View As", ["👩‍⚕️ Clinician","🤰 Patient","🔬 Pharma Researcher"])
    custom_q  = st.text_input("Your question", placeholder="e.g. Which is safest in T1? Compare placental transfer. Best option for chronic use?")
    if st.button("🤖 Run AI Consultation", use_container_width=True) and sel_drugs:
        key_a = get_api_key()
        if not key_a: st.warning("API key not configured.")
        else:
            ctx = "DRUG COMPARISON DATA:\n"
            for d in sel_drugs:
                r = DATA["adme"].loc[d]; phi2 = compute_phi(r)
                dart_sig2 = DATA["dart_s"].loc[d,"Overall_DART_signal"] if d in DATA["dart_s"].index else "N/A"
                ctx += f"\n{d} | PHI:{phi2} | DART:{dart_sig2} | DILI:{safe(r.get('DILI'),3)} | hERG:{safe(r.get('hERG'),3)} | BBB:{safe(r.get('BBB'),3)} | LogP:{safe(r.get('logP'),3)} | MW:{safe(r.get('MW'),2)}"
            sys_map = {
                "👩‍⚕️ Clinician":"Clinical pharmacology expert. Direct, evidence-based, clinically actionable.",
                "🤰 Patient":"Patient-friendly pregnancy advisor. Plain, warm language.",
                "🔬 Pharma Researcher":"Pharmaceutical scientist. Technical, mechanistic, reference specific values."
            }
            with st.spinner("🤖 Gravitas AI consulting..."):
                try:
                    client = anthropic.Anthropic(api_key=key_a)
                    resp = client.messages.create(
                        model=CLAUDE_MODEL, max_tokens=1400,
                        system=sys_map.get(user_type, sys_map["👩‍⚕️ Clinician"]),
                        messages=[{"role":"user","content":f"{ctx}\n\nQuestion: {custom_q or 'Compare these drugs for pregnancy safety, highlight key differences and clinical recommendations.'}"}]
                    )
                    st.markdown(f'<div class="ai-bubble">🤖 {resp.content[0].text}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"AI error: {e}")
    st.markdown('<div class="disclaimer">⚠️ Research tool only. Not for clinical decision-making.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"""
        <img src="data:image/png;base64,{LOGO_B64}"
             style="width:160px; border-radius:12px;
                    filter:drop-shadow(0 4px 20px rgba(255,160,140,0.3))"/>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        ## Gravitas AI 2.0
        ### The Menon Laboratory · Perinatal Research
        *"It's About Saving Babies"*

        **Gravitas AI 2.0** is a pregnancy drug safety intelligence platform by The Menon Laboratory.
        It combines experimentally validated multi-modal data with AI-powered analysis for novel molecules.

        ---
        **Pipeline:** ADME/Tox · Molecular Docking (6 PTB targets) · PBPK (T1/T2/T3) · DART · Network Pharmacology

        **Parameters:** 175 ADME/tox features · 6 PTB docking targets · 7 CYP enzymes · PBPK fetal exposure modelling

        **PHI Formula:** DILI(0.25) + hERG(0.15) + Ames(0.10) + NR-ER(0.15) + NR-AR(0.10) + SR-p53(0.10) + BBB(0.15)

        **Data sources:** ProTox-II · SwissADME · DrugBank · ChEMBL · GTEx Placenta · AutoDock Vina
        """)
    st.markdown("---")
    st.caption("⚠️ Research and educational use only. Not validated for clinical decision-making.")
