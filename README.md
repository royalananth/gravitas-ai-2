# Gravitas AI 2.0
### Pregnancy Drug Safety Intelligence Platform
**The Menon Laboratory · Perinatal Research**
> *"It's About Saving Babies"*

---

## 🚀 Deployment (Streamlit Cloud)

### Step 1 — GitHub
1. Create a new GitHub repository called `gravitas-ai-2`
2. Upload ALL files from this folder maintaining the exact structure below
3. Make sure `data/` folder and all subfolders are included

### Step 2 — Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app** → connect your GitHub repo
3. Set **Main file**: `app.py`
4. Go to **Settings → Secrets** and add:
```
ANTHROPIC_API_KEY = "sk-ant-your-actual-key"
```
5. Deploy!

---

## 📁 Required Folder Structure

```
gravitas-ai-2/
├── app.py                          ← Main application
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── data/
│   ├── ADME_Toxicity_Moleculardocking_Parameters.xlsx
│   ├── DART_AI_ready_schema.xlsx
│   └── Pregnancy_PBPK_AI_Prototype.xlsx
├── Anastrozole/
│   └── Network_Pharmacology/
│       ├── Anastrozole_KEGG.xlsx
│       │   ├── Anastrozole_Reactome.xlsx
│       │   ├── Anastrozole_GO_BP.xlsx
│       │   ├── Anastrozole_GO_CC.xlsx
│       │   ├── Anastrozole_GO_MF.xlsx
│       │   ├── Anastrozole_Hubgenes_STRING_centrality.xlsx
│       │   ├── Commongenes_Venn.xlsx
│       │   ├── Anastrozole_Dotplot_GO_BP.png  (or .jpg/.svg)
│       │   ├── Anastrozole_Dotplot_GO_CC.png
│       │   ├── Anastrozole_Dotplot_GO_MF.png
│       │   ├── Anastrozole_EnrichmentMap_KEGG.png
│       │   └── Anastrozole_EnrichmentMap_Reactome.png
│       ├── Aspirin/
│       │   └── [same structure]
│       ├── Azithromycin/
│       ├── Cladribine/
│       ├── Epicatechin/
│       ├── Exemestane/
│       ├── Fluorouracil/
│       ├── Indomethacin/
│       ├── Leflunomide/
│       ├── Letrozole/
│       ├── Olaparib/
│       ├── Osthole/
│       ├── Pravastatin/
│       ├── Temozolomide/
│       ├── Thalidomide/
│       └── Valproic Acid sodium salt/
│           └── [same structure]
```

---

## 📂 Network Pharmacology File Naming

For each drug, name your files **exactly** as follows (case-sensitive):

| File | Content |
|------|---------|
| `DrugName_KEGG.xlsx` | KEGG pathway enrichment table |
| `DrugName_Reactome.xlsx` | Reactome pathway enrichment table |
| `DrugName_GO_BP.xlsx` | GO Biological Process table |
| `DrugName_GO_CC.xlsx` | GO Cellular Component table |
| `DrugName_GO_MF.xlsx` | GO Molecular Function table |
| `DrugName_Hubgenes_STRING_centrality.xlsx` | Hub genes with centrality scores |
| `Commongenes_Venn.xlsx` | Common genes from Venn analysis |
| `DrugName_Dotplot_GO_BP.png` | GO-BP dot plot figure |
| `DrugName_Dotplot_GO_CC.png` | GO-CC dot plot figure |
| `DrugName_Dotplot_GO_MF.png` | GO-MF dot plot figure |
| `DrugName_EnrichmentMap_KEGG.png` | KEGG enrichment map figure |
| `DrugName_EnrichmentMap_Reactome.png` | Reactome enrichment map figure |

> ⚠️ **Important:** Use the exact drug name as it appears in the database:
> `Valproic Acid sodium salt` (not `Valproic_Acid` or `VPA`)

---

## 🧪 Adding In Vitro Cytokine Data (Future)

When ready, add a new Excel file: `data/InVitro_Cytokine_Viability.xlsx`

Expected columns:
```
Drug_name | Concentration_uM | IL6_inhibition_pct | IL8_inhibition_pct | 
TNFa_inhibition_pct | Cell_viability_pct | Assay_type | Cell_line | Notes
```

The app will automatically detect and display this data when present.

---

## 🏗️ What's Included in v2.0

| Feature | Details |
|---------|---------|
| **16 Validated Drugs** | Full multi-modal annotation |
| **175 ADME Parameters** | From ADMET Lab 3.0 |
| **50+ Toxicity Endpoints** | From ProTox 3.0 |
| **6 Docking Targets** | P38, NF-κB, MAPK, JAK2, TGFβR1, HIF1α |
| **DART Studies** | 27 reproductive endpoints, raw evidence |
| **Pregnancy PBPK** | T1/T2/T3 maternal + fetal exposure |
| **Network Pharmacology** | KEGG, Reactome, GO-BP/CC/MF, Hub genes |
| **3 User Profiles** | Clinician · Patient · Pharma Researcher |
| **AI Analysis** | Claude-powered profile-specific analysis |
| **PHI Score** | Composite Pregnancy Hazard Index |

---

## 🔑 API Key Setup

### Local development
Create `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

### Streamlit Cloud
Settings → Secrets → paste:
```
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

---

*Gravitas AI 2.0 · The Menon Laboratory · SRI Translational AI Hackathon 2026*
