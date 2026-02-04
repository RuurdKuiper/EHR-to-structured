"""
EHR to OMOP CDM Demo - Streamlit App
Extracts structured data from Dutch EHR discharge notes using OpenAI
and converts to OMOP Common Data Model format.
"""

import json
import os
import re
import requests
import streamlit as st
import pandas as pd
import torch
from functools import lru_cache

# ============================================================================
# Configuration
# ============================================================================

# System prompt for extraction
SYSTEM_PROMPT = """You extract structured clinical information from Dutch EHR discharge notes and cite the exact text spans you used.

Return ONLY valid JSON with the following structure:

{
    "age": <int or null>,
    "sex": "<male|female|other|unknown>",
    "height": {"value": <number or null>, "unit": "<unit or null>"},
    "weight": {"value": <number or null>, "unit": "<unit or null>"},
    "bmi": {"value": <number or null>, "unit": "<unit or null>"},
    "smoking_status": "<never|former|current|null>",
    "conditions": [
        {
            "name": "<name>"
        }
    ],
    "recent_labs": {
        "hbA1c": {"value": <number or null>, "unit": "<unit or null>"},
        "creatinine": {"value": <number or null>, "unit": "<unit or null>"},
        "ldl_cholesterol": {"value": <number or null>, "unit": "<unit or null>"}
    },
    "vital_signs": {
        "blood_pressure": {"value": "<string or null>", "unit": "<unit or null>"},
        "heart_rate": {"value": <number or null>, "unit": "<unit or null>"},
        "oxygen_saturation": {"value": <number or null>, "unit": "<unit or null>"}
    },
    "evidence": [
        {
            "field": "<field name>",
            "value": "<value you returned>",
            "quote": "<exact verbatim substring copied from the note>"
        }
    ]
}

Important rules:
- For each evidence item, copy the EXACT verbatim substring from the note into the "quote" field. Do not paraphrase or modify it.
- Use these exact field names in evidence: "age", "sex", "height", "weight", "bmi", "smoking_status", "vital_signs.blood_pressure", "vital_signs.heart_rate", "vital_signs.oxygen_saturation", "recent_labs.hbA1c", "recent_labs.creatinine", "recent_labs.ldl_cholesterol", "conditions".
- Extract the unit EXACTLY as written in the note (e.g., "mmol/L", "mg/dL", "µmol/L", "cm", "kg", "%", "bpm").
- If the discharge note does not mention a piece of information, return null (and no evidence for that field).
- If the discharge note DOES mention a condition, it MUST appear in the JSON.
- There is only ONE condition per patient.
- The name of the condition should be noted in ENGLISH.
- Do NOT infer or guess missing values.
- Do NOT add fields, comments, or explanations.
- Respond with JSON only."""

# Field colors for highlighting
FIELD_COLORS = {
    'age': '#ffd166',
    'sex': '#ff6b9d',
    'height': '#74c0fc',
    'weight': '#63e6be',
    'bmi': '#8ce99a',
    'smoking_status': '#e599f7',
    'vital_signs.blood_pressure': '#ffa94d',
    'vital_signs.heart_rate': '#ff8787',
    'vital_signs.oxygen_saturation': '#b197fc',
    'recent_labs.hbA1c': '#fcc2d7',
    'recent_labs.creatinine': '#99e9f2',
    'recent_labs.ldl_cholesterol': '#a5d8ff',
    'conditions': '#ffec99'
}

# ============================================================================
# Data Loading
# ============================================================================

@st.cache_data
def load_sample_data():
    """Load sample patient data."""
    try:
        df = pd.read_json("ehr_val_sampled.jsonl", lines=True)
        return df
    except Exception as e:
        st.error(f"Could not load sample data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_code_reference():
    """Load ICD10/SNOMED reference data."""
    try:
        df = pd.read_parquet("ICD10_Snomed.parquet")
        return df
    except Exception as e:
        st.warning(f"Could not load code reference: {e}")
        return None

# ============================================================================
# Athena API for OMOP Concepts
# ============================================================================

@st.cache_data(ttl=3600)
def lookup_omop_concept(search_term, vocabulary_id=None, domain_id=None):
    """Look up an OMOP concept using the Athena API."""
    try:
        url = "https://athena.ohdsi.org/api/v1/concepts"
        params = {
            "query": search_term,
            "pageSize": 5,
            "page": 1,
            "standardConcept": "Standard"
        }
        if vocabulary_id:
            params["vocabulary"] = vocabulary_id
        if domain_id:
            params["domain"] = domain_id
            
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            concepts = data.get("content", [])
            if concepts:
                c = concepts[0]
                return {
                    "concept_id": c.get("id"),
                    "concept_name": c.get("name"),
                    "concept_code": c.get("code"),
                    "vocabulary_id": c.get("vocabulary", {}).get("id"),
                    "domain_id": c.get("domain", {}).get("id"),
                }
        return None
    except Exception:
        return None

def get_omop_concept_id(search_term, vocabulary_id=None, domain_id=None, fallback_id=0):
    """Get OMOP concept ID with fallback."""
    result = lookup_omop_concept(search_term, vocabulary_id, domain_id)
    if result and result.get("concept_id"):
        return result["concept_id"]
    return fallback_id

# ============================================================================
# Embedding-based SNOMED Matching
# ============================================================================

@st.cache_resource
def load_embedder():
    """Load sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)
    except Exception as e:
        st.warning(f"Could not load embedding model: {e}")
        return None

@st.cache_data
def compute_embeddings(_embedder, descriptions, cache_path):
    """Compute or load cached embeddings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location=device, weights_only=True)
    else:
        embeddings = _embedder.encode(descriptions, convert_to_tensor=True, device=device)
        torch.save(embeddings, cache_path)
        return embeddings

def map_condition_to_snomed(condition_name, code_ref, embedder, icd10_embeddings, code_ref_icd10_unique, top_k=3):
    """Map condition to SNOMED via ICD10 embeddings."""
    if embedder is None or condition_name is None or not condition_name.strip():
        return []
    
    try:
        from sentence_transformers import util
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        query_emb = embedder.encode(condition_name, convert_to_tensor=True, device=device)
        icd_scores = util.cos_sim(query_emb, icd10_embeddings)[0]
        icd_top = torch.topk(icd_scores, k=min(top_k, len(icd_scores)))
        
        results = []
        for score, idx in zip(icd_top.values.tolist(), icd_top.indices.tolist()):
            row = code_ref_icd10_unique.iloc[idx]
            icd_code = str(row.get("icd10_code", ""))
            
            # Map ICD10 to SNOMED
            mapped = code_ref[code_ref["icd10_code"] == icd_code]
            if not mapped.empty:
                first = mapped.iloc[0]
                results.append({
                    "snomed_code": str(first.get("snomed_code", "")),
                    "snomed_description": first.get("snomed_description", ""),
                    "icd10_code": icd_code,
                    "icd10_description": row.get("icd10_description", ""),
                    "similarity": round(float(score), 3)
                })
        return results
    except Exception as e:
        st.warning(f"SNOMED matching error: {e}")
        return []

# ============================================================================
# OpenAI Extraction
# ============================================================================

def extract_with_openai(note, api_key):
    """Extract structured data using OpenAI."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"NOTE:\n{note}"}
            ],
            response_format={"type": "json_object"},
        )
        
        output = response.choices[0].message.content
        parsed = json.loads(output)
        
        # Normalize evidence - find quotes in original text
        evidence = parsed.get("evidence", [])
        normalized = []
        for ev in evidence:
            quote = ev.get("quote", "").strip()
            if quote:
                idx = note.lower().find(quote.lower())
                if idx >= 0:
                    normalized.append({
                        "field": ev.get("field"),
                        "value": ev.get("value"),
                        "text": note[idx:idx + len(quote)],
                        "start": idx,
                        "end": idx + len(quote),
                        "source": "openai"
                    })
        parsed["evidence"] = normalized
        return parsed
        
    except Exception as e:
        st.error(f"OpenAI extraction error: {e}")
        return None

# ============================================================================
# OMOP Conversion
# ============================================================================

def convert_to_omop(extracted_data, patient_id, snomed_matches):
    """Convert extracted data to OMOP CDM format."""
    
    def get_val_unit(obj, default_unit=None):
        if obj is None:
            return None, default_unit
        if isinstance(obj, dict):
            return obj.get("value"), obj.get("unit", default_unit)
        return obj, default_unit
    
    # PERSON
    male_id = get_omop_concept_id("Male", "Gender", "Gender", 8507)
    female_id = get_omop_concept_id("Female", "Gender", "Gender", 8532)
    
    person = {
        "person_id": patient_id,
        "gender_concept_id": male_id if extracted_data.get("sex") == "male" else female_id if extracted_data.get("sex") == "female" else 0,
        "gender_source_value": extracted_data.get("sex"),
        "year_of_birth": 2026 - extracted_data.get("age") if extracted_data.get("age") else None,
    }
    
    # CONDITIONS
    conditions = []
    if snomed_matches:
        best = snomed_matches[0]
        conditions.append({
            "condition_occurrence_id": 1,
            "person_id": patient_id,
            "condition_concept_id": best.get("snomed_code", ""),
            "condition_source_value": extracted_data.get("conditions", [{}])[0].get("name", "") if extracted_data.get("conditions") else "",
            "condition_source_concept_id": best.get("icd10_code", ""),
        })
    
    # MEASUREMENTS
    measurements = []
    mid = 1
    
    labs = extracted_data.get("recent_labs", {}) or {}
    vitals = extracted_data.get("vital_signs", {}) or {}
    
    lab_mappings = [
        ("hbA1c", "Hemoglobin A1c", "LOINC", 3004410, "%"),
        ("creatinine", "Creatinine serum", "LOINC", 3016723, "µmol/L"),
        ("ldl_cholesterol", "LDL Cholesterol", "LOINC", 3028437, "mmol/L"),
    ]
    
    vital_mappings = [
        ("heart_rate", "Heart rate", "LOINC", 3027018, "bpm"),
        ("oxygen_saturation", "Oxygen saturation", "LOINC", 3016502, "%"),
    ]
    
    for key, search, vocab, fallback, default_unit in lab_mappings:
        val, unit = get_val_unit(labs.get(key), default_unit)
        if val is not None:
            measurements.append({
                "measurement_id": mid,
                "person_id": patient_id,
                "measurement_concept_id": get_omop_concept_id(search, vocab, "Measurement", fallback),
                "measurement_source_value": key,
                "value_as_number": val,
                "unit_source_value": unit
            })
            mid += 1
    
    for key, search, vocab, fallback, default_unit in vital_mappings:
        val, unit = get_val_unit(vitals.get(key), default_unit)
        if val is not None:
            measurements.append({
                "measurement_id": mid,
                "person_id": patient_id,
                "measurement_concept_id": get_omop_concept_id(search, vocab, "Measurement", fallback),
                "measurement_source_value": key,
                "value_as_number": val,
                "unit_source_value": unit
            })
            mid += 1
    
    # OBSERVATIONS
    observations = []
    oid = 1
    
    obs_mappings = [
        ("height", "Body height", 3036277, "cm"),
        ("weight", "Body weight", 3025315, "kg"),
        ("bmi", "Body mass index", 3038553, "kg/m²"),
    ]
    
    for key, search, fallback, default_unit in obs_mappings:
        val, unit = get_val_unit(extracted_data.get(key), default_unit)
        if val is not None:
            observations.append({
                "observation_id": oid,
                "person_id": patient_id,
                "observation_concept_id": get_omop_concept_id(search, "LOINC", "Measurement", fallback),
                "observation_source_value": key,
                "value_as_number": val,
                "unit_source_value": unit
            })
            oid += 1
    
    # Smoking status
    smoking = extracted_data.get("smoking_status")
    if smoking:
        smoking_concepts = {
            "never": ("Never smoker", 4144272),
            "former": ("Ex-smoker", 4310250),
            "current": ("Current smoker", 4298794),
        }
        if smoking in smoking_concepts:
            search, fallback = smoking_concepts[smoking]
            observations.append({
                "observation_id": oid,
                "person_id": patient_id,
                "observation_concept_id": get_omop_concept_id(search, "SNOMED", "Observation", fallback),
                "observation_source_value": f"smoking_status: {smoking}",
                "value_as_number": None,
                "unit_source_value": None
            })
    
    return {
        "PERSON": [person],
        "CONDITION_OCCURRENCE": conditions,
        "MEASUREMENT": measurements,
        "OBSERVATION": observations
    }

# ============================================================================
# UI Helpers
# ============================================================================

def render_highlighted_text(note, evidence):
    """Render note with highlighted evidence spans."""
    if not evidence:
        return f"<div style='white-space: pre-wrap; font-family: monospace; font-size: 0.9rem;'>{note}</div>"
    
    # Sort and dedupe evidence
    spans = sorted([e for e in evidence if e.get("start") is not None], key=lambda x: x["start"])
    
    # Remove overlaps
    filtered = []
    last_end = -1
    for ev in spans:
        if ev["start"] >= last_end:
            filtered.append(ev)
            last_end = ev["end"]
    
    # Build HTML
    html = ""
    cursor = 0
    for ev in filtered:
        if ev["start"] > cursor:
            html += note[cursor:ev["start"]]
        color = FIELD_COLORS.get(ev.get("field"), "#ffe066")
        html += f'<span style="background: {color}; padding: 2px 4px; border-radius: 3px;" title="{ev.get("field")}">{note[ev["start"]:ev["end"]]}</span>'
        cursor = ev["end"]
    if cursor < len(note):
        html += note[cursor:]
    
    return f"<div style='white-space: pre-wrap; font-family: monospace; font-size: 0.85rem; line-height: 1.6;'>{html}</div>"

def format_value(val, unit=None, field=None):
    """Format a value with optional unit and color."""
    if val is None:
        return "—"
    
    text = f"{val}"
    if unit:
        text += f" {unit}"
    
    if field and field in FIELD_COLORS:
        color = FIELD_COLORS[field]
        return f'<span style="background: {color}; padding: 2px 6px; border-radius: 4px;">{text}</span>'
    return text

def get_val_unit_display(obj, default_unit=""):
    """Get value and unit for display."""
    if obj is None:
        return None, default_unit
    if isinstance(obj, dict):
        return obj.get("value"), obj.get("unit", default_unit)
    return obj, default_unit

# ============================================================================
# Main App
# ============================================================================

def main():
    st.set_page_config(
        page_title="EHR to OMOP CDM Demo",
        page_icon="🏥",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #2c3e50, #3498db);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stDataFrame { font-size: 0.8rem; }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header"><h1>🏥 EHR to OMOP CDM Demo</h1><p>Extract structured data from Dutch discharge notes</p></div>', unsafe_allow_html=True)
    
    # API Key in sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Try to get API key from secrets first, then allow manual input
        api_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, 'secrets') else ""
        if not api_key:
            api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
        else:
            st.success("✓ API key loaded from secrets")
        
        st.divider()
        st.markdown("### About")
        st.markdown("""
        This demo extracts clinical information from Dutch EHR discharge notes and converts it to OMOP CDM format.
        
        **Features:**
        - 🤖 EHR extraction
        - 🔍 Evidence highlighting
        - 🗄️ OMOP CDM conversion
        - 🏷️ ICD10→SNOMED mapping
        """)
    
    # Load data
    df = load_sample_data()
    code_ref = load_code_reference()
    
    if df.empty:
        st.error("No sample data available")
        return
    
    # Prepare embeddings for SNOMED matching
    embedder = load_embedder()
    icd10_embeddings = None
    code_ref_icd10_unique = None
    
    if code_ref is not None and embedder is not None:
        code_ref_icd10_unique = code_ref.dropna(subset=["icd10_description"]).drop_duplicates(subset=["icd10_description"]).reset_index(drop=True)
        descriptions = code_ref_icd10_unique["icd10_description"].astype(str).tolist()
        icd10_embeddings = compute_embeddings(embedder, descriptions, "icd10_embeddings.pt")
    
    # Patient selector
    patient_options = {f"Patient {row['patient_id']} - {row.get('sex', '?')}, age {row.get('age', '?')}": row['patient_id'] 
                       for _, row in df.iterrows()}
    selected_label = st.selectbox("Select a patient", list(patient_options.keys()))
    selected_id = patient_options[selected_label]
    patient = df[df["patient_id"] == selected_id].iloc[0]
    
    # Three columns
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Column 1: Discharge Note
    with col1:
        st.subheader("📋 Discharge Note")
        note_container = st.container()
        
    # Column 2: Extracted Data
    with col2:
        st.subheader("🤖 Extracted Data")
        extract_btn = st.button("🔍 Extract Details", use_container_width=True, disabled=not api_key)
        extract_container = st.container()
        
    # Column 3: OMOP Tables
    with col3:
        st.subheader("🗄️ OMOP CDM")
        omop_container = st.container()
    
    # Initialize state
    if "extracted" not in st.session_state:
        st.session_state.extracted = None
        st.session_state.snomed_matches = []
        st.session_state.omop = None
    
    # Show note (will update with highlights after extraction)
    with note_container:
        if st.session_state.extracted and st.session_state.extracted.get("evidence"):
            st.markdown(render_highlighted_text(patient["discharge_note"], st.session_state.extracted["evidence"]), unsafe_allow_html=True)
        else:
            st.text_area("", patient["discharge_note"], height=500, disabled=True, label_visibility="collapsed")
    
    # Extract button handler
    if extract_btn and api_key:
        with st.spinner("Extracting..."):
            extracted = extract_with_openai(patient["discharge_note"], api_key)
            if extracted:
                st.session_state.extracted = extracted
                
                # SNOMED matching
                if extracted.get("conditions") and code_ref is not None and embedder is not None:
                    condition_name = extracted["conditions"][0].get("name", "")
                    st.session_state.snomed_matches = map_condition_to_snomed(
                        condition_name, code_ref, embedder, icd10_embeddings, code_ref_icd10_unique
                    )
                
                # OMOP conversion
                st.session_state.omop = convert_to_omop(extracted, selected_id, st.session_state.snomed_matches)
                st.rerun()
    
    # Show extracted data
    with extract_container:
        if st.session_state.extracted:
            data = st.session_state.extracted
            
            # Demographics
            st.markdown("**👤 Demographics**")
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"Age: {format_value(data.get('age'), field='age')}", unsafe_allow_html=True)
                h_val, h_unit = get_val_unit_display(data.get('height'), 'cm')
                st.markdown(f"Height: {format_value(h_val, h_unit, 'height')}", unsafe_allow_html=True)
                bmi_val, bmi_unit = get_val_unit_display(data.get('bmi'), 'kg/m²')
                st.markdown(f"BMI: {format_value(bmi_val, bmi_unit, 'bmi')}", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"Sex: {format_value(data.get('sex'), field='sex')}", unsafe_allow_html=True)
                w_val, w_unit = get_val_unit_display(data.get('weight'), 'kg')
                st.markdown(f"Weight: {format_value(w_val, w_unit, 'weight')}", unsafe_allow_html=True)
                st.markdown(f"Smoking: {format_value(data.get('smoking_status'), field='smoking_status')}", unsafe_allow_html=True)
            
            # Condition
            if data.get("conditions"):
                st.markdown("**🏥 Condition**")
                cond = data["conditions"][0].get("name", "Unknown")
                st.markdown(f'{format_value(cond, field="conditions")}', unsafe_allow_html=True)
                
                if st.session_state.snomed_matches:
                    st.markdown("*SNOMED Matches (via ICD10):*")
                    for m in st.session_state.snomed_matches[:3]:
                        st.markdown(f"- **{m['icd10_code']}**: {m['icd10_description'][:50]}... ({m['similarity']*100:.0f}%)")
            
            # Labs
            labs = data.get("recent_labs", {}) or {}
            if any(labs.get(k) for k in ["hbA1c", "creatinine", "ldl_cholesterol"]):
                st.markdown("**🧪 Labs**")
                hba1c_val, hba1c_unit = get_val_unit_display(labs.get("hbA1c"), "%")
                creat_val, creat_unit = get_val_unit_display(labs.get("creatinine"), "µmol/L")
                ldl_val, ldl_unit = get_val_unit_display(labs.get("ldl_cholesterol"), "mmol/L")
                st.markdown(f"HbA1c: {format_value(hba1c_val, hba1c_unit, 'recent_labs.hbA1c')}", unsafe_allow_html=True)
                st.markdown(f"Creatinine: {format_value(creat_val, creat_unit, 'recent_labs.creatinine')}", unsafe_allow_html=True)
                st.markdown(f"LDL: {format_value(ldl_val, ldl_unit, 'recent_labs.ldl_cholesterol')}", unsafe_allow_html=True)
            
            # Vitals
            vitals = data.get("vital_signs", {}) or {}
            if any(vitals.get(k) for k in ["blood_pressure", "heart_rate", "oxygen_saturation"]):
                st.markdown("**💓 Vitals**")
                bp_val, bp_unit = get_val_unit_display(vitals.get("blood_pressure"), "mmHg")
                hr_val, hr_unit = get_val_unit_display(vitals.get("heart_rate"), "bpm")
                spo2_val, spo2_unit = get_val_unit_display(vitals.get("oxygen_saturation"), "%")
                st.markdown(f"BP: {format_value(bp_val, bp_unit, 'vital_signs.blood_pressure')}", unsafe_allow_html=True)
                st.markdown(f"HR: {format_value(hr_val, hr_unit, 'vital_signs.heart_rate')}", unsafe_allow_html=True)
                st.markdown(f"SpO2: {format_value(spo2_val, spo2_unit, 'vital_signs.oxygen_saturation')}", unsafe_allow_html=True)
        else:
            st.info("Click 'Extract Details' to analyze the discharge note")
    
    # Show OMOP tables
    with omop_container:
        if st.session_state.omop:
            omop = st.session_state.omop
            
            for table_name, records in omop.items():
                if records:
                    with st.expander(f"**{table_name}** ({len(records)} rows)", expanded=True):
                        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
        else:
            st.info("OMOP tables will appear here after extraction")

if __name__ == "__main__":
    main()
