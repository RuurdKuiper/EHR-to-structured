"""
Interactive EHR to OMOP CDM Demo Web Application
Flask backend for extracting structured data from Dutch EHR discharge notes
and converting to OMOP Common Data Model format.
"""

import json
import os
import re
import requests
from functools import lru_cache
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request
import pandas as pd
import torch

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# ============================================================================
# Configuration
# ============================================================================

USE_LOCAL_MODEL = True  # Toggle between local GLiNER and OpenAI

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VAL_DATA_PATH = os.path.join(BASE_DIR, "ehr_val_sampled.jsonl")
CODE_REF_PATH = os.path.join(BASE_DIR, "ICD10_Snomed.parquet")
SNOMED_EMBEDDINGS_CACHE = os.path.join(BASE_DIR, "snomed_embeddings.pt")
ICD10_EMBEDDINGS_CACHE = os.path.join(BASE_DIR, "icd10_embeddings.pt")

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

# GLiNER entity labels for extraction (Dutch + English for better coverage)
GLINER_LABELS = [
    "leeftijd",           # age in Dutch
    "geslacht",           # sex in Dutch  
    "lengte in cm",       # height
    "gewicht in kg",      # weight
    "BMI waarde",         # BMI value
    "rookstatus",         # smoking status
    "diagnose",           # diagnosis
    "aandoening",         # condition/disease
    "ziekte",             # disease
    "HbA1c percentage",   # HbA1c lab
    "creatinine waarde",  # creatinine lab
    "LDL cholesterol",    # LDL
    "bloeddruk",          # blood pressure
    "hartfrequentie",     # heart rate
    "zuurstofsaturatie",  # oxygen saturation
]

# ============================================================================
# Global variables (loaded on startup)
# ============================================================================

val_df = None
code_ref = None
code_ref_unique = None  # Deduplicated by SNOMED description for embedding matching
code_ref_icd10_unique = None  # Deduplicated by ICD10 description for embedding matching
snomed_embeddings = None
icd10_embeddings = None
embedder = None
gliner_model = None

# Cache for Athena concept lookups (persists during server runtime)
athena_concept_cache = {}


@lru_cache(maxsize=100)
def lookup_omop_concept(search_term, vocabulary_id=None, domain_id=None):
    """
    Look up an OMOP concept using the Athena API.
    
    Args:
        search_term: The term to search for (e.g., "HbA1c", "body height")
        vocabulary_id: Optional vocabulary filter (e.g., "LOINC", "SNOMED")
        domain_id: Optional domain filter (e.g., "Measurement", "Observation")
    
    Returns:
        dict with concept_id, concept_name, vocabulary_id, domain_id
        or None if not found
    """
    try:
        # Athena API endpoint
        url = "https://athena.ohdsi.org/api/v1/concepts"
        
        params = {
            "query": search_term,
            "pageSize": 5,
            "page": 1,
            "standardConcept": "Standard"  # Only standard concepts
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
                # Return the best match (first result)
                c = concepts[0]
                result = {
                    "concept_id": c.get("id"),
                    "concept_name": c.get("name"),
                    "concept_code": c.get("code"),
                    "vocabulary_id": c.get("vocabulary", {}).get("id"),
                    "domain_id": c.get("domain", {}).get("id"),
                }
                print(f"[ATHENA] Found concept for '{search_term}': {result['concept_id']} - {result['concept_name']}")
                return result
        
        print(f"[ATHENA] No concept found for '{search_term}'")
        return None
        
    except Exception as e:
        print(f"[ATHENA] API error for '{search_term}': {e}")
        return None


def get_omop_concept_id(search_term, vocabulary_id=None, domain_id=None, fallback_id=0):
    """
    Get OMOP concept ID with fallback to hardcoded value if API fails.
    Uses caching to avoid repeated API calls.
    """
    cache_key = f"{search_term}|{vocabulary_id}|{domain_id}"
    
    if cache_key in athena_concept_cache:
        return athena_concept_cache[cache_key]
    
    result = lookup_omop_concept(search_term, vocabulary_id, domain_id)
    
    if result and result.get("concept_id"):
        concept_id = result["concept_id"]
    else:
        concept_id = fallback_id
        print(f"[ATHENA] Using fallback concept ID {fallback_id} for '{search_term}'")
    
    athena_concept_cache[cache_key] = concept_id
    return concept_id


def load_data():
    """Load validation data and code reference on startup."""
    global val_df, code_ref, code_ref_unique, code_ref_icd10_unique
    
    print("Loading validation data...")
    val_df = pd.read_json(VAL_DATA_PATH, lines=True)
    print(f"Loaded {len(val_df)} patients")
    
    print("Loading ICD10/SNOMED reference...")
    code_ref = pd.read_parquet(CODE_REF_PATH)
    print(f"Loaded {len(code_ref)} code mappings")
    
    # Deduplicate by SNOMED description for embedding matching
    code_ref_unique = code_ref.drop_duplicates(subset=["snomed_description"]).reset_index(drop=True)
    print(f"Unique SNOMED descriptions: {len(code_ref_unique)}")

    # Deduplicate by ICD10 description for embedding matching
    code_ref_icd10_unique = (
        code_ref.dropna(subset=["icd10_description"])
        .drop_duplicates(subset=["icd10_description"])
        .reset_index(drop=True)
    )
    print(f"Unique ICD10 descriptions: {len(code_ref_icd10_unique)}")


def load_embedder():
    """Load sentence transformer for SNOMED matching with caching."""
    global embedder, snomed_embeddings, icd10_embeddings, code_ref_unique, code_ref_icd10_unique
    
    from sentence_transformers import SentenceTransformer
    
    print("Loading embedding model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)
    
    # SNOMED embeddings
    if os.path.exists(SNOMED_EMBEDDINGS_CACHE):
        print(f"Loading cached SNOMED embeddings from {SNOMED_EMBEDDINGS_CACHE}...")
        snomed_embeddings = torch.load(SNOMED_EMBEDDINGS_CACHE, map_location=device, weights_only=True)
        print(f"Loaded {snomed_embeddings.shape[0]} cached embeddings!")
    else:
        print("Pre-computing SNOMED embeddings (first time only)...")
        snomed_descriptions = code_ref_unique["snomed_description"].astype(str).tolist()
        snomed_embeddings = embedder.encode(snomed_descriptions, convert_to_tensor=True, device=device)
        print(f"Saving embeddings to cache: {SNOMED_EMBEDDINGS_CACHE}")
        torch.save(snomed_embeddings, SNOMED_EMBEDDINGS_CACHE)
        print("Embeddings cached for future use!")

    # ICD10 embeddings
    if os.path.exists(ICD10_EMBEDDINGS_CACHE):
        print(f"Loading cached ICD10 embeddings from {ICD10_EMBEDDINGS_CACHE}...")
        icd10_embeddings = torch.load(ICD10_EMBEDDINGS_CACHE, map_location=device, weights_only=True)
        print(f"Loaded {icd10_embeddings.shape[0]} cached embeddings!")
    else:
        print("Pre-computing ICD10 embeddings (first time only)...")
        icd10_descriptions = code_ref_icd10_unique["icd10_description"].astype(str).tolist()
        icd10_embeddings = embedder.encode(icd10_descriptions, convert_to_tensor=True, device=device)
        print(f"Saving embeddings to cache: {ICD10_EMBEDDINGS_CACHE}")
        torch.save(icd10_embeddings, ICD10_EMBEDDINGS_CACHE)
        print("Embeddings cached for future use!")


def load_gliner_model():
    """Load GLiNER2 model for named entity recognition."""
    global gliner_model
    
    from gliner import GLiNER
    
    print("Loading GLiNER2 model...")
    gliner_model = GLiNER.from_pretrained("gliner-community/gliner_small-v2.5")
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gliner_model = gliner_model.to(device)
    
    print("GLiNER2 model ready!")


# ============================================================================
# Helper functions
# ============================================================================

def build_prompt(note):
    """Build the full prompt for the LLM."""
    return f"{SYSTEM_PROMPT}\n\nNOTE:\n{note}\n\nJSON:"


def safe_json(text):
    """Safely parse JSON from LLM output."""
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        return json.loads(text[start:end+1])
    except:
        return None


def parse_number(text):
    """Extract a number from text, handling European decimal format."""
    if text is None:
        return None
    text_str = str(text).strip()
    
    # Find numbers with European format (comma as decimal separator)
    # Match patterns like: 92,3 or 92.3 or 92 or 6,5%
    match = re.search(r'(\d+[,.]\d+|\d+)', text_str)
    if match:
        num_str = match.group(1).replace(',', '.')
        try:
            return float(num_str)
        except:
            return None
    return None


def parse_blood_pressure(text):
    """Extract blood pressure in format systolic/diastolic."""
    if text is None:
        return None
    # Match patterns like 133/92 or 133/92 mmHg
    match = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})', str(text))
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return str(text)


def extract_with_regex(note):
    """Extract structured data using regex patterns for Dutch medical text."""
    result = {
        "age": None,
        "sex": None,
        "height": {"value": None, "unit": None},
        "weight": {"value": None, "unit": None},
        "bmi": {"value": None, "unit": None},
        "smoking_status": None,
        "conditions": [],
        "recent_labs": {
            "hbA1c": {"value": None, "unit": None},
            "creatinine": {"value": None, "unit": None},
            "ldl_cholesterol": {"value": None, "unit": None}
        },
        "vital_signs": {
            "blood_pressure": {"value": None, "unit": None},
            "heart_rate": {"value": None, "unit": None},
            "oxygen_saturation": {"value": None, "unit": None}
        },
        "evidence": []  # list of {field, value, start, end, text, source}
    }

    def add_evidence(field, value, span, source="regex"):
        if span is None:
            return
        start, end = span
        result["evidence"].append({
            "field": field,
            "value": value,
            "start": int(start),
            "end": int(end),
            "text": note[start:end],
            "source": source
        })
    
    note_lower = note.lower()
    
    # Sex detection
    sex_match_m = re.search(r"(mannelijke patiënt|mannelijk|\bman[,\s])", note_lower)
    sex_match_f = re.search(r"(vrouwelijke patiënt|vrouwelijk|\bvrouw[,\s]|mevrouw)", note_lower)
    if sex_match_m:
        result["sex"] = "male"
        add_evidence("sex", "male", sex_match_m.span())
    elif sex_match_f:
        result["sex"] = "female"
        add_evidence("sex", "female", sex_match_f.span())
    
    # Age - patterns like "32 jaar" or "leeftijd 32"
    age_match = re.search(r'(\d{1,3})\s*jaar', note_lower)
    if age_match:
        result["age"] = int(age_match.group(1))
        add_evidence("age", result["age"], age_match.span())
    
    # Height - patterns like "lengte 177 cm" or "177 cm"
    height_match = re.search(r'lengte\s*[:\s]*(\d{2,3})\s*(cm|m)', note_lower)
    if height_match:
        result["height"]["value"] = float(height_match.group(1))
        result["height"]["unit"] = height_match.group(2)
        add_evidence("height", result["height"]["value"], height_match.span())
    
    # Weight - patterns like "gewicht 92,3 kg" or "92,3 kg"
    weight_match = re.search(r'gewicht\s*[:\s]*(\d{2,3}[,.]?\d*)\s*(kg|g|lbs?)', note_lower)
    if weight_match:
        result["weight"]["value"] = float(weight_match.group(1).replace(',', '.'))
        result["weight"]["unit"] = weight_match.group(2)
        add_evidence("weight", result["weight"]["value"], weight_match.span())
    
    # BMI - patterns like "BMI 29,5" or "BMI van 29.5 kg/m²"
    bmi_match = re.search(r'bmi\s*(?:van\s*)?[:\s]*(\d{1,2}[,.]?\d*)\s*(kg/m[²2])?', note_lower)
    if bmi_match:
        result["bmi"]["value"] = float(bmi_match.group(1).replace(',', '.'))
        result["bmi"]["unit"] = bmi_match.group(2) if bmi_match.group(2) else "kg/m²"
        add_evidence("bmi", result["bmi"]["value"], bmi_match.span())
    
    # Smoking status
    smoking_match_never = re.search(r'(niet-roker|nooit gerookt|rookt niet)', note_lower)
    smoking_match_current = re.search(r'(rookt momenteel|\broker\b|rookt)', note_lower)
    smoking_match_former = re.search(r'(gestopt met roken|ex-roker|voorheen roker)', note_lower)
    if smoking_match_never:
        result["smoking_status"] = "never"
        add_evidence("smoking_status", "never", smoking_match_never.span())
    elif smoking_match_current:
        result["smoking_status"] = "current"
        add_evidence("smoking_status", "current", smoking_match_current.span())
    elif smoking_match_former:
        result["smoking_status"] = "former"
        add_evidence("smoking_status", "former", smoking_match_former.span())
    
    # Blood pressure - patterns like "bloeddruk 133/92" or "(133/92 mmHg)"
    bp_match = re.search(r'(?:bloeddruk|tensie)[^0-9]*(\d{2,3})\s*/\s*(\d{2,3})\s*(mmHg)?', note_lower)
    if bp_match:
        result["vital_signs"]["blood_pressure"]["value"] = f"{bp_match.group(1)}/{bp_match.group(2)}"
        result["vital_signs"]["blood_pressure"]["unit"] = bp_match.group(3) if bp_match.group(3) else "mmHg"
        add_evidence("vital_signs.blood_pressure", result["vital_signs"]["blood_pressure"]["value"], bp_match.span())
    else:
        # Try standalone pattern
        bp_match2 = re.search(r'\((\d{2,3})/(\d{2,3})\s*(mmHg)\)', note)
        if bp_match2:
            result["vital_signs"]["blood_pressure"]["value"] = f"{bp_match2.group(1)}/{bp_match2.group(2)}"
            result["vital_signs"]["blood_pressure"]["unit"] = bp_match2.group(3)
            add_evidence("vital_signs.blood_pressure", result["vital_signs"]["blood_pressure"]["value"], bp_match2.span())
    
    # Heart rate - patterns like "hartfrequentie van 86" or "hartslag 86"
    hr_match = re.search(r'(?:hartfrequentie|hartslag|pols)[^0-9]*(\d{2,3})\s*(slagen|bpm|/min)?', note_lower)
    if hr_match:
        result["vital_signs"]["heart_rate"]["value"] = float(hr_match.group(1))
        result["vital_signs"]["heart_rate"]["unit"] = hr_match.group(2) if hr_match.group(2) else "bpm"
        add_evidence("vital_signs.heart_rate", result["vital_signs"]["heart_rate"]["value"], hr_match.span())
    
    # Oxygen saturation - patterns like "zuurstofsaturatie van 98%" or "saturatie 98%"
    spo2_match = re.search(r'(?:zuurstof)?saturatie[^0-9]*(\d{2,3})\s*(%)?', note_lower)
    if spo2_match:
        result["vital_signs"]["oxygen_saturation"]["value"] = float(spo2_match.group(1))
        result["vital_signs"]["oxygen_saturation"]["unit"] = "%"
        add_evidence("vital_signs.oxygen_saturation", result["vital_signs"]["oxygen_saturation"]["value"], spo2_match.span())
    
    # HbA1c - patterns like "HbA1c van 6,5%" or "HbA1c 6.5" or "HbA1c 48 mmol/mol"
    hba1c_match = re.search(r'hba1c[^0-9]*(\d{1,3}[,.]?\d*)\s*(%|mmol/mol)?', note_lower)
    if hba1c_match:
        result["recent_labs"]["hbA1c"]["value"] = float(hba1c_match.group(1).replace(',', '.'))
        result["recent_labs"]["hbA1c"]["unit"] = hba1c_match.group(2) if hba1c_match.group(2) else "%"
        add_evidence("recent_labs.hbA1c", result["recent_labs"]["hbA1c"]["value"], hba1c_match.span())
    
    # Creatinine - patterns like "creatinine 85" or "creatinine van 85 µmol/L"
    creat_match = re.search(r'creatinine[^0-9]*(\d{1,4}[,.]?\d*)\s*(µmol/L|umol/L|μmol/L|mg/dL)?', note_lower)
    if creat_match:
        result["recent_labs"]["creatinine"]["value"] = float(creat_match.group(1).replace(',', '.'))
        result["recent_labs"]["creatinine"]["unit"] = creat_match.group(2) if creat_match.group(2) else "mg/dL"
        add_evidence("recent_labs.creatinine", result["recent_labs"]["creatinine"]["value"], creat_match.span())
    
    # LDL cholesterol - can be in mmol/L or mg/dL
    ldl_match = re.search(r'ldl[^0-9]*(\d{1,3}[,.]?\d*)\s*(mmol/L|mg/dL)?', note_lower)
    if ldl_match:
        result["recent_labs"]["ldl_cholesterol"]["value"] = float(ldl_match.group(1).replace(',', '.'))
        result["recent_labs"]["ldl_cholesterol"]["unit"] = ldl_match.group(2) if ldl_match.group(2) else "mg/dL"
        add_evidence("recent_labs.ldl_cholesterol", result["recent_labs"]["ldl_cholesterol"]["value"], ldl_match.span())
    
    print("\n" + "="*60)
    print("[REGEX] EXTRACTED VALUES:")
    print("="*60)
    print(f"  Sex: {result['sex']}")
    print(f"  Age: {result['age']}")
    print(f"  Height: {result['height']['value']} {result['height']['unit']}")
    print(f"  Weight: {result['weight']['value']} {result['weight']['unit']}")
    print(f"  BMI: {result['bmi']['value']} {result['bmi']['unit']}")
    print(f"  Smoking: {result['smoking_status']}")
    print(f"  BP: {result['vital_signs']['blood_pressure']['value']} {result['vital_signs']['blood_pressure']['unit']}")
    print(f"  HR: {result['vital_signs']['heart_rate']['value']} {result['vital_signs']['heart_rate']['unit']}")
    print(f"  SpO2: {result['vital_signs']['oxygen_saturation']['value']} {result['vital_signs']['oxygen_saturation']['unit']}")
    print(f"  HbA1c: {result['recent_labs']['hbA1c']['value']} {result['recent_labs']['hbA1c']['unit']}")
    print(f"  LDL: {result['recent_labs']['ldl_cholesterol']['value']} {result['recent_labs']['ldl_cholesterol']['unit']}")
    print("="*60 + "\n")
    
    return result


def extract_with_gliner(note):
    """Extract structured data using GLiNER2 named entity recognition."""
    global gliner_model
    
    if gliner_model is None:
        load_gliner_model()
    
    print("\n" + "="*60)
    print("[GLiNER2] INPUT TEXT:")
    print("="*60)
    print(note[:500] + "..." if len(note) > 500 else note)
    print("\n[GLiNER2] LABELS:")
    print("="*60)
    print(GLINER_LABELS)
    print("="*60)
    
    # Run GLiNER extraction
    entities = gliner_model.predict_entities(note, GLINER_LABELS, threshold=0.3)
    
    print("\n" + "="*60)
    print("[GLiNER2] EXTRACTED ENTITIES:")
    print("="*60)
    for ent in entities:
        print(f"  {ent['label']}: '{ent['text']}' (score: {ent['score']:.3f})")
    print("="*60 + "\n")
    
    # First, use regex-based extraction for reliable numeric values
    result = extract_with_regex(note)
    
    # Then use GLiNER primarily for conditions (which regex can't easily detect)
    conditions_found = set()
    
    for ent in entities:
        label = ent["label"].lower()
        text = ent["text"]
        
        # Conditions - this is where GLiNER adds value
        if label in ["diagnose", "aandoening", "ziekte"]:
            # Filter out common false positives
            text_lower = text.lower()
            skip_words = ["ontslag", "opname", "patiënt", "patient", "advies", "controle", 
                          "betreft", "geachte", "collega", "brief", "ziekenhuis"]
            if not any(skip in text_lower for skip in skip_words) and len(text) > 3:
                if text not in conditions_found:
                    conditions_found.add(text)
                    result["conditions"].append({"name": text})
                    # Add evidence using entity offsets if available
                    start = ent.get("start")
                    end = ent.get("end")
                    if start is not None and end is not None:
                        result.setdefault("evidence", []).append({
                            "field": "conditions",
                            "value": text,
                            "start": int(start),
                            "end": int(end),
                            "text": note[int(start):int(end)],
                            "source": "gliner"
                        })
    
    return result


def extract_with_openai(note):
    """Extract structured data using OpenAI GPT-5-mini."""
    from openai import OpenAI
    
    client = OpenAI()  # Uses OPENAI_API_KEY env variable
    
    user_content = f"NOTE:\n{note}"
    
    print("\n" + "="*60)
    print("[OPENAI GPT-5-mini] SYSTEM PROMPT:")
    print("="*60)
    print(SYSTEM_PROMPT)
    print("\n[OPENAI GPT-5-mini] USER MESSAGE:")
    print("="*60)
    print(user_content[:500] + "..." if len(user_content) > 500 else user_content)
    print("="*60)
    
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        response_format={"type": "json_object"},
        reasoning_effort="none"  # Disable reasoning for faster responses
    )
    
    output = response.choices[0].message.content
    
    print("\n" + "="*60)
    print("[OPENAI GPT-5-mini] OUTPUT:")
    print("="*60)
    print(output)
    print("="*60 + "\n")
    
    parsed = safe_json(output)
    if parsed is None:
        return None

    # Normalize evidence structure for frontend highlighting
    # Match quotes against the original note text to find positions
    ev_list = parsed.get("evidence", []) if isinstance(parsed, dict) else []
    normalized = []
    for ev in ev_list:
        try:
            quote = ev.get("quote", "").strip()
            if not quote:
                continue
            # Find the quote in the original note (case-insensitive search, then use actual position)
            note_lower = note.lower()
            quote_lower = quote.lower()
            idx = note_lower.find(quote_lower)
            if idx >= 0:
                # Use the actual text from the note (preserves original casing)
                actual_text = note[idx:idx + len(quote)]
                normalized.append({
                    "field": ev.get("field"),
                    "value": ev.get("value"),
                    "text": actual_text,
                    "start": idx,
                    "end": idx + len(quote),
                    "source": "openai"
                })
            else:
                print(f"[WARN] Could not find quote in note: {quote[:50]}...")
        except Exception as e:
            print(f"[WARN] Error processing evidence: {e}")
            continue
    parsed["evidence"] = normalized
    return parsed


def map_condition_to_snomed(condition_name, top_k=3):
    """Map a condition name by first matching ICD10, then mapping to SNOMED."""
    global embedder, snomed_embeddings, icd10_embeddings, code_ref_unique, code_ref_icd10_unique

    if embedder is None:
        load_embedder()

    from sentence_transformers import util

    if not condition_name or not condition_name.strip():
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1: match to ICD10 descriptions
    query_emb = embedder.encode(condition_name, convert_to_tensor=True, device=device)
    icd_scores = util.cos_sim(query_emb, icd10_embeddings)[0]
    icd_top = torch.topk(icd_scores, k=top_k)

    icd_matches = []
    for score, idx in zip(icd_top.values.tolist(), icd_top.indices.tolist()):
        row = code_ref_icd10_unique.iloc[idx]
        icd_matches.append({
            "icd10_code": str(row.get("icd10_code", "")),
            "icd10_description": row["icd10_description"],
            "similarity": round(float(score), 3)
        })

    # Step 2: map ICD10 candidates to SNOMED using the reference table
    snomed_results = []
    for icd in icd_matches:
        icd_code = icd.get("icd10_code", "")
        mapped_rows = code_ref[code_ref["icd10_code"] == icd_code]
        if mapped_rows.empty:
            continue
        # pick the first SNOMED for that ICD10 code
        first_row = mapped_rows.iloc[0]
        snomed_results.append({
            "snomed_code": str(first_row.get("snomed_code", "")),
            "snomed_description": first_row.get("snomed_description", ""),
            "icd10_code": icd_code,
            "icd10_description": icd.get("icd10_description", ""),
            "similarity": icd.get("similarity", 0.0)
        })

    # If we found SNOMED via ICD10, return those; otherwise fallback to direct SNOMED match
    if snomed_results:
        return snomed_results

    # Fallback direct SNOMED match
    scores = util.cos_sim(query_emb, snomed_embeddings)[0]
    top = torch.topk(scores, k=top_k)
    results = []
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        row = code_ref_unique.iloc[idx]
        results.append({
            "snomed_code": str(row.get("snomed_code", "")),
            "snomed_description": row["snomed_description"],
            "icd10_code": str(row.get("icd10_code", "")),
            "icd10_description": str(row.get("icd10_description", "")),
            "similarity": round(float(score), 3)
        })
    return results


def convert_to_omop(extracted_data, patient_id, snomed_matches):
    """Convert extracted data to OMOP CDM format using Athena API for concept lookups."""
    
    # Look up gender concepts from Athena
    male_concept_id = get_omop_concept_id("Male", vocabulary_id="Gender", domain_id="Gender", fallback_id=8507)
    female_concept_id = get_omop_concept_id("Female", vocabulary_id="Gender", domain_id="Gender", fallback_id=8532)
    
    # PERSON table
    person = {
        "person_id": patient_id,
        "gender_concept_id": male_concept_id if extracted_data.get("sex") == "male" else female_concept_id if extracted_data.get("sex") == "female" else 0,
        "gender_source_value": extracted_data.get("sex"),
        "year_of_birth": 2026 - extracted_data.get("age") if extracted_data.get("age") else None,
    }
    
    # CONDITION_OCCURRENCE table
    conditions = []
    if snomed_matches and len(snomed_matches) > 0:
        best_match = snomed_matches[0]
        conditions.append({
            "condition_occurrence_id": 1,
            "person_id": patient_id,
            "condition_concept_id": best_match.get("snomed_code", ""),
            "condition_source_value": extracted_data.get("conditions", [{}])[0].get("name", "") if extracted_data.get("conditions") else "",
            "condition_source_concept_id": best_match.get("icd10_code", ""),
        })
    
    # MEASUREMENT table (labs and vitals)
    measurements = []
    measurement_id = 1
    
    # Helper to get value and unit from new structure
    def get_val_unit(obj, default_unit=None):
        if obj is None:
            return None, default_unit
        if isinstance(obj, dict):
            return obj.get("value"), obj.get("unit", default_unit)
        return obj, default_unit  # backward compatibility for plain values
    
    # Labs
    labs = extracted_data.get("recent_labs", {}) or {}
    hba1c_val, hba1c_unit = get_val_unit(labs.get("hbA1c"), "%")
    if hba1c_val is not None:
        hba1c_concept = get_omop_concept_id("Hemoglobin A1c", vocabulary_id="LOINC", domain_id="Measurement", fallback_id=3004410)
        measurements.append({
            "measurement_id": measurement_id,
            "person_id": patient_id,
            "measurement_concept_id": hba1c_concept,
            "measurement_source_value": "hbA1c",
            "value_as_number": hba1c_val,
            "unit_source_value": hba1c_unit
        })
        measurement_id += 1
    
    creat_val, creat_unit = get_val_unit(labs.get("creatinine"), "mg/dL")
    if creat_val is not None:
        creat_concept = get_omop_concept_id("Creatinine serum", vocabulary_id="LOINC", domain_id="Measurement", fallback_id=3016723)
        measurements.append({
            "measurement_id": measurement_id,
            "person_id": patient_id,
            "measurement_concept_id": creat_concept,
            "measurement_source_value": "creatinine",
            "value_as_number": creat_val,
            "unit_source_value": creat_unit
        })
        measurement_id += 1
    
    ldl_val, ldl_unit = get_val_unit(labs.get("ldl_cholesterol"), "mg/dL")
    if ldl_val is not None:
        ldl_concept = get_omop_concept_id("LDL Cholesterol", vocabulary_id="LOINC", domain_id="Measurement", fallback_id=3028437)
        measurements.append({
            "measurement_id": measurement_id,
            "person_id": patient_id,
            "measurement_concept_id": ldl_concept,
            "measurement_source_value": "ldl_cholesterol",
            "value_as_number": ldl_val,
            "unit_source_value": ldl_unit
        })
        measurement_id += 1
    
    # Vitals
    vitals = extracted_data.get("vital_signs", {}) or {}
    hr_val, hr_unit = get_val_unit(vitals.get("heart_rate"), "bpm")
    if hr_val is not None:
        hr_concept = get_omop_concept_id("Heart rate", vocabulary_id="LOINC", domain_id="Measurement", fallback_id=3027018)
        measurements.append({
            "measurement_id": measurement_id,
            "person_id": patient_id,
            "measurement_concept_id": hr_concept,
            "measurement_source_value": "heart_rate",
            "value_as_number": hr_val,
            "unit_source_value": hr_unit
        })
        measurement_id += 1
    
    spo2_val, spo2_unit = get_val_unit(vitals.get("oxygen_saturation"), "%")
    if spo2_val is not None:
        spo2_concept = get_omop_concept_id("Oxygen saturation", vocabulary_id="LOINC", domain_id="Measurement", fallback_id=3016502)
        measurements.append({
            "measurement_id": measurement_id,
            "person_id": patient_id,
            "measurement_concept_id": spo2_concept,
            "measurement_source_value": "oxygen_saturation",
            "value_as_number": spo2_val,
            "unit_source_value": spo2_unit
        })
        measurement_id += 1
    
    # OBSERVATION table (BMI, smoking, height, weight)
    observations = []
    observation_id = 1
    
    height_val, height_unit = get_val_unit(extracted_data.get("height"), "cm")
    if height_val is not None:
        height_concept = get_omop_concept_id("Body height", vocabulary_id="LOINC", domain_id="Measurement", fallback_id=3036277)
        observations.append({
            "observation_id": observation_id,
            "person_id": patient_id,
            "observation_concept_id": height_concept,
            "observation_source_value": "height",
            "value_as_number": height_val,
            "unit_source_value": height_unit
        })
        observation_id += 1
    
    weight_val, weight_unit = get_val_unit(extracted_data.get("weight"), "kg")
    if weight_val is not None:
        weight_concept = get_omop_concept_id("Body weight", vocabulary_id="LOINC", domain_id="Measurement", fallback_id=3025315)
        observations.append({
            "observation_id": observation_id,
            "person_id": patient_id,
            "observation_concept_id": weight_concept,
            "observation_source_value": "weight",
            "value_as_number": weight_val,
            "unit_source_value": weight_unit
        })
        observation_id += 1
    
    bmi_val, bmi_unit = get_val_unit(extracted_data.get("bmi"), "kg/m²")
    if bmi_val is not None:
        bmi_concept = get_omop_concept_id("Body mass index", vocabulary_id="LOINC", domain_id="Measurement", fallback_id=3038553)
        observations.append({
            "observation_id": observation_id,
            "person_id": patient_id,
            "observation_concept_id": bmi_concept,
            "observation_source_value": "bmi",
            "value_as_number": bmi_val,
            "unit_source_value": bmi_unit
        })
        observation_id += 1
    
    # Smoking status - look up concept IDs from Athena
    smoking_status = extracted_data.get("smoking_status")
    if smoking_status:
        if smoking_status == "never":
            smoking_concept = get_omop_concept_id("Never smoker", vocabulary_id="SNOMED", domain_id="Observation", fallback_id=4144272)
        elif smoking_status == "former":
            smoking_concept = get_omop_concept_id("Ex-smoker", vocabulary_id="SNOMED", domain_id="Observation", fallback_id=4310250)
        elif smoking_status == "current":
            smoking_concept = get_omop_concept_id("Current smoker", vocabulary_id="SNOMED", domain_id="Observation", fallback_id=4298794)
        else:
            smoking_concept = 0
        
        if smoking_concept:
            observations.append({
                "observation_id": observation_id,
                "person_id": patient_id,
                "observation_concept_id": smoking_concept,
                "observation_source_value": f"smoking_status: {smoking_status}",
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
# Flask routes
# ============================================================================

@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/api/patients")
def get_patients():
    """Get list of patients from validation set."""
    global val_df
    
    patients = []
    for idx, row in val_df.iterrows():
        patients.append({
            "patient_id": int(row["patient_id"]),
            "label": f"Patient {row['patient_id']} - {row.get('sex', 'unknown')}, age {row.get('age', '?')}",
            "discharge_note": row["discharge_note"]
        })
    
    return jsonify(patients)


@app.route("/api/extract", methods=["POST"])
def extract():
    """Extract structured data from discharge note using LLM."""
    data = request.json
    note = data.get("discharge_note", "")
    use_local = data.get("use_local", True)
    
    try:
        if use_local:
            extracted = extract_with_gliner(note)
        else:
            extracted = extract_with_openai(note)
        
        if extracted is None:
            return jsonify({"error": "Failed to parse LLM output"}), 500

        # Ensure evidence key exists for frontend highlighting
        extracted.setdefault("evidence", [])
        
        # Map conditions to SNOMED/ICD10 (use first mention for clarity)
        snomed_matches = []
        if extracted.get("conditions") and len(extracted["conditions"]) > 0:
            primary_condition = extracted["conditions"][0].get("name", "")
            snomed_matches = map_condition_to_snomed(primary_condition)
        
        return jsonify({
            "extracted": extracted,
            "snomed_matches": snomed_matches
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/convert", methods=["POST"])
def convert():
    """Convert extracted data to OMOP CDM format."""
    data = request.json
    extracted = data.get("extracted", {})
    patient_id = data.get("patient_id", 0)
    snomed_matches = data.get("snomed_matches", [])
    
    try:
        omop = convert_to_omop(extracted, patient_id, snomed_matches)
        return jsonify(omop)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model", methods=["POST"])
def set_model():
    """Toggle between local and remote model."""
    global USE_LOCAL_MODEL
    data = request.json
    USE_LOCAL_MODEL = data.get("use_local", True)
    return jsonify({"use_local": USE_LOCAL_MODEL})


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    load_data()
    load_embedder()  # Pre-load embeddings at startup (uses cache if available)
    print("\n" + "="*50)
    print("EHR to OMOP Demo Server")
    print("="*50)
    print("Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
