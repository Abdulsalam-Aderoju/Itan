"""Farmer Language Normalisation Layer for Ìtàn (Blueprint Pillar 3).

Maps local African agricultural vocabulary (Hausa, Yorùbá, Nigerian Pidgin, Igbo)
to standard English corpus terms before retrieval/SQL execution, allowing farmers
to query in local terminology while retrieving verified literature. Also handles
diacritic folding and basic language detection.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Curated agricultural lexicon (Blueprint Section 6)
LEXICON: Dict[str, str] = {
    # Hausa (HA)
    "masara": "maize",
    "gero": "pearl millet",
    "dawa": "sorghum",
    "shinkafa": "rice",
    "rogo": "cassava",
    "gudaji": "cowpea",
    "wake": "cowpea",
    "gyada": "groundnut",
    "yakuwa": "sorrel",
    "kwari": "insect pest",
    "cuta": "disease",
    "takin zamani": "inorganic fertiliser",
    "taki": "fertiliser",
    "maganin kwari": "pesticide",
    "feshin magani": "spray dilution",

    # Yorùbá (YO)
    "oka": "maize",
    "agbado": "maize",
    "ewa": "cowpea",
    "iwi": "cowpea",
    "gbaguda": "cassava",
    "iṣu": "yam",
    "isu": "yam",
    "iresi": "rice",
    "baba": "sorghum",
    "epà": "groundnut",
    "epa": "groundnut",
    "kókó": "cocoa",
    "koko": "cocoa",
    "kokoro": "insect pest",
    "arun": "disease",
    "ile-ira": "fertiliser",
    "ogbin": "crop",

    # Nigerian Pidgin (PCM)
    "corn": "maize",
    "beans": "cowpea",
    "cassava": "cassava",
    "yam": "yam",
    "rice": "rice",
    "groundnut": "groundnut",
    "tomati": "tomato",
    "pepper": "pepper",
    "agric": "agricultural",
    "chemical": "agrochemical",
    "pesticide": "agrochemical",
    "fertilizer": "fertiliser",
    "medicine": "pesticide",
    "bug": "pest",

    # Igbo (IG)
    "oka": "maize",
    "akpu": "cassava",
    "ji": "yam",
    "osikapa": "rice",
    "anara": "garden egg",
    "egusi": "melon",
    "ogwu": "pesticide",
    "nri ala": "fertiliser",
}

# Keywords for lightweight language detection
LANG_KEYWORDS = {
    "ha": {"masara", "gero", "dawa", "shinkafa", "rogo", "wake", "gyada", "kwari", "cuta", "taki", "magani"},
    "yo": {"agbado", "ewa", "gbaguda", "isu", "iresi", "baba", "epa", "kokoro", "arun", "ogbin"},
    "pcm": {"dey", "wey", "fit", "make", "watin", "wetin", "una", "am", "so", "pata", "finish"},
    "ig": {"akpu", "osikapa", "ogwu", "kwanu", "di", "mya"},
}

def remove_diacritics(text: str) -> str:
    """Normalize diacritics and accents (e.g. ọkà -> oka, ẹ̀wà -> ewa)."""
    nfd_form = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd_form if unicodedata.category(c) != "Mn")

def detect_language(text: str) -> str:
    """Detect language register: 'ha', 'yo', 'pcm', 'ig', or 'en' (default)."""
    clean_text = remove_diacritics(text.lower())
    words = set(re.findall(r"\w+", clean_text))

    for lang, keywords in LANG_KEYWORDS.items():
        if words.intersection(keywords):
            return lang
    return "en"

def normalise_query(text: str) -> Tuple[str, str]:
    """Normalise local agricultural terms to standard English.
    Returns (normalised_text, detected_language).
    """
    lang = detect_language(text)
    clean_text = remove_diacritics(text.lower())
    
    # Token replacement against curated lexicon
    tokens = re.findall(r"\b\w+\b", clean_text)
    replaced_tokens = []
    
    for token in tokens:
        if token in LEXICON:
            replaced_tokens.append(LEXICON[token])
        else:
            replaced_tokens.append(token)

    normalised_str = " ".join(replaced_tokens)
    return normalised_str, lang
