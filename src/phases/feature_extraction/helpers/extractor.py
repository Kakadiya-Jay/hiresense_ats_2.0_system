# src/phases/feature_extraction/helpers/extractor.py
"""
Rule-based Feature Extraction helper (Day 3 baseline).
This module expects sectioned resume data (raw_sections dict) produced by
the sectioning phase (src/phases/sectioning). It uses three config JSONs:
 - config/hiresense_skill_dictionary_v2.json
 - config/important_technical_words.json
 - config/project_triggers.json

Main public function:
    extract_features_from_sections(raw_sections: dict, section_confidence: dict, link_metadata: list, emails: list, phones: list) -> dict

Produces a candidate JSON skeleton:
{
  "candidate_id": "...",
  "created_at": "...Z",
  "personal_info": {...},
  "skills": [...],
  "projects": [...],
  "education": [...],
  "experience": [...],
  "certifications": [...],
  "achievements": [...],
  "languages": [...],
  "raw_sections": {...},
  "section_confidence": {...},
  "raw_links": {...},
  "all_emails": [...],
  "all_phones": [...]
}
"""

from pathlib import Path
import json
import re
from datetime import datetime
import phonenumbers
from typing import Dict, List, Any, Optional

ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = ROOT / "config"


def load_json_safe(path: Path) -> Any:
    """
    Load JSON from path robustly.
    - Try utf-8
    - Fallback to latin-1
    - If JSON parsing fails, return {} (caller should handle missing config)
    """
    if not path.exists():
        return {}
    # try utf-8 first
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text)
    except UnicodeDecodeError:
        # fallback to latin-1 (single-byte) to avoid decode error
        try:
            text = path.read_text(encoding="latin-1")
            return json.loads(text)
        except Exception:
            return {}
    except json.JSONDecodeError:
        # file decoded but not valid JSON
        return {}
    except Exception:
        # any other IO errors: return empty dict
        return {}


# Load config files
SKILLS_DICT = load_json_safe(CONFIG_DIR / "hiresense_skill_dictionary_v2.json")
IMPORTANT_WORDS = load_json_safe(CONFIG_DIR / "important_technical_words.json")
PROJECT_TRIGGERS = load_json_safe(CONFIG_DIR / "project_triggers.json")

# Basic regexes
YEAR_RE = re.compile(r"(19|20)\d{2}")
URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"[a-zA-Z0-9.\-_+]+@[a-zA-Z0-9\-_]+\.[a-zA-Z0-9.\-_]+")


def extract_phone_numbers(text: str, region: str = "IN") -> List[str]:
    """
    Extract phone numbers using the phonenumbers library.
    Default region is India ("IN"), but can be changed.
    Returns a list of unique, international-format numbers.
    """
    phones = []
    for match in phonenumbers.PhoneNumberMatcher(text, region):
        formatted = phonenumbers.format_number(
            match.number, phonenumbers.PhoneNumberFormat.INTERNATIONAL
        )
        phones.append(formatted)
    return list(set(phones))


# Section key aliases: maps probable sectioner keys to normalized names
SECTION_KEY_ALIASES = {
    "experience": [
        "experience",
        "work_experience",
        "professional_experience",
        "employment",
    ],
    "education": ["education", "academic", "academics"],
    "skills": ["skills", "technical_skills", "skillset"],
    "projects": ["projects", "personal_projects", "project"],
    "certifications": ["certifications", "certificates"],
    "achievements": ["achievements", "awards"],
    "summary": ["summary", "objective", "profile"],
    "languages": ["languages", "language"],
}


def normalize_section_key(key: str) -> str:
    k = key.lower().strip()
    for norm, cand_list in SECTION_KEY_ALIASES.items():
        if k in cand_list:
            return norm
    # fallback: return simplified key (letters + underscore)
    return re.sub(r"[^a-z0-9_]", "_", k)


def gather_section_text(raw_sections: Dict[str, str]) -> Dict[str, str]:
    """
    Normalize raw_sections keys to known canonical section names.
    """
    normalized = {}
    for k, v in (raw_sections or {}).items():
        nk = normalize_section_key(k)
        if nk in normalized:
            normalized[nk] += "\n\n" + v
        else:
            normalized[nk] = v or ""
    return normalized


# Flatten SKILLS_DICT content to list of skill terms (strings)
def flatten_skill_terms(skills_dict: Dict) -> List[str]:
    terms = set()
    if not isinstance(skills_dict, dict):
        return []

    def recurse(x):
        if isinstance(x, str):
            if x.strip():
                terms.add(x.strip().lower())
        elif isinstance(x, list):
            for it in x:
                recurse(it)
        elif isinstance(x, dict):
            for k, v in x.items():
                recurse(k)
                recurse(v)

    recurse(skills_dict)
    return sorted(terms)


_SKILL_TERMS = flatten_skill_terms(SKILLS_DICT)


def extract_skills_from_sections(
    sections: Dict[str, str], top_k: int = 80
) -> List[str]:
    """
    Rule: check skills section first; then scan entire text for known skill terms.
    Returns unique, lower-cased canonical skill strings.
    """
    found = []
    seen = set()
    # 1) explicit skills section
    skills_block = sections.get("skills", "")
    if skills_block:
        # split on commas / newlines / semicolons
        tokens = re.split(r"[,\n;•\u2022\-\–]", skills_block)
        for t in tokens:
            t = t.strip()
            if not t:
                continue
            # normalize: lower-case, remove extra chars
            norm = re.sub(r"[^a-z0-9\+\#\.\s\-]", "", t.lower()).strip()
            if norm and norm not in seen:
                found.append(norm)
                seen.add(norm)
            if len(found) >= top_k:
                break
    # 2) fallback: scan whole resume for skill keywords
    if len(found) < top_k:
        full_text = "\n".join(sections.values()).lower()
        for term in _SKILL_TERMS:
            if term in full_text and term not in seen:
                seen.add(term)
                found.append(term)
            if len(found) >= top_k:
                break
    return found


def extract_projects_from_sections(sections: Dict[str, str]) -> List[Dict[str, Any]]:
    projects = []
    proj_block = sections.get("projects", "")
    if not proj_block:
        # sometimes projects appear inside experience — try to capture short bullets mentioning 'project'
        exp_block = sections.get("experience", "")
        # naive search for lines containing 'project'
        candidates = [ln for ln in exp_block.splitlines() if "project" in ln.lower()]
        if candidates:
            for i, ln in enumerate(candidates):
                projects.append(
                    {
                        "title": ln.strip()[:120],
                        "tech_stack": [],
                        "summary": ln.strip(),
                        "repo_link": "",
                        "other_links": [],
                    }
                )
            return projects
        return projects

    # split projects by double newline or bullets
    entries = re.split(r"\n{2,}|\n\s*[-•\u2022]\s+", proj_block)
    for ent in entries:
        ent = ent.strip()
        if not ent:
            continue
        # find urls
        urls = URL_RE.findall(ent)
        repo = ""
        other_links = []
        for u in urls:
            if "github" in u.lower() or "gitlab" in u.lower():
                repo = u
            else:
                other_links.append(u)
        # tech extraction: look for bracketed tech lists or 'Tech:'
        tech = []
        bracket_match = re.search(r"\[(.*?)\]", ent)
        if bracket_match:
            tech = [
                t.strip()
                for t in re.split(r"[,\|/]", bracket_match.group(1))
                if t.strip()
            ]
        else:
            # look for "Tech:" or "Technology:"
            m = re.search(r"(tech(nology)?[:\-]\s*)(.+)", ent, flags=re.IGNORECASE)
            if m:
                tech = [t.strip() for t in re.split(r"[,\|/]", m.group(3)) if t.strip()]
        # title heuristic: first line up to hyphen or ':' or newline
        lines = [l.strip() for l in ent.splitlines() if l.strip()]
        title = lines[0] if lines else ent[:80]
        # attempt to strip tech from title
        title = re.sub(r"\[.*?\]", "", title).strip()
        summary = " ".join(lines[1:]) if len(lines) > 1 else ent
        projects.append(
            {
                "title": title,
                "tech_stack": tech,
                "summary": summary,
                "repo_link": repo,
                "other_links": other_links,
                "dates": {},
            }
        )
    return projects


def extract_education_from_sections(sections: Dict[str, str]) -> List[Dict[str, Any]]:
    edus = []
    edu_block = sections.get("education", "")
    if not edu_block:
        # try to find lines containing 'university' 'college' or degrees in all sections
        all_text = "\n".join(sections.values())
        candidate_lines = [
            ln
            for ln in all_text.splitlines()
            if any(
                w in ln.lower()
                for w in [
                    "university",
                    "college",
                    "bachelor",
                    "master",
                    "b.sc",
                    "m.sc",
                    "b.tech",
                    "bca",
                    "bcom",
                    "bba",
                ]
            )
        ]
        for ln in candidate_lines:
            year = YEAR_RE.search(ln)
            edus.append(
                {
                    "degree": ln.strip(),
                    "institute": "",
                    "year": year.group(0) if year else "",
                    "raw": ln.strip(),
                }
            )
        return edus

    parts = re.split(r"\n{1,}|\r\n", edu_block)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        year = YEAR_RE.search(p)
        degree = ""
        inst = ""
        # heuristics: split by ',' or '-' or '|' to obtain tokens
        tokens = re.split(r",|\||\-", p)
        if tokens:
            # look for degree-like token
            for t in tokens:
                lt = t.lower()
                if any(
                    k in lt
                    for k in [
                        "bachelor",
                        "master",
                        "b.sc",
                        "m.sc",
                        "b.tech",
                        "m.tech",
                        "bca",
                        "mba",
                        "phd",
                    ]
                ):
                    degree = t.strip()
                else:
                    # likely institute if it contains 'university' or 'college'
                    if "university" in lt or "college" in lt or "institute" in lt:
                        inst = t.strip()
            if not degree:
                degree = tokens[0].strip()
            if not inst and len(tokens) > 1:
                inst = tokens[1].strip()
        edus.append(
            {
                "degree": degree,
                "institute": inst,
                "year": year.group(0) if year else "",
                "raw": p,
            }
        )
    return edus


def extract_certifications_from_sections(
    sections: Dict[str, str],
) -> List[Dict[str, Any]]:
    certs = []
    cert_block = sections.get("certifications", "")
    if not cert_block:
        return certs
    lines = [l.strip() for l in cert_block.splitlines() if l.strip()]
    for l in lines:
        parts = re.split(r"\||\-|:|,", l)
        title = parts[0].strip()
        issuer = parts[1].strip() if len(parts) > 1 else ""
        certs.append(
            {
                "certificate_title": title,
                "issuing_organization": issuer,
                "issue_date": "",
                "expiration_date": "",
                "credential_id": "",
                "credential_url": "",
                "skills": [],
                "other": "",
                "raw": l,
            }
        )
    return certs


def extract_experience_from_sections(sections: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Very lightweight experience extraction: split experience block by double-newline and
    try to extract company, role, dates. This can be improved later with NER.
    """
    exps = []
    exp_block = sections.get("experience", "")
    if not exp_block:
        return exps
    entries = re.split(r"\n{2,}", exp_block)
    for ent in entries:
        ent = ent.strip()
        if not ent:
            continue
        lines = [l.strip() for l in ent.splitlines() if l.strip()]
        role = ""
        company = ""
        dates = {}
        # heuristics: first line may contain "Role - Company | dates"
        if lines:
            first = lines[0]
            # try splits
            parts = re.split(r"\||\-|\@", first)
            if len(parts) >= 2:
                role = parts[0].strip()
                company = parts[1].strip()
            else:
                # try to find year tokens to capture dates
                y = YEAR_RE.search(first)
                if y:
                    company = first
                else:
                    role = first
            # find date tokens anywhere in ent
            ymatch = YEAR_RE.findall(ent)
            if ymatch:
                # naive: take first and last occurrence
                dates = (
                    {"start": ymatch[0], "end": ymatch[-1]}
                    if len(ymatch) > 1
                    else {"start": ymatch[0], "end": ""}
                )
        summary = " ".join(lines[1:]) if len(lines) > 1 else ""
        exps.append(
            {
                "company_name": company,
                "role": role,
                "dates": dates,
                "project_details": {"summary": summary},
                "skills": [],
                "team_size": 0,
                "other": ent,
            }
        )
    return exps


def extract_achievements_from_sections(
    sections: Dict[str, str],
) -> List[Dict[str, Any]]:
    ach = []
    block = sections.get("achievements", "")
    if not block:
        return ach
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    for l in lines:
        ach.append({"title": l, "raw": l})
    return ach


def extract_languages_from_sections(sections: Dict[str, str]) -> List[str]:
    langs = []
    block = sections.get("languages", "")
    if block:
        tokens = re.split(r"[,\n;]", block)
        for t in tokens:
            t = t.strip()
            if t:
                langs.append(t)
    else:
        # attempt to find common languages in whole text
        text = "\n".join(sections.values()).lower()
        for maybe in ["english", "hindi", "gujarati", "marathi"]:
            if maybe in text and maybe not in langs:
                langs.append(maybe.capitalize())
    return langs


def normalize_links(link_metadata: Optional[List[Dict]]) -> Dict[str, Any]:
    out = {"links": []}
    if not link_metadata:
        return out
    for m in link_metadata:
        out["links"].append(m)
    return out


# public function
def extract_features_from_sections(
    raw_sections: Dict[str, str],
    section_confidence: Optional[Dict[str, float]] = None,
    link_metadata: Optional[List[Dict[str, Any]]] = None,
    emails: Optional[List[str]] = None,
    phones: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Main entrypoint used by pipeline / API service.
    raw_sections: dict produced by sectioner (keys -> text blocks)
    section_confidence: optional confidences for those blocks
    link_metadata: list of link metadata dicts
    emails/phones: lists of strings
    """
    raw_sections = raw_sections or {}
    section_confidence = section_confidence or {}
    link_metadata = link_metadata or []
    emails = emails or []
    phones = phones or []

    # Normalize and canonicalize keys
    sections = gather_section_text(raw_sections)

    # Extract features using rule-based heuristics
    skills = extract_skills_from_sections(sections)
    projects = extract_projects_from_sections(sections)
    education = extract_education_from_sections(sections)
    certifications = extract_certifications_from_sections(sections)
    experience = extract_experience_from_sections(sections)
    achievements = extract_achievements_from_sections(sections)
    languages = extract_languages_from_sections(sections)

    # Simple name resolution: look into 'summary' then 'preamble' then first line of 'raw_sections' if present
    name = ""
    if sections.get("summary"):
        # look for uppercase-phrases likely candidate name (first 5 words)
        first_line = sections["summary"].splitlines()[0].strip()
        name = re.sub(r"[^A-Za-z\s\-\.'']", "", first_line)[:80].strip()
    if not name:
        # use any preamble-like text
        preamble = sections.get("preamble", "")
        if preamble:
            first_line = preamble.splitlines()[0].strip()
            name = re.sub(r"[^A-Za-z\s\-\.'']", "", first_line)[:80].strip()

    candidate = {
        "candidate_id": f"cand_{int(datetime.now().timestamp())}",
        "created_at": datetime.now().isoformat() + "Z",
        "personal_info": {
            "name": name or "",
            "email": emails[0] if emails else "",
            "phone": phones[0] if phones else "",
            "summary": (sections.get("summary") or "")[:800],
        },
        "skills": skills,
        "projects": projects,
        "education": education,
        "experience": experience,
        "certifications": certifications,
        "achievements": achievements,
        "languages": languages,
        "raw_sections": sections,
        "section_confidence": section_confidence,
        "raw_links": normalize_links(link_metadata),
        "all_emails": emails,
        "all_phones": phones,
    }
    return candidate
