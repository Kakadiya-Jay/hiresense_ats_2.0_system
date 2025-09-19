# src/phases/feature_extraction/helpers/enhanced_extractor.py
"""
Enhanced Feature Extraction (Phase 3).

Consumes:
    raw_sections: Dict[str, str]        -- sectioned resume text (from sectioner)
    section_confidence: Dict[str,float] -- confidences per section
    link_metadata: List[Dict]           -- links found in text_extraction
    emails/phones: Lists                -- contact lists from text_extraction

Uses config dictionaries:
    config/hiresense_skill_dictionary_v2.json
    config/important_technical_words.json
    config/project_triggers.json

Produces a candidate dict matching the "Feature extraction output structure"
document. This is rule-based, conservative, and heavily commented so you can
iterate and improve parts independently.

Note: This module intentionally does not call text_extraction or sectioning;
it expects already-sectioned input.
"""

from pathlib import Path
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = ROOT / "config"


# -- Helpers to load config safely
def load_json_safe(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="latin-1"))
        except Exception:
            return {}


SKILLS_DICT = load_json_safe(CONFIG_DIR / "hiresense_skill_dictionary_v2.json")
IMPORTANT_WORDS = load_json_safe(CONFIG_DIR / "important_technical_words.json")
PROJECT_TRIGGERS = load_json_safe(CONFIG_DIR / "project_triggers.json")


# Flatten skills into canonical map: alias -> canonical
def build_skill_alias_map(skills_dict: Dict) -> Dict[str, str]:
    alias_map = {}
    if not isinstance(skills_dict, dict):
        return alias_map

    # traverse: top-level keys are categories -> values are lists/strings/dicts
    def recurse(obj, canonical=None):
        if isinstance(obj, str):
            key = obj.strip().lower()
            alias_map[key] = canonical or key
        elif isinstance(obj, list):
            for it in obj:
                recurse(it, canonical)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                # k might be canonical or alias
                can = k.strip().lower()
                recurse(v, canonical=can)
                # also add key itself
                alias_map[can] = can

    recurse(skills_dict)
    return alias_map


_SKILL_ALIAS_MAP = build_skill_alias_map(SKILLS_DICT)
_SKILL_TERMS = sorted(set([k for k in _SKILL_ALIAS_MAP.keys()]))

# Basic regexes
YEAR_RE = re.compile(r"(19|20)\d{2}")
URL_RE = re.compile(r"https?://[^\s\)\]\}>,;]+")


# Utilities
def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip())


def find_urls(text: str) -> List[str]:
    return URL_RE.findall(text or "")


# Name extraction (best-effort): look at 'personal_info' or preamble first lines
def extract_name_from_sections(sections: Dict[str, str]) -> str:
    # If personal_info section present, check for lines with name-like pattern
    for key in ("personal_info", "contact", "preamble", "summary"):
        block = sections.get(key, "")
        if not block:
            continue
        # heuristics: first non-empty line, remove email/phone parts
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        candidate = lines[0]
        # remove email / phone snippets
        candidate = re.sub(r"[^\w\s\.\-']", " ", candidate)
        # if candidate has >1 token and letters, use it
        if len(candidate.split()) >= 1 and re.search(r"[A-Za-z]{2,}", candidate):
            return " ".join(candidate.split()[:4])
    return ""


# Skill extraction and normalization
def extract_and_normalize_skills(
    sections: Dict[str, str], top_k: int = 200
) -> Dict[str, Any]:
    """
    Strategy:
      1. If 'skills' section exists, split by commas/newlines and map aliases -> canonical terms
      2. Fallback: scan all text for known skill terms from config
      3. Categorize skills (attempt to map to categories from SKILLS_DICT keys)
    Returns:
      {
        "raw": [...],            # raw tokens found
        "canonical": [...],      # canonical normalized tokens
        "by_category": {cat: [skills]}
      }
    """
    skills_found = []
    sect = sections.get("skills", "")
    if sect:
        tokens = re.split(r"[,\n;•\u2022\-\–/]", sect)
        for t in tokens:
            t = t.strip()
            if not t:
                continue
            skills_found.append(normalize_text(t).lower())

    # fallback scan
    if not skills_found:
        full = " ".join(sections.values()).lower()
        for term in _SKILL_TERMS:
            if term in full and term not in skills_found:
                skills_found.append(term)

    # normalization map
    canonical = []
    for raw in skills_found:
        # try exact key then substring mapping
        key = raw.lower()
        if key in _SKILL_ALIAS_MAP:
            can = _SKILL_ALIAS_MAP[key]
        else:
            # try substring match to find a longer alias
            can = None
            for alias in _SKILL_ALIAS_MAP:
                if alias in key and len(alias) >= 3:
                    can = _SKILL_ALIAS_MAP[alias]
                    break
            can = can or key
        if can not in canonical:
            canonical.append(can)

    # categorize: attempt by checking keys in SKILLS_DICT top-level
    by_category = {}
    if isinstance(SKILLS_DICT, dict):
        for cat, content in SKILLS_DICT.items():
            cat_key = cat.strip()
            by_category.setdefault(cat_key, [])
            # collect any canonical skills that appear in content
            content_text = json.dumps(content).lower() if content is not None else ""
            for can in canonical:
                if can in content_text or can in cat_key.lower():
                    if can not in by_category[cat_key]:
                        by_category[cat_key].append(can)
    # remove empty categories
    by_category = {k: v for k, v in by_category.items() if v}

    return {"raw": skills_found, "canonical": canonical, "by_category": by_category}


# Roles extraction (simple matching using IMPORTANT_WORDS or PROJECT_TRIGGERS roles)
def extract_roles(sections: Dict[str, str]) -> List[str]:
    roles = set()
    full = " ".join(sections.values()).lower()
    # IMPORTANT_WORDS may contain roles
    if isinstance(IMPORTANT_WORDS, dict):
        for k, v in IMPORTANT_WORDS.items():
            if isinstance(v, list):
                for term in v:
                    if term.lower() in full:
                        roles.add(k)
    # fallback: search for common role words
    common_roles = [
        "developer",
        "engineer",
        "data scientist",
        "data engineer",
        "analyst",
        "manager",
        "tester",
        "qa",
        "devops",
        "mobile",
        "android",
        "flutter",
    ]
    for r in common_roles:
        if r in full:
            roles.add(r)
    return sorted(list(roles))


# Projects extraction (attempt to produce STAR fields)
def extract_projects(
    sections: Dict[str, str], link_metadata: List[Dict]
) -> List[Dict[str, Any]]:
    projs = []
    proj_text = sections.get("projects", "")
    if not proj_text:
        # sometimes projects are under experience
        exp = sections.get("experience", "")
        proj_lines = [l for l in exp.splitlines() if "project" in l.lower() or "[" in l]
        proj_text = "\n\n".join(proj_lines) if proj_lines else ""

    if not proj_text:
        return projs

    # split into paragraphs
    entries = [p.strip() for p in re.split(r"\n{2,}", proj_text) if p.strip()]
    for ent in entries:
        title = ent.splitlines()[0].strip()
        # remove trailing tech bracket from title if present
        tech_stack = []
        m = re.search(r"\[(.*?)\]", ent)
        if m:
            tech_stack = [
                t.strip() for t in re.split(r"[,\|/]", m.group(1)) if t.strip()
            ]
            title = re.sub(r"\[.*?\]", "", title).strip()
        # summary: remaining lines
        summary = (
            "\n".join(ent.splitlines()[1:]).strip() if len(ent.splitlines()) > 1 else ""
        )
        # urls
        urls = find_urls(ent)
        repo = ""
        other_links = []
        for u in urls:
            if "github" in u.lower() or "gitlab" in u.lower():
                repo = u
            else:
                other_links.append(u)

        # attempt to extract small STAR-ish fields using heuristics (verbs + results)
        situation = ""
        task = ""
        action = ""
        result = ""
        # split sentences and look for keywords
        sents = re.split(r"[\.!\?]\s+", summary)
        for s in sents:
            sl = s.lower()
            if any(
                k in sl
                for k in ["develop", "build", "implemented", "created", "designed"]
            ):
                action = action + " " + s if action else s
            if any(
                k in sl
                for k in [
                    "result",
                    "reduced",
                    "improved",
                    "increase",
                    "decrease",
                    "success",
                    "achieved",
                    "launched",
                ]
            ):
                result = result + " " + s if result else s
            if any(k in sl for k in ["responsible", "task", "responsibilities"]):
                task = task + " " + s if task else s

        # compute simple feature_strength: presence of repo or tech increases strength
        strength = 0.2
        if tech_stack:
            strength += 0.3
        if repo:
            strength += 0.3
        if summary:
            strength += 0.2
        if strength > 1.0:
            strength = 1.0

        projs.append(
            {
                "title": normalize_text(title),
                "dates": {"start": "", "end": ""},
                "role": "",
                "tech_stack": tech_stack,
                "action": normalize_text(action),
                "outcome": normalize_text(result),
                "repo_link": repo,
                "live_link": "",
                "team_size": 0,
                "summary": normalize_text(summary),
                "other": "",
                "feature_strength": round(float(strength), 2),
                "raw": ent,
            }
        )
    return projs


# Education extraction
def extract_education(sections: Dict[str, str]) -> List[Dict[str, Any]]:
    eds = []
    block = sections.get("education", "")
    if not block:
        # try to find degree-like strings in whole text
        full = " ".join(sections.values())
        matches = re.findall(
            r"(Bachelor|B\.?Tech|B\.?Sc|BCA|Master|M\.?Tech|MBA|Ph\.?D)[^\n,\.]{0,80}",
            full,
            flags=re.I,
        )
        for m in matches:
            eds.append(
                {
                    "graduation": [
                        {"degree_name": m.strip(), "university_name": "", "year": ""}
                    ],
                    "raw": m,
                }
            )
        return eds

    parts = [p.strip() for p in re.split(r"\n{1,}|\r\n", block) if p.strip()]
    for p in parts:
        year_match = YEAR_RE.search(p)
        year = year_match.group(0) if year_match else ""
        # tokens split by '|' or '-'
        tokens = re.split(r"\||\-", p)
        degree = tokens[0].strip() if tokens else p
        institute = tokens[1].strip() if len(tokens) > 1 else ""
        eds.append(
            {
                "graduation": [
                    {
                        "degree_name": normalize_text(degree),
                        "university_name": normalize_text(institute),
                        "result": "",
                        "year": year,
                    }
                ],
                "raw": p,
            }
        )
    return eds


# Experience extraction (job level)
def extract_experience(sections: Dict[str, str]) -> List[Dict[str, Any]]:
    exps = []
    block = sections.get("experience", "")
    if not block:
        return exps
    entries = [p.strip() for p in re.split(r"\n{2,}", block) if p.strip()]
    for ent in entries:
        lines = [l.strip() for l in ent.splitlines() if l.strip()]
        if not lines:
            continue
        # heuristics: first line often contains role and company and dates
        header = lines[0]
        role = ""
        company = ""
        dates = {"start": "", "end": ""}
        # try split by ' - ' or ' | '
        if "|" in header:
            parts = [t.strip() for t in header.split("|")]
        elif "-" in header:
            parts = [t.strip() for t in header.split("-")]
        else:
            parts = [header]
        if len(parts) >= 2:
            role = parts[0]
            company = parts[1]
        else:
            # attempt to capture company via 'at' token
            if " at " in header.lower():
                role, company = header.split(" at ", 1)
            else:
                role = header

        # search dates in entry
        y = YEAR_RE.findall(ent)
        if y:
            if len(y) >= 2:
                dates = {"start": y[0], "end": y[-1]}
            else:
                dates = {"start": y[0], "end": ""}

        summary = " ".join(lines[1:]) if len(lines) > 1 else ""
        exps.append(
            {
                "company_name": normalize_text(company),
                "role": normalize_text(role),
                "years_of_experience": 0,
                "dates": dates,
                "project_details": {"summary": normalize_text(summary)},
                "skills": [],
                "team_size": 0,
                "other": ent,
                "feature_strength": 0.5,
            }
        )
    return exps


# Certifications/achievements
def extract_certifications(sections: Dict[str, str]) -> List[Dict[str, Any]]:
    certs = []
    block = sections.get("certifications", "") or sections.get("certificates", "")
    if not block:
        return certs
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    for l in lines:
        parts = re.split(r"\||\-|:|,", l)
        title = parts[0].strip()
        issuer = parts[1].strip() if len(parts) > 1 else ""
        certs.append(
            {
                "certificate_title": normalize_text(title),
                "issuing_organization": normalize_text(issuer),
                "issue_date": "",
                "expiration_date": "",
                "credential_id": "",
                "credential_url": "",
                "skills": [],
                "other": "",
                "raw": l,
                "feature_strength": 0.4,
            }
        )
    return certs


def extract_achievements(sections: Dict[str, str]) -> List[Dict[str, Any]]:
    ach = []
    block = sections.get("achievements", "") or sections.get("awards", "")
    if not block:
        return ach
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    for l in lines:
        ach.append({"title": l, "raw": l, "feature_strength": 0.3})
    return ach


def extract_open_source(
    sections: Dict[str, str], link_metadata: List[Dict]
) -> Dict[str, Any]:
    # find all repo links and label them as open_source if github/gitlab
    repos = []
    for lm in link_metadata or []:
        url = lm.get("normalized_url", "")
        if "github" in url.lower() or "gitlab" in url.lower():
            repos.append(
                {
                    "repo": url,
                    "summary": "",
                    "total_pr": 0,
                    "merged_pr": 0,
                    "feature_strength": 0.5,
                }
            )
    # also scan projects for repo links
    proj_block = sections.get("projects", "")
    for u in find_urls(proj_block):
        if "github" in u.lower() or "gitlab" in u.lower():
            if not any(r["repo"] == u for r in repos):
                repos.append(
                    {
                        "repo": u,
                        "summary": "",
                        "total_pr": 0,
                        "merged_pr": 0,
                        "feature_strength": 0.5,
                    }
                )
    return {"total_number_of_contributions": len(repos), "open_source": repos}


def extract_research_papers(sections: Dict[str, str]) -> List[Dict[str, Any]]:
    pubs = []
    # quick scan for 'arxiv' 'doi' 'paper' 'published' tokens
    full = " ".join(sections.values())
    candidates = []
    for line in full.splitlines():
        if any(
            k in line.lower()
            for k in ["doi", "arxiv", "paper", "published", "journal", "conference"]
        ):
            candidates.append(line.strip())
    for c in candidates:
        pubs.append(
            {
                "title": c[:200],
                "publication_name": "",
                "publication_link": "",
                "doi": "",
                "publication_type": [],
                "impact_factor": 0.0,
                "citations": 0,
                "indexing": [],
                "peer_reviewed": False,
                "date": "",
                "topic_keywords": [],
                "co_authors": [],
                "author_position": "",
                "summary": c,
                "other": "",
            }
        )
    return pubs


def extract_languages(sections: Dict[str, str]) -> List[str]:
    block = sections.get("languages", "")
    if block:
        toks = re.split(r"[,\n;]", block)
        return [t.strip().capitalize() for t in toks if t.strip()]
    # fallback common langs
    full = " ".join(sections.values()).lower()
    langs = []
    for l in ["english", "hindi", "gujarati", "marathi", "spanish", "french"]:
        if l in full and l.capitalize() not in langs:
            langs.append(l.capitalize())
    return langs


def extract_personal_info(
    sections: Dict[str, str],
    link_metadata: List[Dict],
    emails: List[str],
    phones: List[str],
) -> Dict[str, Any]:
    personal = {
        "email": emails[0] if emails else "",
        "phone": phones[0] if phones else "",
        "address": "",
        "linkedin": "",
        "github": "",
        "portfolio": "",
        "other": "",
    }
    # scan link metadata for linkedin/github/portfolio
    for lm in link_metadata or []:
        url = lm.get("normalized_url", "")
        if "linkedin" in url.lower():
            personal["linkedin"] = url
        if "github" in url.lower():
            personal["github"] = url
        if (
            "portfolio" in url.lower()
            or "behance" in url.lower()
            or "dribbble" in url.lower()
        ):
            personal["portfolio"] = url
    # try to extract address line from 'personal_info' or 'preamble'
    addr_block = (
        sections.get("personal_info")
        or sections.get("contact")
        or sections.get("preamble", "")
    )
    # naive: look for commas and numbers
    addr_lines = [l.strip() for l in addr_block.splitlines() if len(l.strip()) > 20]
    if addr_lines:
        personal["address"] = addr_lines[-1][:200]
    return personal


# Main builder
def build_candidate_from_sections(
    raw_sections: Dict[str, str],
    section_confidence: Optional[Dict[str, float]] = None,
    link_metadata: Optional[List[Dict]] = None,
    emails: Optional[List[str]] = None,
    phones: Optional[List[str]] = None,
) -> Dict[str, Any]:
    raw_sections = raw_sections or {}
    section_confidence = section_confidence or {}
    link_metadata = link_metadata or []
    emails = emails or []
    phones = phones or []

    # Normalize section keys (lowercase keys) for internal convenience
    sections = {k.lower(): v for k, v in raw_sections.items()}

    candidate = {
        "candidate_id": f"cand_{int(datetime.now().timestamp())}",
        "created_at": datetime.now().isoformat() + "Z",
        "personal_info": {},
        "features": {
            "skills": {},
            "roles": {},
            "experience": {},
            "education": {},
            "certifications": {},
            "hobbies": {},
            "personal_info": {},
            "projects": {},
            "competitions": {},
            "achievements": {},
            "open_source": {},
            "research_papers": {},
            "volunteering": {},
            "extra_activity": {},
            "social_accounts": {},
        },
        "raw_sections": sections,
        "section_confidence": section_confidence,
    }

    # personal info
    personal = extract_personal_info(sections, link_metadata, emails, phones)
    candidate["personal_info"] = personal
    candidate["features"]["personal_info"] = personal

    # skills
    skills_block = extract_and_normalize_skills(sections)
    candidate["features"]["skills"] = skills_block

    # roles
    roles = extract_roles(sections)
    candidate["features"]["roles"] = {"roles": roles}

    # projects
    projects = extract_projects(sections, link_metadata)
    candidate["features"]["projects"] = {"projects": projects}

    # education
    education = extract_education(sections)
    candidate["features"]["education"] = {"education": education}

    # experience
    experience = extract_experience(sections)
    candidate["features"]["experience"] = {"experience": experience}

    # certifications
    certifications = extract_certifications(sections)
    candidate["features"]["certifications"] = {"certifications": certifications}

    # achievements
    achievements = extract_achievements(sections)
    candidate["features"]["achievements"] = {"achievements": achievements}

    # languages
    languages = extract_languages(sections)
    candidate["features"]["languages"] = languages

    # open source
    open_src = extract_open_source(sections, link_metadata)
    candidate["features"]["open_source"] = open_src

    # research papers
    pubs = extract_research_papers(sections)
    candidate["features"]["research_papers"] = {"research_papers": pubs}

    # attach raw link/email/phone lists to top-level for convenience
    candidate["link_metadata"] = link_metadata
    candidate["all_emails"] = emails
    candidate["all_phones"] = phones

    return candidate
