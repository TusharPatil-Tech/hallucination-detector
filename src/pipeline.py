import os
import re
import json
import requests
import pandas as pd
import torch
import wikipediaapi
from dotenv import load_dotenv, find_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline as hf_pipeline

load_dotenv(find_dotenv())

MODEL = "openai/gpt-oss-120b"
CON_TOP = 1  # placeholder: top-1 vs top-3 gets decided on the WikiBio tuning split in Week 3

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
wiki = wikipediaapi.Wikipedia(
    user_agent="hallucination-detector/1.0 (student project)",
    language="en",
)

_embedder = None
_nli = None

VERDICT = {
    "entailment": "Supported",
    "contradiction": "Contradicted",
    "neutral": "Not enough info",
}

RESULT_COLUMNS = [
    "claim",
    "verdict",
    "confidence",
    "evidence",
    "similarity",
    "entailment_score",
    "contradiction_score",
    "neutral_score",
    "explanation",
]


def load_models():
    global _embedder, _nli
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    if _nli is None:
        _nli = hf_pipeline(
            "text-classification",
            model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            top_k=None,
            truncation=True,
            max_length=512,
            device=0 if torch.cuda.is_available() else -1,
        )


def _ask(prompt, max_tokens=1500, temperature=0):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content.strip()


def get_llm_answer(question):
    return _ask(f"Answer in 4-6 factual sentences.\n\nQuestion: {question}", 1000, 0.7)


def extract_claims(text):
    out = _ask(
        f"""Break the text below into independent, atomic factual claims.

Rules:
- One fact per claim.
- Replace pronouns (he, she, it, they) with the full name.
- Each claim must make sense on its own.
- Return ONLY a JSON list of strings, nothing else.

Text:
{text}"""
    )
    m = re.search(r"\[.*\]", out, re.DOTALL)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(0))
        if isinstance(parsed, list):
            return [str(c).strip() for c in parsed if str(c).strip()]
    except json.JSONDecodeError:
        pass
    return []


def find_entity(text):
    return _ask(
        "Return ONLY the name of the main person, place or thing this text is about, nothing else.\n\n"
        f"Text: {text}",
        500,
    )


def find_wiki_page(name):
    if not name or not name.strip():
        return None
    resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": name.strip(),
            "format": "json",
            "srlimit": 1,
        },
        headers={"User-Agent": "hallucination-detector/1.0 (student project)"},
        timeout=15,
    )
    resp.raise_for_status()
    hits = resp.json().get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None


def resolve_subject(sent, title):
    parts = title.split()
    if not parts:
        return sent
    last = re.escape(parts[-1])
    return re.sub(rf"^The {last}\b", f"The {title}", sent, flags=re.I)


def build_evidence(title):
    text = wiki.page(title).text
    sents = []
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        s = s.strip()
        if len(s.split()) >= 8 and s.endswith(".") and ":" not in s:
            sents.append(resolve_subject(s, title))
    if not sents:
        raise ValueError(f"Wikipedia page '{title}' does not contain enough usable text.")
    return sents, _embedder.encode(sents, convert_to_tensor=True)


def get_evidence_from(claim, sents, emb, top_k=5):
    if not sents:
        return []
    ce = _embedder.encode(claim, convert_to_tensor=True)
    scores = util.cos_sim(ce, emb)[0]
    top = scores.topk(min(top_k, len(sents)))
    return [(sents[i], float(s)) for s, i in zip(top.values, top.indices)]


def verify_claim(claim, evidence, thr=0.8, con_top=CON_TOP, min_sim=0.5, con_thr=0.9):
    if not evidence:
        return {
            "label": "neutral",
            "score": 0.0,
            "evidence": "",
            "similarity": 0.0,
            "scores": {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0},
            "explanation": "No relevant evidence sentences were found on Wikipedia.",
        }

    best_ent = None
    best_con = None
    top_neutral = None

    for rank, (sent, sim) in enumerate(evidence):
        out = _nli({"text": sent, "text_pair": claim})
        out = out[0] if isinstance(out[0], list) else out
        sc = {o["label"].lower(): float(o["score"]) for o in out}

        candidate = {
            "evidence": sent,
            "similarity": sim,
            "scores": {
                "entailment": sc.get("entailment", 0.0),
                "contradiction": sc.get("contradiction", 0.0),
                "neutral": sc.get("neutral", 0.0),
            },
        }

        if rank == 0:
            top_neutral = candidate

        if best_ent is None or candidate["scores"]["entailment"] > best_ent["scores"]["entailment"]:
            best_ent = candidate

        if rank < con_top and sim >= min_sim:
            if best_con is None or candidate["scores"]["contradiction"] > best_con["scores"]["contradiction"]:
                best_con = candidate

    if best_ent and best_ent["scores"]["entailment"] >= thr:
        score = best_ent["scores"]["entailment"]
        return {
            "label": "entailment",
            "score": score,
            "evidence": best_ent["evidence"],
            "similarity": best_ent["similarity"],
            "scores": best_ent["scores"],
            "explanation": (
                f"Supported by Wikipedia evidence "
                f"(entailment: {score:.2f}, similarity: {best_ent['similarity']:.2f})."
            ),
        }

    if best_con and best_con["scores"]["contradiction"] >= con_thr:
        score = best_con["scores"]["contradiction"]
        return {
            "label": "contradiction",
            "score": score,
            "evidence": best_con["evidence"],
            "similarity": best_con["similarity"],
            "scores": best_con["scores"],
            "explanation": (
                f"Contradicted by Wikipedia evidence "
                f"(contradiction: {score:.2f}, similarity: {best_con['similarity']:.2f})."
            ),
        }

    # Neutral fallback: use the top semantically matched sentence and its actual neutral score
    chosen = top_neutral
    score = chosen["scores"]["neutral"]
    return {
        "label": "neutral",
        "score": score,
        "evidence": chosen["evidence"],
        "similarity": chosen["similarity"],
        "scores": chosen["scores"],
        "explanation": (
            f"Insufficient evidence to verify or refute "
            f"(neutral: {score:.2f}, entailment: {chosen['scores']['entailment']:.2f}, "
            f"contradiction: {chosen['scores']['contradiction']:.2f}, "
            f"similarity: {chosen['similarity']:.2f})."
        ),
    }


def quantify_hallucination(df):
    """Compute claim-level hallucination and factuality metrics from results DataFrame."""
    total = int(len(df))
    if total == 0 or "verdict" not in df.columns:
        return {
            "total": 0,
            "supported": 0,
            "contradicted": 0,
            "unverifiable": 0,
            "factuality_score": 0.0,
            "hallucination_rate": 0.0,
            "unverified_rate": 0.0,
            "hallucination_risk_score": 0.0,
        }

    counts = df["verdict"].value_counts().to_dict()
    supported = int(counts.get("Supported", 0))
    contradicted = int(counts.get("Contradicted", 0))
    unverifiable = int(counts.get("Not enough info", 0))

    return {
        "total": total,
        "supported": supported,
        "contradicted": contradicted,
        "unverifiable": unverifiable,
        "factuality_score": round((supported / total) * 100, 2),
        "hallucination_rate": round((contradicted / total) * 100, 2),
        "unverified_rate": round((unverifiable / total) * 100, 2),
        "hallucination_risk_score": round(((contradicted + unverifiable) / total) * 100, 2),
    }


def run_pipeline(question=None, answer=None):
    """Give a question (Groq answers it) OR an answer you pasted. Returns (page title, answer, results table)."""
    load_models()
    if answer is None:
        answer = get_llm_answer(question)

    claims = extract_claims(answer)
    title = find_wiki_page(find_entity(question or answer))
    if not title or not wiki.page(title).exists():
        raise ValueError("Could not find a Wikipedia page for this topic.")

    if not claims:
        return title, answer, pd.DataFrame(columns=RESULT_COLUMNS)

    sents, emb = build_evidence(title)
    rows = []
    for c in claims:
        r = verify_claim(c, get_evidence_from(c, sents, emb))
        rows.append(
            {
                "claim": c,
                "verdict": VERDICT[r["label"]],
                "confidence": round(r["score"], 2),
                "evidence": r["evidence"],
                "similarity": round(r["similarity"], 2),
                "entailment_score": round(r["scores"]["entailment"], 2),
                "contradiction_score": round(r["scores"]["contradiction"], 2),
                "neutral_score": round(r["scores"]["neutral"], 2),
                "explanation": r["explanation"],
            }
        )
    return title, answer, pd.DataFrame(rows, columns=RESULT_COLUMNS)
