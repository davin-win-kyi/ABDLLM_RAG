"""
Important sources:
- WCAG 2.2 documentation
-

Things to consider: 

Query Enhancement Module
1. query rewriting so that it is tailored to WCAG
- this might need to include a few common expansion techniques
    - Query rewriting 
        - expand so that it is able to better retrive WCAG information 
    - step back prompting
        - to also better generalize the query to a wide range of 
          WCAG information
Doing step-back first is most likley best as it will allow for 
a rewriter with a cleaner target and reduces the amount of 
off-topic expansions

# maybe the most important part is to 

Step back Prompting
- use the most up to date WCAG documentation
"""

# query_enhancement.py
# file: query_enhancement_minimal_no_meta.py
from __future__ import annotations
import json, csv
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from step_back_prompter import StepBackPrompter
from query_rewriter import QueryRewriter

from dotenv import load_dotenv
import os

load_dotenv()  # looks for .env in the working dir, then parents
api_key = os.getenv("OPENAI_API_KEY")

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from sentence_transformers import CrossEncoder

def _rrf(results_list: List[Dict[str, Any]], k: int = 10, c: int = 60) -> List[str]:
    scores: Dict[str, float] = {}
    for res in results_list:
        ids = res.get("ids", [[]])[0] if res.get("ids") else []
        for rank, _id in enumerate(ids):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (c + rank)
    return [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]

@dataclass
class QueryEnhancement:
    profile_path: str = "init_profile.json"
    stepback_model: str = "gpt-5"
    rewrite_model: str = "gpt-5"

    chroma_path: str = "./wcag_chroma"
    collection_name: str = "wcag_docs"
    embed_model: str = "all-MiniLM-L6-v2"

    n_each: int = 40
    top_k: int = 12

    def _load_profile(self) -> Dict[str, Any]:
        with open(self.profile_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _collection(self):
        client = chromadb.PersistentClient(path=self.chroma_path)
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=self.embed_model)
        return client.get_collection(self.collection_name, embedding_function=embed_fn)

    def run(self) -> Dict[str, Any]:
        profile = self._load_profile()

        plan = StepBackPrompter(model=self.stepback_model).generate(profile)
        rew  = QueryRewriter(model=self.rewrite_model).generate(plan)

        recall = rew["queries"]["recall_query"]
        precise = rew["queries"]["precise_query"]
        hyde = rew["queries"]["hyde_paragraph"]

        col = self._collection()
        c1 = col.query(query_texts=[recall],  n_results=self.n_each)   # ids, documents only
        c2 = col.query(query_texts=[precise], n_results=self.n_each)
        c3 = col.query(query_texts=[hyde],    n_results=self.n_each)

        fused_ids = _rrf([c1, c2, c3], k=self.top_k)
        fetched = col.get(ids=fused_ids)  # returns {"ids": [...], "documents": [...]} (no metadatas)

        # Normalize to a simple list of records
        results = []
        for _id, doc in zip(fetched.get("ids", []), fetched.get("documents", [])):
            preview = (doc[:320] + "...") if doc and len(doc) > 320 else (doc or "")
            results.append({"id": _id, "document": doc or "", "preview": preview})

        return {
            "step_back": {
                "objective": plan.get("objective"),
                "constraints": plan.get("constraints", []),
                "invariants": plan.get("invariants", []),
                "open_variables": plan.get("open_variables", []),
                "generalized_query": plan.get("generalized_query", "")
            },
            "queries": {
                "recall_query": recall,
                "precise_query": precise,
                "hyde_paragraph": hyde
            },
            "results": results  # list[{"id","document","preview"}]
        }

# -------- write helpers --------
def write_results_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def write_results_jsonl(path: str, results: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_results_csv(path: str, results: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "document", "preview"])
        w.writeheader()
        for r in results:
            w.writerow(r)

def rerank_docs(query: str, documents: list[str], top_k: int | None = None):
    """
    Pointwise rerank with a cross-encoder.
    query: your precise_query (or HyDE paragraph)
    documents: list[str] from your fused Top-N
    returns: list of dicts sorted by ce score desc
    """
    # Optional: truncate very long docs to control latency/cost
    pruned = [(i, d if d is not None else "") for i, d in enumerate(documents)]
    # Make (query, doc) pairs
    pairs = [(query, d[:2000]) for _, d in pruned]  # 2k chars is a practical cap

    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", trust_remote_code=True)
    scores = ce.predict(pairs)  # numpy array of floats; higher = more relevant
    ranked = sorted(
        [{"index": i, "doc": d, "score": float(s)} for (i, d), s in zip(pruned, scores)],
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranked if top_k is None else ranked[:top_k]

# -------- CLI --------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="StepBack → Rewrite → RAG (no metadata) → save")
    ap.add_argument("--profile", default="init_profile.json")
    ap.add_argument("--persist", default="./wcag_chroma")
    ap.add_argument("--collection", default="wcag_docs")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out", default="results.jsonl", help="results.{json|jsonl|csv}")
    args = ap.parse_args()

    qe = QueryEnhancement(
        profile_path=args.profile,
        chroma_path=args.persist,
        collection_name=args.collection,
        top_k=args.k
    )
    out = qe.run()

    # choose writer by extension
    ext = args.out.lower().split(".")[-1]

    if ext == "jsonl":
        write_results_jsonl(args.out, out["results"])
    elif ext == "json":
        write_results_json(args.out, out)
    elif ext == "csv":
        write_results_csv(args.out, out["results"])
    else:
        raise ValueError("Use .jsonl, .json, or .csv")

    print(f"Saved Top-{args.k} to {args.out}")

    precise_query = out["queries"]["precise_query"]

    docs = [r.get("document", "") for r in out["results"]]
    top_k_docs = rerank_docs(precise_query, docs, top_k=10)

    with open("top_k_docs.json", "w", encoding="utf-8") as f:
        json.dump(top_k_docs, f, ensure_ascii=False, indent=2)

    print("TOP K Docs: ", top_k_docs)
