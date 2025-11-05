# query_rewriter.py
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from openai import OpenAI  # pip install openai

from dotenv import load_dotenv
import os

load_dotenv()  # looks for .env in the working dir, then parents
api_key = os.getenv("OPENAI_API_KEY")

def _extract_json_block(text: str) -> Dict[str, Any]:
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1:
        raise ValueError("QueryRewriter: model did not return JSON.")
    return json.loads(text[i:j+1])

@dataclass
class QueryRewriter:
    """
    Input (step_back dict):
      - objective: str
      - constraints: list[str]
      - invariants: list[str]
      - open_variables: list[str]
      - generalized_query: str

    Output:
      {
        "queries": {
          "recall_query": str,
          "precise_query": str,
          "hyde_paragraph": str
        },
        "metadata": {
          "objective": str,
          "constraints": list[str],
          "invariants": list[str],
          "open_variables": list[str]
        }
      }
    """
    model: str = "gpt-5"
    temperature: float = 0.1
    client: Optional[Any] = None

    def __post_init__(self):
        self.client = self.client or OpenAI()

    def generate(self, step_back: Dict[str, Any]) -> Dict[str, Any]:
        generalized_query = step_back.get("generalized_query", "")
        constraints: List[str] = step_back.get("constraints", [])
        invariants: List[str]  = step_back.get("invariants", [])
        open_vars: List[str]   = step_back.get("open_variables", [])
        objective: str         = step_back.get("objective", "")

        print("Query Rewriter generating...")

        prompt = f"""
        You are a retrieval query rewriter for a WCAG Techniques corpus.
        Using the generalized query and constraints below, return ONLY JSON with:
        {{
        "recall_query": "broad, synonyms/aliases/morphology; maximize recall",
        "precise_query": "canonical WCAG/ARIA phrasing; include SC IDs only if VERY likely",
        "hyde_paragraph": "3–5 sentence neutral, plausible paragraph (no citations/IDs)"
        }}
        Rules:
        - Each field ≤ 80 words.
        - Reflect these invariants when sensible: {json.dumps(invariants, ensure_ascii=False)}
        - Stay aligned with constraints: {json.dumps(constraints, ensure_ascii=False)}
        - Avoid brand/platform names unless implied by the query.

        GENERALIZED_QUERY:
        {generalized_query}
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        rewrites = _extract_json_block(resp.choices[0].message.content.strip())

        return {
            "queries": {
                "recall_query": rewrites["recall_query"],
                "precise_query": rewrites["precise_query"],
                "hyde_paragraph": rewrites["hyde_paragraph"],
            },
            "metadata": {
                "objective": objective,
                "constraints": constraints,
                "invariants": invariants,
                "open_variables": open_vars,
            },
        }
