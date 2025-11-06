"""
Description: zoom out and personalize the goal and rules
- turn a messy/short ask + user profile into a clean plan 

Outputs: 
- objective: what we're really solving 
- constraints: WCAG 2.2
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from openai import OpenAI  # pip install openai

from dotenv import load_dotenv
import os

load_dotenv()  # looks for .env in the working dir, then parents
api_key = os.getenv("OPENAI_API_KEY")

@dataclass
class StepBackPrompter:
    model: str = "gpt-5"
    client: Optional[Any] = None

    def __post_init__(self):
        self.client = self.client or OpenAI()

    def generate(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input: profile dict (e.g., loaded from init_profile.json)
        Output keys:
          - objective, constraints[], invariants[], open_variables[], generalized_query
        """

        """
        Objectives: it is the high level overview of the task
        Constraints: helps give the bounaries/filters on what is outputted in query rewrite 
        Invariants: used to figure out how to rerank 
        open_variables: a way to personalize the output
        Generalized_query: it is like a clean slate which the QueryRewriter can 
                           build off of
        """

        print("Step back prompter generating...: ", profile)

        prompt = f"""
        Return ONLY compact JSON with keys:
        - objective (1–2 sentences)
        - constraints (list; prefer 'WCAG 2.2 AA'; include doc types if relevant)
        - invariants (list; official WCAG/ARIA must-haves)
        - open_variables (list)
        - generalized_query (vendor-neutral, retrieval-friendly)
        Do NOT invent Success Criterion IDs. Each field ≤ 40 words.
        USER_PROFILE: {json.dumps(profile, ensure_ascii=False)}
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()
        i, j = text.find("{"), text.rfind("}")
        if i == -1 or j == -1:
            raise ValueError("StepBackPrompter: model did not return JSON.")
        return json.loads(text[i:j+1])