from typing import Annotated, List, Literal, Optional

import operator
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class SearchResult(BaseModel):
    source: str
    content: str
    url: str
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    chunk_id: str = ""
    content_hash: str = ""
    is_memory: bool = False


class Claim(BaseModel):
    claim: str
    source: str
    source_url: str = ""
    confidence: Literal["high", "medium", "low"]
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_snippet: str = ""
    evidence_chunk_id: str = ""


class Summary(BaseModel):
    claims: List[Claim]
    raw_summary: str


class CritiqueGap(BaseModel):
    description: str
    suggested_query: str


class Critique(BaseModel):
    gaps: List[CritiqueGap]
    verified_claims: List[str]
    should_continue: bool
    reasoning: str


class ResearchState(TypedDict):
    query: str
    conversation_context: str
    memory_mode: Literal["balanced", "prefer_memory", "search_only"]
    search_results: Annotated[List[SearchResult], operator.add]
    summary: Optional[Summary]
    critique: Optional[Critique]
    iterations: int
    final_answer: str
    source_urls: List[str]
    evidence_coverage: float
    avg_confidence: float
    agent_logs: Annotated[List[dict], operator.add]
    error: Optional[str]
