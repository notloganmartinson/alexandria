# src/alexandria/storage/schemas.py
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class EntityRelation(BaseModel):
    """Represents a structured multi-hop relation graph edge."""
    source: str = Field(..., description="The originating entity node name.")
    target: str = Field(..., description="The terminating target entity node name.")
    relation: str = Field(..., description="The semantic edge relationship type (e.g., works_with, impacts).")

class ContentChunkMetadata(BaseModel):
    """Vitals tracking array for content ingestion tracking."""
    url: str = Field(..., description="Source URL mapping origin.")
    domain: str = Field(..., description="Extracted authority domain.")
    timestamp_epoch: int = Field(..., description="Unix timestamp of acquisition sequence.")
    lexical_density_score: float = Field(..., description="Heuristic semantic text profiling metric.")
    topic_cluster: str = Field(..., description="Dynamically assigned thematic data bucket.")

class IngestionDocumentRecord(BaseModel):
    """The master LanceDB record schema reflecting the high-fidelity hybrid storage architecture."""
    chunk_id: str = Field(..., description="Deterministic cryptographic primary key string.")
    text: str = Field(..., description="Cleaned document string sequence.")
    metadata: ContentChunkMetadata = Field(..., description="Deep structural analytics payload object.")
    relations: List[EntityRelation] = Field(default_factory=list, description="Extracted entity triples tied to this chunk.")
