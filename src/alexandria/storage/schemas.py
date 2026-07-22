# src/alexandria/storage/schemas.py
from typing import List, Optional, Type
from pydantic import BaseModel, Field
from lancedb.pydantic import LanceModel, Vector

class EntityRelation(BaseModel):
    source: str = Field(..., description="The originating entity node name.")
    target: str = Field(..., description="The terminating target entity node name.")
    relation: str = Field(..., description="The semantic edge relationship type.")

class GraphEdgeRecord(LanceModel):
    source: str = Field(..., description="The originating entity node name.")
    target: str = Field(..., description="The terminating target entity node name.")
    relation: str = Field(..., description="The semantic edge relationship type.")
    chunk_id: str = Field(..., description="Deterministic primary key string linking to the vector chunk.")

class ContentChunkMetadata(BaseModel):
    url: Optional[str] = None
    domain: Optional[str] = None
    timestamp_epoch: Optional[int] = None
    lexical_density_score: Optional[float] = None
    topic_cluster: Optional[str] = None
    file_path: Optional[str] = None
    chunk_type: Optional[str] = None
    node_type: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    language: Optional[str] = None
    header: Optional[str] = None

def create_document_schema(dim: int) -> Type[LanceModel]:
    class IngestionDocumentRecord(LanceModel):
        chunk_id: str = Field(..., description="Deterministic cryptographic primary key string.")
        text: str = Field(..., description="Cleaned document string sequence.")
        vector: Vector(dim) = Field(..., description="Dense numerical embedding.")
        metadata: ContentChunkMetadata = Field(..., description="Deep structural analytics payload object.")
        relations: List[EntityRelation] = Field(default_factory=list)
    return IngestionDocumentRecord

def create_entity_schema(dim: int) -> Type[LanceModel]:
    class EntityRecord(LanceModel):
        name: str = Field(..., description="Canonical entity name.")
        vector: Vector(dim) = Field(..., description="Dense embedding of the entity string.")
    return EntityRecord

def create_community_schema(dim: int) -> Type[LanceModel]:
    class CommunityRecord(LanceModel):
        community_id: str = Field(..., description="UUID of the community cluster.")
        summary: str = Field(..., description="LLM-generated high-level abstract.")
        vector: Vector(dim) = Field(..., description="Dense embedding of the abstract.")
        entities: List[str] = Field(..., description="Members of this community.")
    return CommunityRecord
