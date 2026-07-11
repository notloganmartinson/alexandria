# this module encapsualtes the precise PyArrow structs required to strongly type the data ingestion pipeline, explicitly sizing the semantic vector column to 768 parameters
import pyarrow as pa

KNOWLEDGE_NODES_SCHEMA = pa.schema(
    [
        pa.field("node_id", pa.string(), nullable=False),
        pa.field("node_type", pa.string(), nullable=False),
        pa.field("source_registry_id", pa.string(), nullable=False),
        pa.field("topic_cluster", pa.string(), nullable=False),
        pa.field("structural_breadcrumbs", pa.string(), nullable=False),
        pa.field("semantic_content", pa.string(), nullable=False),
        pa.field("vector", pa.list_(pa.float32(), 768), nullable=False),
        pa.field("lexical_density_score", pa.float32(), nullable=False),
        pa.field("node_metadata", pa.string(), nullable=False),
    ]
)

SOURCE_REGISTRY_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_uri", pa.string(), nullable=False),
        pa.field("raw_text_hash", pa.string(), nullable=False),
        pa.field("harvest_timestamp", pa.int64(), nullable=False),
        pa.field("total_nodes_generated", pa.int32(), nullable=False),
        pa.field("raptor_compilation_state", pa.bool_(), nullable=False),
    ]
)
