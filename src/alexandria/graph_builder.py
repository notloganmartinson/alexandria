# src/alexandria/graph_builder.py
import os
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import List
from alexandria.storage.schemas import EntityRelation

# We define a wrapper Pydantic model so the LLM knows to return a list of relations
class GraphExtractionResult(BaseModel):
    relations: List[EntityRelation] = Field(
        description="A list of entity relationships extracted from the text."
    )

class OpenRouterExtractor:
    def __init__(self, api_key: str = None, model: str = None):
        # Grabs the key from the environment if not passed directly
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is missing. Set OPENROUTER_API_KEY in your environment.")

        # Initialize the OpenAI async client but point it to OpenRouter
        self.client = instructor.from_openai(
            AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            ),
            mode=instructor.Mode.JSON,
        )
        
        # Using a fast, cheap model that excels at JSON for testing (can swap to deepseek or local later)
        self.model = model or os.getenv("ALEXANDRIA_EXTRACTION_MODEL", "openai/gpt-4o-mini") 

    async def extract_relations(self, markdown_text: str) -> List[EntityRelation]:
        """
        Reads raw markdown and uses OpenRouter to force-extract strict Graph Triples.
        """
        sliced_text = markdown_text[:4000]
        prompt = f"""
        You are an expert data ontologist building a knowledge graph.
        Read the following text and extract the most important entity relationships.
        Entities should be specific nouns (people, technologies, companies, concepts).
        Relations should be verbs connecting them (e.g., 'invented', 'powers', 'funded').

        TEXT TO ANALYZE:
        {sliced_text}
        """

        try:
            # Instructor forces the LLM to output exactly the GraphExtractionResult schema
            response = await self.client.chat.completions.create(
                model=self.model,
                response_model=GraphExtractionResult,
                messages=[
                    {"role": "system", "content": "Extract strict JSON relations."},
                    {"role": "user", "content": prompt},
                ],
                max_retries=3
            )
            return response.relations
        except Exception as e:
            print(f"Failed to extract relations: {e}")
            return []
