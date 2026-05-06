"""
SmartShopper Assistant — FastAPI Endpoint
Langkah 6: Menyajikan sistem rekomendasi sebagai REST API
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from functools import partial
from typing import Optional
import re
import json
import os
import dotenv

dotenv.load_dotenv()

from haystack import Pipeline, component
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from haystack.utils import Secret
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.tools.tool import Tool
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever
from pymongo import MongoClient
from typing import List, Annotated

from template import METADATA_FILTER_TEMPLATE

# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="SmartShopper Assistant API",
    description="Personalized product recommendation API menggunakan Haystack + MongoDB Atlas",
    version="1.0.0",
)

# ── Request / Response Models ────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query: str

class RecommendResponse(BaseModel):
    query: str
    paraphrased_query: str
    recommendation: str

# ── Haystack Components (sama persis seperti website.py) ─────────────────────

@component
class GetMaterials:
    def __init__(self):
        client = MongoClient(os.environ["MONGO_CONNECTION_STRING"])
        self.collection = client.depato_store.materials

    @component.output_types(materials=List[str])
    def run(self):
        return {"materials": [doc["name"] for doc in self.collection.find()]}


@component
class GetCategories:
    def __init__(self):
        client = MongoClient(os.environ["MONGO_CONNECTION_STRING"])
        self.collection = client.depato_store.categories

    @component.output_types(categories=List[str])
    def run(self):
        return {"categories": [doc["name"] for doc in self.collection.find()]}


class ParaphraserPipeline:
    def __init__(self, chat_message_store):
        self.pipeline = Pipeline()
        self.pipeline.add_component(
            "prompt_builder",
            ChatPromptBuilder(variables=["query", "memories"], required_variables=["query", "memories"]),
        )
        self.pipeline.add_component(
            "generator",
            OpenAIChatGenerator(model="llama-3.3-70b-versatile", api_key=Secret.from_token(os.environ["GROQ_API_KEY"]), api_base_url="https://api.groq.com/openai/v1"),
        )
        self.pipeline.add_component("memory_retriever", ChatMessageRetriever(chat_message_store))
        self.pipeline.connect("prompt_builder.prompt", "generator.messages")
        self.pipeline.connect("memory_retriever", "prompt_builder.memories")

    def run(self, query: str) -> str:
        messages = [
            ChatMessage.from_system("You are a helpful assistant that paraphrases user queries based on previous conversations."),
            ChatMessage.from_user(
                "Please paraphrase the following query based on the conversation history. "
                "If history is empty, return the query as is.\n"
                "history:\n{% for memory in memories %}{{memory.text}}{% endfor %}\n"
                "query: {{query}}\nanswer:"
            ),
        ]
        res = self.pipeline.run(
            data={
                "prompt_builder": {"query": query, "template": messages},
                "memory_retriever": {"chat_history_id": "default"},
            },
            include_outputs_from=["generator"],
        )
        return res["generator"]["replies"][0].text


class MetaDataFilterPipeline:
    def __init__(self):
        self.pipeline = Pipeline()
        self.pipeline.add_component("materials", GetMaterials())
        self.pipeline.add_component("categories", GetCategories())
        self.pipeline.add_component(
            "prompt_builder",
            ChatPromptBuilder(variables=["input", "materials", "categories"], required_variables=["input", "materials", "categories"]),
        )
        self.pipeline.add_component(
            "generator",
            OpenAIChatGenerator(model="llama-3.3-70b-versatile", api_key=Secret.from_token(os.environ["GROQ_API_KEY"]), api_base_url="https://api.groq.com/openai/v1"),
        )
        self.pipeline.connect("materials.materials", "prompt_builder.materials")
        self.pipeline.connect("categories.categories", "prompt_builder.categories")
        self.pipeline.connect("prompt_builder.prompt", "generator.messages")

    def run(self, query: str) -> str:
        template = [ChatMessage.from_user(METADATA_FILTER_TEMPLATE)]
        res = self.pipeline.run(data={"prompt_builder": {"input": query, "template": template}})
        return res["generator"]["replies"][0].text


class RetrieveAndGeneratePipeline:
    def __init__(self, document_store):
        self.pipeline = Pipeline()
        self.pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
        self.pipeline.add_component("retriever", MongoDBAtlasEmbeddingRetriever(document_store=document_store, top_k=10))
        self.pipeline.add_component(
            "prompt_builder",
            ChatPromptBuilder(variables=["query", "documents"], required_variables=["query", "documents"]),
        )
        self.pipeline.add_component(
            "generator",
            OpenAIChatGenerator(model="llama-3.3-70b-versatile", api_key=Secret.from_token(os.environ["GROQ_API_KEY"]), api_base_url="https://api.groq.com/openai/v1"),
        )
        self.pipeline.connect("embedder", "retriever")
        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder.prompt", "generator.messages")

    def run(self, query: str, filters: dict = {}) -> str:
        messages = [
            ChatMessage.from_system("You are a helpful shop assistant that gives product recommendations."),
            ChatMessage.from_user(
                "Give a list of products that best match the query.\n\n"
                "Format each product as:\n"
                "<index>. <product_name>\nPrice: <price>\nMaterial: <material>\n"
                "Category: <category>\nBrand: <brand>\nRecommendation: <why recommended>\n\n"
                "Query: {{query}}\n"
                "{% if documents|length > 0 %}Products:\n"
                "{% for p in documents %}"
                "---\n{{loop.index}}. {{p.meta.title}} | ${{p.meta.price}} | {{p.meta.material}} | {{p.meta.category}}\n{{p.content}}\n"
                "{% endfor %}{% else %}No matching products found.{% endif %}\n\nAnswer:"
            ),
        ]
        res = self.pipeline.run(
            data={
                "embedder": {"text": query},
                "retriever": {"filters": filters},
                "prompt_builder": {"query": query, "template": messages},
            },
            include_outputs_from=["generator"],
        )
        return res["generator"]["replies"][0].text


# ── App State (singleton pipelines, inisialisasi saat startup) ───────────────

chat_message_store = InMemoryChatMessageStore()
chat_message_writer = ChatMessageWriter(chat_message_store)

document_store = MongoDBAtlasDocumentStore(
    database_name="depato_store",
    collection_name="products",
    vector_search_index="vector_index",
    full_text_search_index="search_index",
)

paraphraser = ParaphraserPipeline(chat_message_store)
metadata_filter = MetaDataFilterPipeline()
rag_pipeline = RetrieveAndGeneratePipeline(document_store)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Cek status API."""
    return {"status": "healthy", "service": "SmartShopper API"}


@app.post("/recommend", response_model=RecommendResponse)
def get_recommendation(request: RecommendRequest):
    """
    Endpoint utama: terima query → paraphrase → filter → retrieve → generate.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query tidak boleh kosong.")

    # Step 1: Paraphrase query (dengan context chat history)
    paraphrased = paraphraser.run(request.query)

    # Step 2: Extract metadata filters dari query
    raw_filter = metadata_filter.run(paraphrased)
    filters = {}
    try:
        match = re.search(r"```json\n(.*?)\n```", raw_filter, re.DOTALL)
        if match:
            filters = json.loads(match.group(1))
    except Exception:
        filters = {}

    # Step 3: Retrieve products + generate recommendation
    recommendation = rag_pipeline.run(paraphrased, filters)

    # Step 4: Simpan ke chat memory
    chat_message_writer.run(messages=[ChatMessage.from_user(request.query)], chat_history_id="default")
    chat_message_writer.run(messages=[ChatMessage.from_assistant(recommendation)], chat_history_id="default")

    return RecommendResponse(
        query=request.query,
        paraphrased_query=paraphrased,
        recommendation=recommendation,
    )


@app.delete("/history")
def clear_history():
    """Reset chat history / memory."""
    global chat_message_store, chat_message_writer, paraphraser
    chat_message_store = InMemoryChatMessageStore()
    chat_message_writer = ChatMessageWriter(chat_message_store)
    paraphraser = ParaphraserPipeline(chat_message_store)
    return {"status": "history cleared"}


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)