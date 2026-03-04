# Alzheimer_CNN/src/rag_demo.py
from openai import OpenAI
import numpy as np
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

KNOWLEDGE_BASE = [
    "Hippocampal volume reduction is a primary biomarker for AD.",
    "Early stage AD often presents with thinning of the entorhinal cortex.",
]

def get_embedding(text):
    # simple placeholder
    return np.random.rand(1536)

def get_rag_qa(question):
    query_vec = get_embedding(question)
    return "Mock answer based on RAG context."
