import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import numpy as np
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def get_llm(temp=0.5):
    return ChatOpenAI(model_name="gpt-4o", temperature=temp, openai_api_key=OPENAI_API_KEY)


def get_embeddings():
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def generate_answer(retrieved_text, query):
    llm = get_llm(0.2)

    template = PromptTemplate(
        template="""
        Use the following context to answer the question:
        {context}
        Question: {query}
        Answer:""",
        input_variables=["context", "query"],
    )
    final_prompt = template.format(context=retrieved_text, query=query)
    return llm.invoke(final_prompt)


def compute_embedding(text: str) -> np.ndarray:
    return np.array(embedding_model.embed_query(text))
