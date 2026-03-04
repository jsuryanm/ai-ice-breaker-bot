from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq 
from icebreaker.modules import config
from icebreaker.modules.config import EMBEDDING_MODEL_ID,LLM_MODEL_ID,TEMPERATURE
from icebreaker.modules.data_extraction import logger
from dotenv import load_dotenv

load_dotenv()

_llm_instance = None


def create_hf_embeddings() -> HuggingFaceEmbedding:
    """Creates HuggingFaceEmbeddings model for vector representation"""
    hf_embeddings =  HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_ID)
    logger.info(f"Created huggingface embeddings using:{EMBEDDING_MODEL_ID}")
    return hf_embeddings

def get_llm(temperature: float = TEMPERATURE) -> Groq:
    """Lazy load Groq LLM (singleton pattern)."""

    global _llm_instance

    if _llm_instance is None:
        logger.info(f"Initializing Groq LLM: {LLM_MODEL_ID}")

        _llm_instance = Groq(
            model=LLM_MODEL_ID,
            temperature=temperature
        )

    return _llm_instance


def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model and reset the cached instance."""

    global _llm_instance
    from icebreaker.modules import config

    config.LLM_MODEL_ID = new_model_id
    _llm_instance = None

    logger.info(f"Changed LLM model to: {new_model_id}")