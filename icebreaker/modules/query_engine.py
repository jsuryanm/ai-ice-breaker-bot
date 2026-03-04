import logging
from typing import Any, Dict, Optional

from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.core.output_parsers import PydanticOutputParser

from icebreaker.modules.llm_interface import get_llm
from icebreaker.modules.output_schema import PersonFacts
from icebreaker.modules import config


logger = logging.getLogger(__name__)

def generate_initial_facts(index: VectorStoreIndex) -> str:
    """Generates interesting facts about the person\'s career or education.
    
    Args:
        index: VectorStoreIndex containing the LinkedIn profile data.
        
    Returns:
        String containing interesting facts about the person.
    """
    try:
        # Create LLM for generating facts
        groq_llm = get_llm(
            temperature=0.0,
        )

        output_parser = PydanticOutputParser(PersonFacts) 
        
        # Create prompt template
        facts_prompt = PromptTemplate(template=config.INITIAL_FACTS_TEMPLATE + "\n\n" + output_parser.get_format_string())
        
        # Create query engine
        query_engine = index.as_query_engine(
            streaming=False,
            similarity_top_k=config.SIMILARITY_TOP_K,
            llm=groq_llm,
            text_qa_template=facts_prompt
        )
        
        # Execute the query
        query = "Provide three interesting facts about this person\'s career or education."
        response = query_engine.query(query)
        
        # Return the facts
        parsed_output = output_parser.parse(response.response)
        facts_text = f"""
                    • {parsed_output.fact1}

                    • {parsed_output.fact2}

                    • {parsed_output.fact3}
                    """

        return facts_text
    except Exception as e:
        logger.error(f"Error in generate_initial_facts: {e}")
        return "Failed to generate initial facts."

def answer_user_query(index: VectorStoreIndex, user_query: str) -> Any:
    """Answers the user\'s question using the vector database and the LLM.
    
    Args:
        index: VectorStoreIndex containing the LinkedIn profile data.
        user_query: The user\'s question.
        
    Returns:
        Response object containing the answer to the user\'s question.
    """
    try:
        # Create LLM for answering questions
        groq_llm = get_llm(
            temperature=0.0,
        )
        
        # Create prompt template
        question_prompt = PromptTemplate(template=config.USER_QUESTION_TEMPLATE)
        
        # Retrieve relevant nodes
        base_retriever = index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)
        source_nodes = base_retriever.retrieve(user_query)
        
        # Build context string
        context_str = "\n\n".join([node.node.get_text() for node in source_nodes])
        
        # Create query engine
        query_engine = index.as_query_engine(
            streaming=False,
            similarity_top_k=config.SIMILARITY_TOP_K,
            llm=groq_llm,
            text_qa_template=question_prompt
        )
        
        # Execute the query
        answer = query_engine.query(user_query)
        return answer
    except Exception as e:
        logger.error(f"Error in answer_user_query: {e}")
        return "Failed to get an answer."
