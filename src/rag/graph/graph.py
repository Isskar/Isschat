"""
LangGraph workflow definition for RAG pipeline.
Orchestrates query processing, retrieval, and generation.
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END

from .state import RAGState
from .nodes import QueryProcessorNode, RetrieverNode, GeneratorNode
from ...config import get_config


def should_retrieve(state: RAGState) -> str:
    """Decide whether to proceed with retrieval"""
    if state.get("error"):
        return "error_handler"
    if state.get("should_retrieve", False):
        return "retrieve"
    return "generate"  # Skip retrieval if not needed


def should_generate(state: RAGState) -> str:
    """Decide whether to proceed with generation"""
    if state.get("error"):
        return "error_handler"
    return "generate"


def error_handler(state: RAGState) -> Dict[str, Any]:
    """Handle errors in the workflow"""
    error_msg = state.get("error", "Unknown error occurred")
    return {"answer": f"Une erreur s'est produite: {error_msg}", "sources": "Erreur", "step": "error_handled"}


def create_rag_graph() -> StateGraph:
    """Create the RAG workflow graph"""

    # Initialize nodes
    query_processor = QueryProcessorNode()
    retriever = RetrieverNode()
    generator = GeneratorNode()

    # Create the graph
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("process_query", query_processor)
    workflow.add_node("retrieve", retriever)
    workflow.add_node("generate", generator)
    workflow.add_node("error_handler", error_handler)

    # Define the flow
    workflow.set_entry_point("process_query")

    # Add conditional edges
    workflow.add_conditional_edges(
        "process_query",
        should_retrieve,
        {"retrieve": "retrieve", "generate": "generate", "error_handler": "error_handler"},
    )

    workflow.add_conditional_edges(
        "retrieve", should_generate, {"generate": "generate", "error_handler": "error_handler"}
    )

    # End points
    workflow.add_edge("generate", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile()


class RAGGraph:
    """Wrapper class for the RAG workflow graph"""

    def __init__(self):
        self.config = get_config()
        self.graph = create_rag_graph()

    def invoke(self, query: str, history: str = "") -> Dict[str, Any]:
        """Run the RAG workflow"""
        initial_state = {
            "query": query,
            "history": history,
            "processed_query": None,
            "query_metadata": {},
            "retrieved_documents": [],
            "retrieval_scores": [],
            "context": "",
            "answer": "",
            "sources": "",
            "generation_metadata": {},
            "should_retrieve": False,
            "should_generate": False,
            "error": None,
            "step": "initialized",
        }

        try:
            result = self.graph.invoke(initial_state)
            return result
        except Exception as e:
            return {
                **initial_state,
                "answer": f"Erreur système: {str(e)}",
                "sources": "Erreur système",
                "error": str(e),
                "step": "system_error",
            }

    def stream(self, query: str, history: str = ""):
        """Stream the RAG workflow execution"""
        initial_state = {
            "query": query,
            "history": history,
            "processed_query": None,
            "query_metadata": {},
            "retrieved_documents": [],
            "retrieval_scores": [],
            "context": "",
            "answer": "",
            "sources": "",
            "generation_metadata": {},
            "should_retrieve": False,
            "should_generate": False,
            "error": None,
            "step": "initialized",
        }

        try:
            for step in self.graph.stream(initial_state):
                yield step
        except Exception as e:
            yield {
                "error": {
                    **initial_state,
                    "answer": f"Erreur système: {str(e)}",
                    "sources": "Erreur système",
                    "error": str(e),
                    "step": "system_error",
                }
            }
