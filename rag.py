import os
from llama_index.core import VectorStoreIndex, Settings, get_response_synthesizer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, SentenceTransformerRerank

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_query_engine(
    client,
    qdrant_collection_name,
    sparse_embedding_model,
    rerank_model,
    sparse_top_k=25,
    top_k_results=20,
    top_n_rerank=5,
    relevancy_score=0.5
):
    """
    Create a query engine for the Qdrant Cloud cluster collection.

    Args:
        client (QdrantClient):
            Qdrant client.
        qdrant_collection_name (str):
            The Qdrant collection we want to retreive nodes from.
        sparse_embedding_model (str):
            The sparse embedding model to use for hybrid search.
        rerank_model (str):
            The rerank model to use for hybrid search.
        sparse_top_k (integer):
            Number of sparse top k results to return.
        top_k_results (integer):
            Number of top k results to return.
        top_n_rerank (integer):
            Number of reranked results to return.
        relevancy_score (float):
            Similarity score threshold.
    Returns:
        query_engine (RetrieverQueryEngine):
            LlamaIndex query engine.
    """

    Settings.embed_model = OpenAIEmbedding()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=qdrant_collection_name,
        enable_hybrid=True,
        fastembed_sparse_model=sparse_embedding_model,
        batch_size=1
    )
    print("===== Vector Store Created =====")

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )
    print("===== Index Created =====")

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k_results,
        sparse_top_k=sparse_top_k,
        vector_store_query_mode="hybrid"
    )
    print("===== Retreiver Created =====")

    response_synthesizer = get_response_synthesizer()

    node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=relevancy_score),
        SentenceTransformerRerank(model=rerank_model, top_n=top_n_rerank),
    ]

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors,
    )
    print("===== Query Engine Created =====")
    return query_engine


def get_response_object(query_engine, query: str) -> str:
    """
    Get a response to a query using the RAG query engine.

    Args:
        query_engine (RetrieverQueryEngine):
            LlamaIndex query engine.
        query (str):
            The query that the RAG query engine will answer.
    Returns:
        response (Response):
            RAG response object
    """

    response = query_engine.query(query)
    for node in response.source_nodes:
        node.score = float(node.score)
    return response


def parse_response_object(response):
    """
    Parse a response from the RAG query engine into a more structured format.

    Args:
        response (Response):
            RAG response object.
    Returns:
        response_text (str):
            The RAG text response.
        retrieved_sources (list):
            The retrieved sources for the RAG response. 
    """

    response_text = response.response
    retrieved_sources = [node.dict() for node in response.source_nodes]
    return response_text, retrieved_sources


def parse_retrieved_source(retrieved_source):
    """
    Parse a retrieved source into a more structured format.

    Args:
        retrieved_source (dict):
            Retreived source for a RAG response.
    Returns:
        result (dict):
            Dictionary of parsed fields for the retrieved source.
    """

    result = {
        "title" : retrieved_source["node"]["metadata"].get("title", "No title available"),
        "authors" : ', '.join(retrieved_source["node"]["metadata"].get("authors", [])),
        "published" : retrieved_source["node"]["metadata"].get("published", "Unknown date"),
        "updated" : retrieved_source["node"]["metadata"].get("updated", "Unknown date"),
        "pdf_url" : retrieved_source["node"]["metadata"].get("pdf_url", "No URL available"),
        "score" : retrieved_source.get("score", 0),
        "text" : retrieved_source["node"].get("text", "No excerpt available")[:1000]
    }

    return result