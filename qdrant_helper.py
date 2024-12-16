from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

def init_qdrant_client(qdrant_cluster_url, qdrant_cluster_api_key):
    """
    Creates a Qdrant client for a Qdrant Cloud cluster.

    Args:
        qdrant_cluster_url (string):
            Host URL for our Qdrant Cloud cluster.
        qdrant_cluster_api_key (string):
            API key for our Qdrant Cloud cluster.
    Returns:
        client (QdrantClient):
            Qdrant client.
    """

    client = QdrantClient(
        url=qdrant_cluster_url,
        api_key = qdrant_cluster_api_key
    )
    print("===== Connected to Qdrant =====")
    return client


def get_qdrant_nodes(client, qdrant_collection_name, qdrant_node_limit):
    """
    Retreive nodes from a Qdrant Cloud cluster collection.

    Args:
        client (QdrantClient):
            Qdrant client.
        qdrant_collection_name (string):
            The Qdrant collection we want to retreive nodes from.
        qdrant_node_limit (integer):
            The number of Qdrant nodes we want to retrieve.

    Returns:
        nodes (list):
            List of Qdrant nodes.
    """

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=qdrant_collection_name,
    )
    nodes = vector_store.get_nodes(limit=qdrant_node_limit)
    print(f"===== {len(nodes)} Qdrant Node(s) Retrieved =====")
    return nodes