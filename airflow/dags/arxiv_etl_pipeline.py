import pendulum
from airflow.decorators import dag, task
from airflow.models import Variable
from datetime import timedelta
from qdrant_client import models

OPENAI_API_KEY = Variable.get("OPENAI_API_KEY")
QDRANT_CONN_ID = "qdrant_default"
QDRANT_COLLECTION_NAME = 'arxiv_papers'
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
EMBEDDING_DIMENSION = 1536
SIMILARITY_METRIC = models.Distance.COSINE
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 32
QUERY = "machine learning"
MAX_API_RESULTS = 5


default_args = {
    "retries": 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False,
}

@dag(
    dag_id="arxiv_etl_pipeline",
    description="Extract arXiv data and upload to Qdrant",
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["ETL"],
    default_args=default_args,
)
def arxiv_etl_pipeline():
    """
    ### Documentation
    An ETL pipeline that will pull the latest relevant papers from arXiv,
    chunk them, embed them, and upload them to Qdrant. This pipeline can be modified to accomodate
    other data sources as well, based on your specific use case.
    """
    
    @task()
    def fetch_arxiv_papers(query, max_api_results):
        """
        Fetch and parse the latest relevant papers from arXiv.

        Args:
            query (str):
                The query for searching arXiv papers.
            max_api_results (integer):
                The maximum number of arXiv papers to return.
        Returns:
            papers (list):
                List of dictionaries containing data and metadata for each paper.
        """

        import arxiv
        import pymupdf
        import requests
        from io import BytesIO

        client = arxiv.Client()

        search = arxiv.Search(
            query=query,
            max_results=max_api_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        papers = []
        
        for result in client.results(search):
            paper = {
                "id": result.entry_id,
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published.isoformat(),
                "updated": result.updated.isoformat(),
                "pdf_url": result.pdf_url
            }

            response = requests.get(paper["pdf_url"])
            if response.status_code != 200:
                raise Exception(f"Failed to download PDF: {response.status_code}")
            
            pdf_data = BytesIO(response.content)
            
            doc = pymupdf.open(stream=pdf_data, filetype="pdf")
            
            # Extract text from all pages
            full_text = ""
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                full_text += page.get_text()
            paper["full_text"] = full_text

            papers.append(paper)
        
        print("===== API Extraction Complete =====")
        print(f"===== Number of papers extracted: {len(papers)} =====")

        return papers


    @task
    def init_qdrant_collection(
        qdrant_collecton_name,
        embedding_dimension,
        similarity_metric
    ):
        """
        Create a new Qdrant collection if it does not already exist.

        Args:
            qdrant_collection_name (str):
                The Qdrant collection we want to retreive nodes from.
            embedding_dimension (integer):
                The dimension of our dense embedding model.
            similarity_metric (models.Distance):
                The distance metric for our dense embedding search.
        Returns:
            None
        """

        from airflow.providers.qdrant.hooks.qdrant import QdrantHook
            
        qdrant_hook = QdrantHook(conn_id=QDRANT_CONN_ID)
        print("===== Qdrant Hook Created =====")

        qdrant_client_hook = qdrant_hook.get_conn()
        print("===== Connected to Qdrant =====")

        if not qdrant_client_hook.collection_exists(qdrant_collecton_name):
            qdrant_client_hook.create_collection(
                qdrant_collecton_name,
                vectors_config={
                    "text-dense": models.VectorParams(
                        size=embedding_dimension, distance=similarity_metric
                    )
                },
                sparse_vectors_config={
                    "text-sparse": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                }      
            )
            print(f"===== Qdrant Collection \"{qdrant_collecton_name}\" Created =====")
        else:
            print(f"===== Qdrant Collection \"{qdrant_collecton_name}\" Already Exists =====")
            collection_point_count = qdrant_client_hook.count(qdrant_collecton_name)
            print(f"===== Number of points in \"{qdrant_collecton_name}\": {collection_point_count} =====")


    @task()
    def upload_arxiv_to_qdrant(
        papers,
        openai_api_key,
        chunk_size,
        chunk_overlap,
        qdrant_collection_name,
        sparse_embedding_model
    ):
        """
        Chunk, embed, and upload arXiv papers to Qdrant.

        Args:
            papers (list):
                List of dictionaries containing data and metadata for each paper.  
            openai_api_key (str):
                API key for our OpenAI account.
            chunk_size (integer):
                Number of tokens in each each chunk.
            chunk_overlap (integer):
                Number of tokens overlapping between chunks.
            qdrant_collection_name (str):
                The Qdrant collection we want to upload documents to.  
            sparse_embedding_model (str):
                The sparse embedding model to use for hybrid search.        
        Returns:
            None
        """

        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.core import Settings, VectorStoreIndex
        from llama_index.core.node_parser import SimpleNodeParser
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from llama_index.core.schema import Document
        from airflow.providers.qdrant.hooks.qdrant import QdrantHook
            
        qdrant_hook = QdrantHook(conn_id=QDRANT_CONN_ID)
        print("===== Qdrant Hook Created =====")

        qdrant_client_hook = qdrant_hook.get_conn()
        print("===== Connected to Qdrant =====")

        Settings.embed_model = OpenAIEmbedding(api_key=openai_api_key, max_retries=3)
        Settings.text_splitter = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print("===== Settings Created =====")

        vector_store = QdrantVectorStore(
            client=qdrant_client_hook,
            collection_name=qdrant_collection_name,
            enable_hybrid=True,
            fastembed_sparse_model=sparse_embedding_model,
            batch_size=1
        )
        print("===== Vector Store Created =====")

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
        )
        print("===== Index Created =====")

        upload_count = 0

        for paper in papers:
            document = Document(
                id_ = paper["id"],
                text = paper["full_text"],
                metadata = {
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "summary": paper["summary"],
                    "published": paper["published"],
                    "updated": paper["updated"],
                    "pdf_url": paper["pdf_url"]
                }
            )

            index.update_ref_doc(document)

            print(f"===== Inserted \"{paper['title']}\" into Qdrant =====")
            upload_count += 1
        
        print(f"===== Number of papers uploaded to Qdrant: {upload_count} =====")

        collection_point_count = qdrant_client_hook.count(qdrant_collection_name)
        print(f"===== Number of points in \"{qdrant_collection_name}\": {collection_point_count} =====")

        print(f"===== Qdrant Upload Complete =====")

    papers = fetch_arxiv_papers(query=QUERY, max_api_results=MAX_API_RESULTS)

    init_collection_task = init_qdrant_collection(
        qdrant_collecton_name=QDRANT_COLLECTION_NAME,
        embedding_dimension=EMBEDDING_DIMENSION,
        similarity_metric=SIMILARITY_METRIC
    )

    if papers:
        upload_task = upload_arxiv_to_qdrant(
            papers=papers,
            openai_api_key=OPENAI_API_KEY,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            qdrant_collection_name=QDRANT_COLLECTION_NAME,
            sparse_embedding_model=SPARSE_EMBEDDING_MODEL
        )
    
    [papers, init_collection_task] >> upload_task

arxiv_etl_pipeline()