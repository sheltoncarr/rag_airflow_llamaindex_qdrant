import pendulum
from airflow.decorators import dag, task
from airflow.models import Variable
from datetime import timedelta
from llama_index.llms.openai import OpenAI

OPENAI_API_KEY = Variable.get("OPENAI_API_KEY")
QDRANT_CONN_ID = "qdrant_default"
QDRANT_COLLECTION_NAME = 'arxiv_papers'
QDRANT_NODE_LIMIT = 2
NUM_EVAL_QUESTIONS_PER_CHUNK = 2
EVAL_LLM = OpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0.0)
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"

default_args = {
    "retries": 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False,
}

@dag(
    dag_id="rag_evaluation_pipeline",
    description="RAG evaluation pipeline",
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["Evaluation"],
    default_args=default_args,
)
def rag_evaluation_pipeline():
    """
    ### Documentation
    An automated RAG evaluation pipeline that generates questions based on a Qdrant collection,
    answers those questions using our RAG system, and then uses an evaluation LLM to score the responses based
    on faithfulness and relevancy to the generated questions.
    """
    
    @task()
    def get_qdrant_nodes_and_generate_questions(
        qdrant_collection_name,
        qdrant_node_limit,
        num_eval_questions_per_chunk,
        openai_api_key
    ):
        """
        Retreive nodes from a Qdrant Cloud cluster collection 
        and generates evaluation questions based on those nodes.

        Args:
            qdrant_collection_name (str):
                The Qdrant collection we want to retreive nodes from.
            qdrant_node_limit (integer):
                The number of Qdrant nodes we want to retrieve.
            num_eval_questions_per_chunk (integer):
                The number of questions to generate for each Qdrant node.
            openai_api_key (str):
                API key for our OpenAI account.

        Returns:
            questions (list):
                List of generated questions.
        """

        from airflow.providers.qdrant.hooks.qdrant import QdrantHook
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from llama_index.core.evaluation import DatasetGenerator
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.core import Settings

        qdrant_hook = QdrantHook(conn_id=QDRANT_CONN_ID)
        print("===== Qdrant Hook Created =====")

        qdrant_client_hook = qdrant_hook.get_conn()
        print("===== Connected to Qdrant =====")

        vector_store = QdrantVectorStore(
            client=qdrant_client_hook,
            collection_name=qdrant_collection_name,
        )
        nodes = vector_store.get_nodes(limit=qdrant_node_limit)
        print(f"===== {len(nodes)} Qdrant Node(s) Retrieved =====")

        Settings.embed_model = OpenAIEmbedding(api_key=openai_api_key, max_retries=3)

        question_generator = DatasetGenerator(
            nodes=nodes,
            llm=EVAL_LLM,
            num_questions_per_chunk=num_eval_questions_per_chunk
        )
        questions = question_generator.generate_questions_from_nodes()
        print(f"===== {len(questions)} Evaluation Question(s) Generated =====")

        return questions

    @task()
    def response_evaluation(
        questions,
        qdrant_collection_name,
        sparse_embedding_model,
        rerank_model,
        openai_api_key,
        sparse_top_k=25,
        top_k_results=20,
        top_n_rerank=5,
        relevancy_score=0.5
    ):
        """
        Retreive nodes from a Qdrant Cloud cluster collection 
        and generates evaluation questions based on those nodes.

        Args:
            questions (list):
                List of generated questions.
            qdrant_collection_name (string):
                The Qdrant collection we want to retreive nodes from.
            sparse_embedding_model (str):
                The sparse embedding model to use for hybrid search.
            rerank_model (str):
                The rerank model to use for hybrid search.
            openai_api_key (str):
                API key for our OpenAI account.
            sparse_top_k (integer):
                Number of sparse top k results to return.
            top_k_results (integer):
                Number of top k results to return.
            top_n_rerank (integer):
                Number of reranked results to return.
            relevancy_score (float):
                Similarity score threshold.
        Returns:
            results (list):
                List of dictionaries containing evaluation results for each generated question.
        """

        from airflow.providers.qdrant.hooks.qdrant import QdrantHook
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
        from llama_index.core.postprocessor import SimilarityPostprocessor, SentenceTransformerRerank
        from llama_index.core import VectorStoreIndex, get_response_synthesizer
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.core import Settings

        Settings.llm = OpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0.0)
        Settings.embed_model = OpenAIEmbedding(api_key=openai_api_key, max_retries=3)

        qdrant_hook = QdrantHook(conn_id=QDRANT_CONN_ID)
        print("===== Qdrant Hook Created =====")

        qdrant_client_hook = qdrant_hook.get_conn()
        print("===== Connected to Qdrant =====")

        vector_store = QdrantVectorStore(
            client=qdrant_client_hook,
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
        print("===== Response Synthesizer Created =====")

        node_postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=relevancy_score),
            SentenceTransformerRerank(model=rerank_model, top_n=top_n_rerank),
        ]
        print("===== Node Postprocessors Created =====")

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
        )
        print("===== Query Engine Created =====")

        results = []

        for question in questions:
            response = query_engine.query(question)
            for node in response.source_nodes:
                node.score = float(node.score)
            response_text = response.response
            retrieved_sources = [node.dict() for node in response.source_nodes]
            faithfulness_evaluator = FaithfulnessEvaluator()
            faithfulness_eval_result = faithfulness_evaluator.evaluate_response(response=response)
            relevancy_evaluator = RelevancyEvaluator()
            relevancy_eval_result = relevancy_evaluator.evaluate_response(query=question, response=response)

            result = {
                "Generated Question": question,
                "Response Text": response_text,
                "Sources": retrieved_sources,
                "Faithfulness Passing": faithfulness_eval_result.passing,
                "Faithfulness Score": faithfulness_eval_result.score,
                "Faithfulness Feedback": faithfulness_eval_result.feedback,
                "Relevancy Passing": relevancy_eval_result.passing,
                "Relevancy Score": relevancy_eval_result.score,
                "Relevancy Feedback": relevancy_eval_result.feedback
            }
            results.append(result)

        print("===== Response Evaluation Finished =====")
        return results

    @task()
    def evaluation_summary_metrics(evaluation_results):
        """
        Calculate the average percentage values for faithfulness and relevancy scores 
        from the evaluation results.

        Args:
            evaluation_results (list):
                List of dictionaries containing faithfulness and relevancy scores.

        Returns:
            dict:
                Dictionary containing the average percentages of faithfulness and relevancy scores.
        """

        total_faithfulness_score = 0
        total_relevancy_score = 0
        count = 0
        avg_faithfulness_percentage = None
        avg_relevancy_percentage = None

        for result in evaluation_results:
            total_faithfulness_score += result.get("Faithfulness Score", 0)
            total_relevancy_score += result.get("Relevancy Score", 0)
            count += 1

        if count != 0:  # Avoid division by zero
            avg_faithfulness_percentage = round((total_faithfulness_score / count) * 100, 2)
            avg_relevancy_percentage = round((total_relevancy_score / count) * 100, 2)

        return {
            "Average Faithfulness Percentage": avg_faithfulness_percentage,
            "Average Relevancy Percentage": avg_relevancy_percentage
        }    


    questions = get_qdrant_nodes_and_generate_questions(
        qdrant_collection_name=QDRANT_COLLECTION_NAME,
        qdrant_node_limit=QDRANT_NODE_LIMIT,
        num_eval_questions_per_chunk=NUM_EVAL_QUESTIONS_PER_CHUNK,
        openai_api_key=OPENAI_API_KEY
    )

    evaluation_results = response_evaluation(
        questions=questions,
        qdrant_collection_name=QDRANT_COLLECTION_NAME,
        sparse_embedding_model=SPARSE_EMBEDDING_MODEL,
        rerank_model=RERANK_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    avg_metric_percentages = evaluation_summary_metrics(evaluation_results=evaluation_results)

rag_evaluation_pipeline()