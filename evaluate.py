import os
from dotenv import load_dotenv
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, DatasetGenerator
from llama_index.llms.openai import OpenAI
from qdrant_helper import init_qdrant_client, get_qdrant_nodes
from rag import get_query_engine, get_response_object, parse_response_object

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EVAL_LLM = OpenAI(model="gpt-3.5-turbo", host="https://api.openai.com/v1/chat/completions", temperature=0.0)

def generate_questions(nodes, num_eval_questions_per_chunk):
    """
    Generates evaluation questions based on our Qdrant nodes.

    Args:
        nodes (list):
            List of Qdrant nodes.
        num_eval_questions_per_chunk (integer):
            The number of questions to generate for each Qdrant node.
    Returns:
        questions (list):
            List of generated questions
    """
    
    question_generator = DatasetGenerator(
        nodes=nodes,
        llm=EVAL_LLM,
        num_questions_per_chunk=num_eval_questions_per_chunk
    )
    questions = question_generator.generate_questions_from_nodes()
    print(f"===== {len(questions)} Evaluation Question(s) Generated =====")
    return questions


def evaluate_response_faithfulness(response):
    """
    Evaluate the faithfulness of a RAG response using an evaluation LLM.

    Args:
        response (Response):
            RAG response object.
    Returns:
        eval_result.passing (bool):
            Boolean evaluation score.
        eval_result.score (float):
            Float evaluation score.
        eval_result.feedback (str):
            String evaluation score.
    """

    evaluator = FaithfulnessEvaluator(llm=EVAL_LLM)
    eval_result = evaluator.evaluate_response(response=response)
    return eval_result.passing, eval_result.score, eval_result.feedback


def evaluate_response_relevancy(query, response):
    """
    Evaluate the relevancy of a RAG response using an evaluation LLM.

    Args:
        query (str):
            The query that the RAG query engine answered.
        response (Response):
            RAG response object.
    Returns:
        eval_result.passing (bool):
            Boolean evaluation score.
        eval_result.score (float):
            Float evaluation score.
        eval_result.feedback (str):
            String evaluation score.
    """

    evaluator = RelevancyEvaluator(llm=EVAL_LLM)
    eval_result = evaluator.evaluate_response(
        query=query,
        response=response
    )
    return eval_result.passing, eval_result.score, eval_result.feedback


def rag_evaluation_pipeline(
    qdrant_cluster_url,
    qdrant_cluster_api_key,
    qdrant_collection_name,
    qdrant_node_limit,
    num_eval_questions_per_chunk,
    sparse_embedding_model,
    rerank_model
): 
    """
    An automated RAG evaluation pipeline that generates questions based on a Qdrant collection,
    answers those questions using our RAG system, and then uses an evaluation LLM to score the responses based
    on faithfulness and relevancy to the generated questions.

    Args:
        qdrant_cluster_url (str):
            Host URL for our Qdrant Cloud cluster.
        qdrant_cluster_api_key (str):
            API key for our Qdrant Cloud cluster.
        qdrant_collection_name (str):
            The Qdrant collection we want to evaluate.
        qdrant_node_limit (integer):
            The number of Qdrant nodes we want to retrieve.
        num_eval_questions_per_chunk (integer):
            The number of questions to generate for each Qdrant node.
        sparse_embedding_model (str):
            The sparse embedding model to use for hybrid search.
        rerank_model (str):
            The rerank model to use for hybrid search.
    Returns:
        results (list):
            List of dictionaries containing evaluation results for each generated question.
    """

    print("===== Evaluation Pipeline Starting... =====")
    client = init_qdrant_client(
        qdrant_cluster_url=qdrant_cluster_url,
        qdrant_cluster_api_key=qdrant_cluster_api_key
    )
    nodes = get_qdrant_nodes(
        client=client,
        qdrant_collection_name=qdrant_collection_name,
        qdrant_node_limit=qdrant_node_limit
    )
    questions = generate_questions(
        nodes=nodes,
        num_eval_questions_per_chunk=num_eval_questions_per_chunk
    )
    query_engine = get_query_engine(
        client=client,
        qdrant_collection_name=qdrant_collection_name,
        sparse_embedding_model=sparse_embedding_model,
        rerank_model=rerank_model,
    )

    results = []

    for question in questions:
        response = get_response_object(
            query_engine=query_engine,
            query=question
        )
        response_text, retrieved_sources = parse_response_object(response=response)
        faithfulness_passing, faithfulness_score, faithfulness_feedback = evaluate_response_faithfulness(response=response)
        relevancy_passing, relevancy_score, relevancy_feedback = evaluate_response_relevancy(query=question, response=response)

        result = {
            "Generated Question": question,
            "Response Text": response_text,
            "Sources": retrieved_sources,
            "Faithfulness Passing": faithfulness_passing,
            "Faithfulness Score": faithfulness_score,
            "Faithfulness Feedback": faithfulness_feedback,
            "Relevancy Passing": relevancy_passing,
            "Relevancy Score": relevancy_score,
            "Relevancy Feedback": relevancy_feedback
        }
        results.append(result)

    print("===== Response Evaluation Finished =====")
    return results


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