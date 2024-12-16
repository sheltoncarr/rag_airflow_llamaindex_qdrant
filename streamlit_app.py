import os
import streamlit as st
from qdrant_helper import init_qdrant_client
from rag import get_query_engine, get_response_object, parse_response_object, parse_retrieved_source
from evaluate import rag_evaluation_pipeline, evaluation_summary_metrics
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
QDRANT_CLUSTER_API_KEY = os.environ["QDRANT_CLUSTER_API_KEY"]
QDRANT_CLUSTER_URL = os.environ["QDRANT_CLUSTER_URL"]
QDRANT_COLLECTION_NAME = "arxiv_papers"
QDRANT_NODE_LIMIT = 2
NUM_EVAL_QUESTIONS_PER_CHUNK = 2
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"


def main():

    # Check if the Qdrant client and query engine have already been initialized
    if 'client' not in st.session_state:
        st.session_state.client = init_qdrant_client(
            qdrant_cluster_url=QDRANT_CLUSTER_URL,
            qdrant_cluster_api_key=QDRANT_CLUSTER_API_KEY
        )

    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = get_query_engine(
            client=st.session_state.client,
            qdrant_collection_name=QDRANT_COLLECTION_NAME,
            sparse_embedding_model=SPARSE_EMBEDDING_MODEL,
            rerank_model=RERANK_MODEL,
        )

    st.set_page_config(layout="wide", page_title="RAG", page_icon=":mag:")

    page = st.sidebar.selectbox("Select a Page", ["RAG Search", "RAG Response Evaluation"])

    # Page for RAG Search
    if page == "RAG Search":
        st.title("RAG with Apache Airflow, LlamaIndex, and Qdrant")
        st.markdown("### 🔍 **Search**")

        user_query = st.text_input(
            "Enter your query to retrieve the most relevant information based on our RAG model and Qdrant knowledge base", 
            placeholder="Type your query here...",
        )

        if st.button("Search"):
            if user_query.strip():
                with st.spinner("Searching for relevant information..."):
                    try:
                        response = get_response_object(
                            query_engine=st.session_state.query_engine,
                            query=user_query
                        )
                        response_text, retrieved_sources = parse_response_object(response)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        response_text, retrieved_sources = 'Empty Response', []

                if response_text != 'Empty Response':
                    answer_col, sources_col = st.columns([1, 2])

                    with answer_col:
                        st.markdown("### 🤖 **Answer**")
                        st.markdown(
                            f"<div style='background-color: #222222; padding: 10px; border-radius: 5px; color: #f0f4f8;'>{response_text}</div>",
                            unsafe_allow_html=True
                        )

                    with sources_col:
                        st.markdown("### 📚 **Sources**")
                        for source in retrieved_sources:
                            parsed_source = parse_retrieved_source(source)

                            st.markdown(f"📜 **Paper Title:** {parsed_source.get('title')}")
                            st.markdown(f"🧑 **Authors:** {parsed_source.get('authors')}")
                            st.markdown(f"📅 **Published:** {parsed_source.get('published')}")
                            st.markdown(f"📅 **Updated:** {parsed_source.get('updated')}")
                            st.markdown(f"🔗 **PDF URL:** {parsed_source.get('pdf_url')}")
                            st.markdown(f"🎯 **Relevancy Score:** {parsed_source.get('score'):.3f}")
                            st.markdown(f"📝 **Excerpt:** {parsed_source.get('text')}...")
                            st.divider()

                        print("===== Query Response Generated =====")
                else:
                    st.write("I could not find any sources relevant to your question.")
            else:
                st.warning("Please enter a query to search.")

    # Page for Evaluation
    elif page == "RAG Response Evaluation":
        st.title("RAG Response Evaluation")

        st.write("Click the button below to start a fully automated RAG response evaluation flow over our Qdrant knowledge base")

        if st.button("Evaluate"):
            with st.spinner("Evaluating our RAG system, this may take a second..."):
                evaluation_results = rag_evaluation_pipeline(
                    qdrant_cluster_url=QDRANT_CLUSTER_URL,
                    qdrant_cluster_api_key=QDRANT_CLUSTER_API_KEY,
                    qdrant_collection_name=QDRANT_COLLECTION_NAME,
                    qdrant_node_limit=QDRANT_NODE_LIMIT,
                    num_eval_questions_per_chunk=NUM_EVAL_QUESTIONS_PER_CHUNK,
                    sparse_embedding_model=SPARSE_EMBEDDING_MODEL,
                    rerank_model=RERANK_MODEL
                )

                avg_metric_percentages = evaluation_summary_metrics(evaluation_results=evaluation_results)
                
                st.markdown("## 📊 **Evaluation Results**")
                st.divider()

                st.markdown("### Summary Metrics")
                st.markdown(f"Average Faithfulness Percentage (%): {avg_metric_percentages.get('Average Faithfulness Percentage')}")
                st.markdown(f"Average Relevancy Percentage (%): {avg_metric_percentages.get('Average Relevancy Percentage')}")
                st.divider()

                for i, result in enumerate(evaluation_results):
                    st.markdown(f"### Evaluation Result #{i+1}")
                    st.markdown(f"**Generated Question:** {result['Generated Question']}")
                    st.markdown(f"**Response:** {result['Response Text']}")
                    
                    for j, source in enumerate(result["Sources"]):
                        st.markdown(f"#### Source #{j+1}:")
                        parsed_source = parse_retrieved_source(source)

                        st.markdown(f"📜 **Paper Title:** {parsed_source.get('title')}")
                        st.markdown(f"🧑 **Authors:** {parsed_source.get('authors')}")
                        st.markdown(f"📅 **Published:** {parsed_source.get('published')}")
                        st.markdown(f"📅 **Updated:** {parsed_source.get('updated')}")
                        st.markdown(f"🔗 **PDF URL:** {parsed_source.get('pdf_url')}")
                        st.markdown(f"🎯 **Relevancy Score:** {parsed_source.get('score'):.3f}")
                        st.markdown(f"📝 **Excerpt:** {parsed_source.get('text')}...")

                    st.markdown(f"#### Evaluation Metrics:")
                    st.markdown(f"**Faithfulness Passing:** {result['Faithfulness Passing']}")
                    st.markdown(f"**Faithfulness Score:** {result['Faithfulness Score']}")
                    st.markdown(f"**Faithfulness Feedback:** {result['Faithfulness Feedback']}")
                    st.markdown(f"**Relevancy Passing:** {result['Relevancy Passing']}")
                    st.markdown(f"**Relevancy Score:** {result['Relevancy Score']}")
                    st.markdown(f"**Relevancy Feedback:** {result['Relevancy Feedback']}")

                    st.divider()

if __name__ == "__main__":
    main()