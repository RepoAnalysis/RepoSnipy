import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
from docarray import BaseDoc
from docarray.index import InMemoryExactNNIndex
from docarray.typing import TorchTensor
from transformers import pipeline

INDEX_PATH = Path(__file__).parent.joinpath("data/index.bin")


@st.cache_resource(show_spinner="Loading dataset...")
def load_index():
    class RepoDoc(BaseDoc):
        name: str
        topics: List[str]
        stars: int
        license: str
        code_embedding: Optional[TorchTensor[768]]
        doc_embedding: Optional[TorchTensor[768]]

    default_doc = RepoDoc(
        name="",
        topics=[],
        stars=0,
        license="",
        code_embedding=None,
        doc_embedding=None,
    )

    return InMemoryExactNNIndex[RepoDoc](index_file_path=INDEX_PATH), default_doc


@st.cache_resource(show_spinner="Loading RepoSim pipeline...")
def load_model():
    return pipeline(
        model="Lazyhope/RepoSim",
        trust_remote_code=True,
        device_map="auto",
    )


@st.cache_data(show_spinner=False)
def run_model(_model, repo_name, github_token):
    with st.spinner(
        f"Downloading and extracting the {repo_name}, this may take a while..."
    ):
        extracted_infos = _model.preprocess(repo_name, github_token=github_token)

    if not extracted_infos:
        return None

    with st.spinner(f"Generating embeddings for {repo_name}..."):
        repo_info = _model.forward(extracted_infos, st_progress=st.progress(0.0))[0]

    return repo_info


def run_search(index, query, search_field, limit):
    top_matches, scores = index.find(
        query=query, search_field=search_field, limit=limit
    )

    search_results = top_matches.to_dataframe()
    search_results["scores"] = scores

    return search_results


index, default_doc = load_index()
model = load_model()

with st.sidebar:
    st.text_input(
        label="GitHub Token",
        key="github_token",
        type="password",
        placeholder="Paste your GitHub token here",
        help="Consider setting GitHub token to avoid hitting rate limits: https://docs.github.com/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token",
    )

    st.slider(
        label="Search results limit",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        key="search_results_limit",
        help="Limit the number of search results",
    )

    st.multiselect(
        label="Display columns",
        options=["scores", "name", "topics", "stars", "license"],
        default=["scores", "name", "topics", "stars", "license"],
        help="Select columns to display in the search results",
        key="display_columns",
    )


repo_regex = r"^((git@|http(s)?://)?(github\.com)(/|:))?(?P<owner>[\w.-]+)(/)(?P<repo>[\w.-]+?)(\.git)?(/)?$"

st.title("RepoSnipy")

st.text_input(
    "Enter a GitHub repo URL or owner/repo (case-sensitive):",
    value="",
    max_chars=200,
    placeholder="numpy/numpy",
    key="repo_input",
)

st.checkbox(
    label="Add/Update this repo to the index",
    value=False,
    key="update_index",
    help="Encode the latest version of this repo and add/update it to the index",
)


search = st.button("Search")
if search:
    match_res = re.match(repo_regex, st.session_state.repo_input)
    if match_res is not None:
        repo_name = f"{match_res.group('owner')}/{match_res.group('repo')}"

        records = index.filter({"name": {"$eq": repo_name}})
        query_doc = default_doc.copy() if not records else records[0]
        if st.session_state.update_index or not records:
            repo_info = run_model(model, repo_name, st.session_state.github_token)
            if repo_info is None:
                st.error("Repo not found or invalid GitHub token!")
                st.stop()

            # Update document inplace
            query_doc.name = repo_info["name"]
            query_doc.topics = repo_info["topics"]
            query_doc.stars = repo_info["stars"]
            query_doc.license = repo_info["license"]
            query_doc.code_embedding = repo_info["mean_code_embedding"]
            query_doc.doc_embedding = repo_info["mean_doc_embedding"]

        if st.session_state.update_index:
            if not records:
                if not query_doc.license:
                    st.warning(
                        "License is missing in this repo and will not be persisted!"
                    )
                elif (
                    query_doc.code_embedding is None and query_doc.doc_embedding is None
                ):
                    st.warning(
                        "This repo has no function code or docstring extracted and will not be persisted!"
                    )
                else:
                    index.index(query_doc)
                    st.success("Repo added to the index!")
            else:
                st.success("Repo updated in the index!")

            with st.spinner("Persisting the index..."):
                index.persist(file=INDEX_PATH)

        st.session_state["query_doc"] = query_doc
    else:
        st.error("Invalid input!")

if "query_doc" in st.session_state:
    query_doc = st.session_state.query_doc
    limit = st.session_state.search_results_limit
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "name": query_doc.name,
                    "topics": query_doc.topics,
                    "stars": query_doc.stars,
                    "license": query_doc.license,
                }
            ],
        )
    )

    display_columns = st.session_state.display_columns
    code_sim_tab, doc_sim_tab = st.tabs(["Code Similarity", "Docstring Similarity"])

    if query_doc.code_embedding is not None:
        code_sim_res = run_search(index, query_doc, "code_embedding", limit)
        code_sim_tab.dataframe(code_sim_res[display_columns])
    else:
        code_sim_tab.error("No function code was extracted for this repo!")

    if query_doc.doc_embedding is not None:
        doc_sim_res = run_search(index, query_doc, "doc_embedding", limit)
        doc_sim_tab.dataframe(doc_sim_res[display_columns])
    else:
        doc_sim_tab.error("No function docstring was extracted for this repo!")
