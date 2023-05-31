import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
from docarray import BaseDoc, DocList
from docarray.typing import TorchTensor
from docarray.utils.find import find
from transformers import pipeline

DATASET_PATH = Path(__file__).parent.joinpath("data/index.bin")


@st.cache_resource(show_spinner="Loading dataset...")
def load_index():
    class RepoDoc(BaseDoc):
        name: str
        topics: list  # TODO: List[str]
        stars: int
        license: str
        code_embedding: Optional[TorchTensor[768]]
        doc_embedding: Optional[TorchTensor[768]]

    return DocList[RepoDoc].load_binary(DATASET_PATH)


@st.cache_resource(show_spinner="Loading RepoSim pipeline...")
def load_model():
    return pipeline(
        model="Lazyhope/RepoSim",
        trust_remote_code=True,
        device_map="auto",
        use_auth_token=st.secrets.hf_token,  # TODO: delete this line when the pipeline is public
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
    top_matches, scores = find(
        index=index, query=query, search_field=search_field, limit=limit
    )

    search_results = top_matches.to_dataframe()
    search_results["scores"] = scores

    return search_results


index = load_index()
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
    )

    st.multiselect(
        label="Display columns",
        options=["scores", "name", "topics", "stars", "license"],
        default=["scores", "name", "topics"],
        key="display_columns",
    )


repo_regex = r"^((git@|http(s)?://)?(github\.com)(/|:))?(?P<owner>[\w.-]+)(/)(?P<repo>[\w.-]+?)(\.git)?(/)?$"

st.title("RepoSnipy")

st.text_input(
    "Enter a GitHub repo URL or owner/repo (case-sensitive):",
    value="",
    max_chars=200,
    placeholder="huggingface/transformers",
    key="repo_input",
)

st.checkbox(
    label="Add/Update this repo to the index",
    value=False,
    key="update_index",
    help="Update index by generating embeddings for the latest version of this repo",
)


search = st.button("Search")
if search:
    match_res = re.match(repo_regex, st.session_state.repo_input)
    if match_res is not None:
        repo_name = f"{match_res.group('owner')}/{match_res.group('repo')}"

        doc_index = -1
        update_index = st.session_state.update_index
        try:
            doc_index = index.name.index(repo_name)
            assert update_index is False

            repo_doc = index[doc_index]
        except (ValueError, AssertionError):
            repo_info = run_model(model, repo_name, st.session_state.github_token)
            if repo_info is None:
                st.error("Repo not found or invalid GitHub token!")
                st.stop()

            repo_doc = index.doc_type(
                name=repo_info["name"],
                topics=repo_info["topics"],
                stars=repo_info["stars"],
                license=repo_info["license"],
                code_embedding=repo_info["mean_code_embedding"],
                doc_embedding=repo_info["mean_doc_embedding"],
            )

        if update_index:
            if not repo_doc.license:
                st.warning("License is missing in this repo!")

            if doc_index == -1:
                index.append(repo_doc)
                st.success("Repo added to the index!")
            else:
                index[doc_index] = repo_doc
                st.success("Repo updated in the index!")

        st.session_state["query"] = repo_doc
    else:
        st.error("Invalid input!")

if "query" in st.session_state:
    query = st.session_state.query

    limit = st.session_state.search_results_limit
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "name": query.name,
                    "topics": query.topics,
                    "stars": query.stars,
                    "license": query.license,
                }
            ],
        )
    )

    display_columns = st.session_state.display_columns
    code_sim_tab, doc_sim_tab = st.tabs(["Code Similarity", "Docstring Similarity"])

    if query.code_embedding is not None:
        code_sim_res = run_search(index, query, "code_embedding", limit)
        code_sim_tab.dataframe(code_sim_res[display_columns])
    else:
        code_sim_tab.error("No code was extracted for this repo!")

    if query.doc_embedding is not None:
        doc_sim_res = run_search(index, query, "doc_embedding", limit)
        doc_sim_tab.dataframe(doc_sim_res[display_columns])
    else:
        doc_sim_tab.error("No docstring was extracted for this repo!")
