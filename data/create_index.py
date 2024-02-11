# Run this script and specify repos to create the index.bin file for the web app.
import os
import numpy as np
from typing import List, Optional
from docarray import BaseDoc, DocList
from docarray.index import InMemoryExactNNIndex
from docarray.typing import NdArray
from transformers import pipeline
from tqdm.auto import tqdm

# For testing index
from pathlib import Path


# REPOS = tuple(
#     input("Input repository names as owner/name, seperated by comma: ").split(",")
# )


class RepoDoc(BaseDoc):
    name: str
    topics: List[str]
    stars: int
    license: str
    code_embedding: Optional[NdArray[768]]
    doc_embedding: Optional[NdArray[768]]
    readme_embedding: Optional[NdArray[768]]
    requirement_embedding: Optional[NdArray[768]]
    repository_embedding: Optional[NdArray[3072]]


def get_model():
    # model_path = ".\RepoSim4Py"
    model_path = "Henry65/RepoSim4Py"
    return pipeline(
        model=model_path,
        trust_remote_code=True,
        device_map="auto",
        github_token=os.environ.get("GITHUB_TOKEN")
    )


def get_repositories():
    with open("repositories.txt", "r") as file:
        repositories = file.read().splitlines()
    return repositories


def get_sub_repositories_list(repositories):
    target_sub_length = len(repositories) // 100
    sub_repositories_list = []
    start_index = 0

    while start_index < len(repositories):
        current_length = min(target_sub_length, len(repositories) - start_index)
        current_sub_repositories = repositories[start_index: start_index + current_length]
        sub_repositories_list.append(current_sub_repositories)
        start_index += current_length

    print(f"We totally have {len(sub_repositories_list)} sub repositories")
    return sub_repositories_list, target_sub_length


def create_index_by_sub(model, sub_repositories_list, target_sub_length):
    for i, sub_repositories in tqdm(enumerate(sub_repositories_list)):
        tqdm.write(f"Processing sub repositories {i}")
        dl = DocList[RepoDoc]()
        repo_dataset = model(tuple(sub_repositories))

        for info in repo_dataset:
            dl.append(
                RepoDoc(
                    name=info["name"],
                    topics=info["topics"],
                    stars=info["stars"],
                    license=info["license"],
                    code_embedding=info["mean_code_embedding"].reshape(-1),
                    doc_embedding=info["mean_doc_embedding"].reshape(-1),
                    readme_embedding=info["mean_readme_embedding"].reshape(-1),
                    requirement_embedding=info["mean_requirement_embedding"].reshape(-1),
                    repository_embedding=info["mean_repo_embedding"].reshape(-1)
                )
            )

        index = InMemoryExactNNIndex[RepoDoc](dl)
        index.persist(f"index{i}_{i * target_sub_length}.bin")


def merge_index(target_sub_length):
    INDEX_PATH = Path(__file__).parent.joinpath("index0_0.bin")
    index = InMemoryExactNNIndex[RepoDoc](index_file_path=INDEX_PATH)
    docs = index._docs

    file_name_list = [f"index{i}_{i * target_sub_length}.bin" for i in range(1, 101)]
    for file_name in file_name_list:
        INDEX_PATH_TMP = Path(__file__).parent.joinpath(file_name)
        index_tmp = InMemoryExactNNIndex[RepoDoc](index_file_path=INDEX_PATH_TMP)
        docs_tmp = index_tmp._docs
        docs.extend(docs_tmp)

    index = InMemoryExactNNIndex[RepoDoc](docs)
    index.persist(f"index.bin")


def find_exception_repositories(file_name):
    INDEX_PATH = Path(__file__).parent.joinpath(file_name)
    index = InMemoryExactNNIndex[RepoDoc](index_file_path=INDEX_PATH)
    docs = index._docs
    docs_set = set([doc.name for doc in docs])
    repo_set = set(get_repositories())
    diff_set = repo_set - docs_set
    with open("exception_repositories.txt", "w") as f:
        for repo_name in diff_set:
            print(repo_name, file=f)

def remove_zero_vectors(file_name1, file_name2):
    INDEX_PATH = Path(__file__).parent.joinpath(file_name1)
    index = InMemoryExactNNIndex[RepoDoc](index_file_path=INDEX_PATH)
    docs = index._docs
    for doc in docs:
        if np.all(doc.code_embedding == 0):
            doc.code_embedding = None
        if np.all(doc.doc_embedding == 0):
            doc.doc_embedding = None
        if np.all(doc.requirement_embedding == 0):
            doc.requirement_embedding = None
        if np.all(doc.readme_embedding == 0):
            doc.readme_embedding = None
        if np.all(doc.repository_embedding == 0):
            doc.repository_embedding = None

    index = InMemoryExactNNIndex[RepoDoc](docs)
    index.persist(file_name2)

if __name__ == "__main__":
    # Creating index
    model = get_model()
    repositories = get_repositories()
    sub_repositories_list, target_sub_length = get_sub_repositories_list(repositories)
    create_index_by_sub(model, sub_repositories_list, target_sub_length)

    # Merging index
    merge_index(target_sub_length)

    # Finding exception repositories
    find_exception_repositories("index.bin")

    # Removing numpy zero arrays in index
    remove_zero_vectors("index.bin", "index_reduced.bin")
