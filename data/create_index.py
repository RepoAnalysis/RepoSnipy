# Run this script and specify repos to create the index.bin file for the web app.
from typing import List, Optional

from docarray import BaseDoc, DocList
from docarray.index import InMemoryExactNNIndex
from docarray.typing import TorchTensor
from transformers import pipeline

REPOS = tuple(
    input("Input repository names as owner/name, seperated by comma: ").split(",")
)


class RepoDoc(BaseDoc):
    name: str
    topics: List[str]
    stars: int
    license: str
    code_embedding: Optional[TorchTensor[768]]
    doc_embedding: Optional[TorchTensor[768]]


model = pipeline(
    model="Lazyhope/RepoSim",
    trust_remote_code=True,
    device_map="auto",
)

dl = DocList[RepoDoc]()
repo_dataset = model(REPOS)

for info in repo_dataset:
    dl.append(
        RepoDoc(
            name=info["name"],
            topics=info["topics"],
            stars=info["stars"],
            license=info["license"],
            code_embedding=info["mean_code_embedding"],
            doc_embedding=info["mean_doc_embedding"],
        )
    )

index = InMemoryExactNNIndex[RepoDoc](dl)
index.persist("index.bin")
