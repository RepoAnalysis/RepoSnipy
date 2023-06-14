from itertools import combinations
from math import comb
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from docarray import BaseDoc
from docarray.index import InMemoryExactNNIndex
from docarray.typing import TorchTensor
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from torch.nn.functional import cosine_similarity
from tqdm.auto import tqdm

INDEX_PATH = Path(__file__).parent.joinpath("data/index.bin")


class RepoDoc(BaseDoc):
    name: str
    topics: List[str]
    stars: int
    license: str
    code_embedding: Optional[TorchTensor[768]]
    doc_embedding: Optional[TorchTensor[768]]


index = InMemoryExactNNIndex[RepoDoc](index_file_path=INDEX_PATH)
df = index._docs.to_dataframe()
code_df = df[df["code_embedding"].notna()][["name", "topics", "code_embedding"]]

all_topics = set()
for topics in code_df["topics"]:
    all_topics.update(topics)
all_topics = sorted(list(all_topics))

topic_model = SentenceTransformer("all-MiniLM-L6-v2")
word_embeddings = torch.from_numpy(topic_model.encode(all_topics))
topic_embeddings = dict(zip(all_topics, word_embeddings))

memory = {}


def has_same_topic(topics1, topics2):
    intersection = set(topics1) & set(topics2) - {"python", "python3"}
    return len(intersection) > 0


rows_list = []
for row1, row2 in tqdm(
    combinations(code_df.itertuples(), 2), total=comb(len(code_df), 2)
):
    rows_list.append(
        {
            "repo1": row1.name,
            "repo2": row2.name,
            "has_same_topic": has_same_topic(row1.topics, row2.topics),
            "code_similarity": max(
                0.0,
                cosine_similarity(
                    row1.code_embedding, row2.code_embedding, dim=0
                ).item(),
            ),
        }
    )

similarity_df = pd.DataFrame(rows_list)
similarity_df.to_csv("eval_res.csv", index=False)

print(roc_auc_score(similarity_df["has_same_topic"], similarity_df["code_similarity"]))
