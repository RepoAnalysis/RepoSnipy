from itertools import combinations
from math import comb
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
from docarray import BaseDoc
from docarray.index import InMemoryExactNNIndex
from docarray.typing import TorchTensor
from sklearn.metrics import roc_curve, auc
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
# Filter out repos without code embedding and select only relevant columns
code_df = df[df["code_embedding"].notna()][["name", "topics", "code_embedding"]]


def has_same_topic(topics1, topics2):
    # Find shared topics other than "python" and "python3"
    intersection = set(topics1) & set(topics2) - {"python", "python3"}
    return len(intersection) > 0


# Find code similarity and shared topics between all pairs of repos
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
                0.0,  # zero out negative similarities
                cosine_similarity(
                    row1.code_embedding, row2.code_embedding, dim=0
                ).item(),
            ),
        }
    )

similarity_df = pd.DataFrame(rows_list)
similarity_df.to_csv("code_eval_res.csv", index=False)

# Plot ROC curve
y_true, y_score = similarity_df["has_same_topic"], similarity_df["code_similarity"]
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")  # Diagonal line representing random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()
