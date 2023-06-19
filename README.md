# RepoSnipy 🐍🔫

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/Lazyhope/RepoSnipy)

Neural search engine for discovering semantically similar Python repositories on GitHub.

## Demo

Searching an indexed repository:

![Search Indexed Repo Demo](assets/search.gif)

## About

RepoSnipy is a neural search engine built with [streamlit](https://github.com/streamlit/streamlit) and [docarray](https://github.com/docarray/docarray). You can query a public Python repository hosted on GitHub and find popular repositories that are semantically similar to it.

It uses the [RepoSim](https://github.com/RepoAnalysis/RepoSim/) pipeline to create embeddings for Python repositories. We have created a [vector dataset](data/index.bin) (stored as docarray index) of over 9700 GitHub Python repositories that has license and over 300 stars by the time of 20th May, 2023.

## Running Locally

Download the repository and install the required packages:

```bash
git clone https://github.com/RepoAnalysis/RepoSnipy
cd RepoSnipy
pip install -r requirements.txt
```

Then run the app on your local machine using:

```bash
streamlit run app.py
```

## Evaluation

The [evaluation script](evaluate.py) finds all combinations of repository pairs in the dataset and calculates the cosine similarity between their embeddings. It also checks if they share at least one topic (except for `python` and `python3`). Then we compare them and use ROC AUC score to evaluate the embeddings performance. The resultant dataframe containing all pairs of cosine similarity and topics similarity can be downloaded from [here](https://huggingface.co/datasets/Lazyhope/RepoSnipy_eval/tree/main), including both code embeddings and docstring embeddings evaluations. The resultant ROC AUC score of code embeddings is around 0.84, and the docstring embeddings is around 0.81.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## Acknowledgments

The model and the fine-tuning dataset used:

* [UniXCoder](https://arxiv.org/abs/2203.03850)
* [AdvTest](https://arxiv.org/abs/1909.09436)
