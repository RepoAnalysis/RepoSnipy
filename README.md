---
title: RepoSnipy
emoji: 🐍🔫
colorFrom: grey
colorTo: grey
sdk: streamlit
sdk_version: 1.21.0
python_version: 3.11.3
app_file: app.py
pinned: true
license: mit
---
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

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## Acknowledgments

The model and the fine-tuning dataset used:

* [UniXCoder](https://arxiv.org/abs/2203.03850)
* [AdvTest](https://arxiv.org/abs/1909.09436)
