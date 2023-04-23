# autoqrels

`autoqrels` is a tool for automatically inferring query relevance assessments (qrels).

Currently, it supports the one-shot labeling approach (1SL) presented in *[MacAvaney and
Soldaini, One-Shot Labeling for Automatic Relevance Estimation, SIGIR 2023](https://arxiv.org/pdf/2302.11266.pdf)*.

This package adheres to the [`ir-measures`](https://ir-measur.es/) API, which means it can
be directly used by various tools, such as [PyTerrier](https://pyterrier.readthedocs.io/).

## Getting started

You can install `autoqrels` using pip:

```bash
pip install autoqrels
```

You can also work with the repository locally:

```bash
git clone https://github.com/seanmacavaney/autoqrels.git
cd autoqrels
python setup.py develop
```

## API

The primary interface in `autoqrels` is `autoqrels.Labeler`. A `Labeler` exposes a
method, `infer_qrels(run, qrels)`, which returns a new set of qrels that covers the
provided run:

 - `run` is a Pandas DataFrame with the columns `query_id` (str), `doc_id` (str), and `score` (float)
 - `qrels` is a Pandas DataFrame with the columns `query_id` (str), `doc_id` (str), and `relevance` (int)
 - The return value is a Pandas DataFrame with the columns `query_id` (str), `doc_id` (str), and `relevance` (float)

`Labeler`s also expose several measure definitions compatible with `ir_measures`:
[`labeler.SDCG@k`](https://ir-measur.es/en/latest/measures.html#sdcg),
[`labeler.RBP(p=persistence)`](https://ir-measur.es/en/latest/measures.html#rbp),
[`labeler.P@k`](https://ir-measur.es/en/latest/measures.html#p).
These measures can be used to calculate the corresponding effectivness, with the
addition of the labeler's inferred qrels. See the [ir-measures documentation](https://ir-measur.es/)
for more details.

We'll now explore the available `Labeler` implementations.

### `autoqrels.oneshot`: 1SL (One-shot Labeling)

**Reproduction: See repro instructions in [`repro/oneshot`](repro/oneshot).**

One-shot labelers work over a single known relevant document per query. An error
is raised if multiple relevant documents are provided.

Example:

```python
import autoqrels
import ir_datasets
dataset = ir_datasets.load('msmarco-passage/trec-dl-2019')
duot5 = autoqrels.oneshot.DuoT5(dataset=dataset, cache_path='data/duot5.cache.json.gz')
# measures:
duot5.SDCG@10
duot5.P@10
duot5.RBP
```

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{autoqrels,
  author = {MacAvaney, Sean and Soldaini, Luca},
  title = {One-Shot Labeling for Automatic Relevance Estimation},
  booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {2023},
  url = {https://arxiv.org/abs/2302.11266}
}
```
