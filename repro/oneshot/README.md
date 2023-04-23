# Reproduction of "One-Shot Labeling for Automatic Relevnace Estimation"

We provide code for both ["last-metre" and "complete" reproduction](https://dl.acm.org/doi/10.1145/3477495.3531721)
of the paper ["One-Shot Labeling for Automatic Relevnace Estimation"](https://arxiv.org/abs/2302.11266).

## Data

You'll need the submitted TREC runs to reproduce our results. These can be downloaded
from [the TREC website](https://trec.nist.gov/results.html), once you request access and agree
to the terms. We provide [`download_trec_runs.sh`](download_trec_runs.sh) to automate the
downloads and filter them to only the judged documents, once you get the required username
and password (enter these at the top of the script).

## Last-Metre Reproduction

This setting uses our pre-computed relevance estimations to support the tables and figures
in the paper. These are provided in the cache files in this directory: `[method].cache.json.gz`.
The `autoqrels` package uses these cached files `cache_path` is provided to the consstructor.

**Figure 1:** [`figure1.py`](figure1.py) generates the precision-recall curves, AP, and F1 for
the four 1SLs explored in the paper. The generated figure is saved to [`figure1.pdf`](figure1.pdf).

**Table 2:** [`table2.py`](table2.py) generates the correlation, FPR, and FNR results presented
in Table 2. The latex is printed to stdout.

## Complete Reproduction

Coming soon!
