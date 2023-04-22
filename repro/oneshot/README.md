# Reproduction of "One-Shot Labeling for Automatic Relevnace Estimation"

We provide code for both ["last-metre" and "complete" reproduction](https://dl.acm.org/doi/10.1145/3477495.3531721)
of the paper ["One-Shot Labeling for Automatic Relevnace Estimation"](https://arxiv.org/abs/2302.11266).

## Last-Metre Reproduction

This setting uses our pre-computed relevance estimations to support the tables and figures
in the paper.

**Figure 1:** `figure1.py` generates the precision-recall curves, AP, and F1 for the four 1SLs
explored in the paper. The generated figure is saved to `figure1.pdf`.

**Table 2:** `table2.py` generates the correlation, FPR, and FNR results presented in Table 2.
The latex is printed to stdout.

## Complete Reproduction

In order to conduct a complete reproduction, you'll first need the submitted TREC
runs. These can be downloaded from [this website](https://trec.nist.gov/results.html), once
you request access and agree to the terms. We provide `download_trec_runs.sh` to automate
the downloads and filter them to only the judged documents, once you get the required username
and password (enter these at the top of the script).
