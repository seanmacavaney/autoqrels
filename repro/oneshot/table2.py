import os
from itertools import combinations
import ir_measures
from ir_measures import SDCG, P, RBP
import ir_datasets
from scipy.stats import kendalltau, spearmanr, ttest_rel
import numpy as np
from tqdm import tqdm
import autoqrels


def rbo(a, b, p=0.9):
  # Adapted from the ir-measures implementation of Compat
  A = np.array(a).argsort()
  B = np.array(b).argsort()
  A_set = set()
  B_set = set()
  score = 0.0
  normalizer = 0.0
  weight = 1.0
  i = 0
  while i < len(A) or i < len(B):
    if i < len(A):
      A_set.add(A[i])
    if i < len(B):
      B_set.add(B[i])
    score += weight * len(A_set.intersection(B_set)) / (i + 1)
    normalizer += weight
    weight *= p
    i += 1
  return score / normalizer


def stat_test(results1, results2, alternative, pvalue=0.05):
  qids = list(results1.keys() | results2.keys())
  scores1 = [results1.get(qid, 0.) for qid in qids]
  scores2 = [results2.get(qid, 0.) for qid in qids]
  test = ttest_rel(scores1, scores2, alternative=alternative)
  return test.pvalue < pvalue


def false_rates_top(official_results, these_results, pvalue=0.05):
  # This function helps us answer whether one would draw the same statistical
  # conclusion about the "top" system as the "official" results if you used these qrels.

  # correct for multiple tests
  pvalue = pvalue / (len(official_results.keys() | these_results.keys()) - 1)

  # find the top system using these results
  top_run = max(these_results, key=lambda x: sum(these_results[x].values()))

  # we'll count TPs, FPs, TNs, FNs
  tp, fp, tn, fn = 0, 0, 0, 0

  for comp_run in official_results.keys() | these_results.keys():
    if top_run == comp_run:
      continue # don't compare against itself
    official_test = stat_test(official_results[top_run], official_results[comp_run],
      alternative='greater', pvalue=pvalue)
    this_test = stat_test(these_results[top_run], these_results[comp_run],
      alternative='greater', pvalue=pvalue)
    if this_test:
      if official_test:
        tp += 1
      else:
        fp += 1
    else:
      if official_test:
        fn += 1
      else:
        tn += 1
  fpr = fp / (fp + tn + 1e-100)
  fnr = fn / (fn + tp + 1e-100)
  return fpr, fnr


def eval(official_results, these_results):
  runs = sorted(official_results.keys() | these_results.keys())
  A, B = [], []
  for run in runs:
    A.append(sum(official_results[run].values()) / len(official_results[run]))
    B.append(sum(these_results[run].values()) / len(these_results[run]))
  fpr, fnr = false_rates_top(official_results, these_results)
  return {
    'kendall': kendalltau(A, B)[0],
    'spearman': spearmanr(A, B)[0],
    'rbo': rbo(A, B),
    'fpr': fpr,
    'fnr': fnr,
  }


def calc_measures(Measure, runs, qrels):
  results = {}
  for run_id, run in runs.items():
    results[run_id] = {m.query_id: m.value for m in Measure.iter_calc(qrels, run)}
  return results


DATASETS = []
for ds, dsid in tqdm([
  ('dl19', 'msmarco-passage/trec-dl-2019'),
  ('dl20', 'msmarco-passage/trec-dl-2020'),
  ('dl21', 'msmarco-passage-v2/trec-dl-2021')], desc='reading runs'):
  runs = {}
  for run_id in os.listdir(f'{ds}-runs'):
    runs[run_id] = list(ir_measures.read_trec_run(f'{ds}-runs/{run_id}'))
  DATASETS.append((
    ds,
    list(ir_datasets.load(dsid).qrels),
    list(ir_measures.read_trec_qrels(f'{ds}.bm25-firstrel.qrels')),
    runs,
  ))


SYSTEMS = [
  ('Non-relevant', autoqrels.DummyLabeler()),
  ('MaxRep-BM25', autoqrels.oneshot.OneShotLabeler('maxrep.bm25-128.cache.json.gz')),
  ('MaxRep-TCT', autoqrels.oneshot.OneShotLabeler('maxrep.tcthnp-128.cache.json.gz')),
  ('DuoT5', autoqrels.oneshot.OneShotLabeler('duot5.cache.json.gz')),
  ('DuoPrompt', autoqrels.oneshot.OneShotLabeler('duoprompt.cache.json.gz')),
]


MEASURES = [
  ('SDCG@10', SDCG(max_rel=3)@10, lambda x: x.SDCG@10),
  ('P@10', P(rel=2)@10, lambda x: x.P@10),
  ('RBP(p=0.8)', RBP(p=0.8, rel=2), lambda x: x.RBP(p=0.8)),
]


def main():
  with tqdm(total=len(MEASURES) * len(DATASETS) * (len(SYSTEMS) + 1)) as pbar:
    official_measure_map = {}
    for m_name, OfficialMeasure, f_measure in MEASURES:
      for ds, official_qrels, sparse_qrels, runs in DATASETS:
        pbar.set_description(f'calculating official {ds} {m_name}')
        official_measure_map[m_name, ds] = calc_measures(
          OfficialMeasure, runs, official_qrels)
        pbar.update()

    for m_name, OfficialMeasure, inf_measure_fn in MEASURES:
      for i, (provider_name, provider) in enumerate(SYSTEMS):
        InfMeasure = inf_measure_fn(provider)
        ev_map = {}
        first = f'\\textbf{{{m_name}}}' if i == 0 else ''  # print measure on first line
        for i, (ds, official_qrels, sparse_qrels, runs) in enumerate(DATASETS):
          pbar.set_description(f'calculating {provider_name} {ds} {m_name}')
          official_measures = official_measure_map[m_name, ds]
          these_measures = calc_measures(InfMeasure, runs, sparse_qrels)
          ev = eval(official_measures, these_measures)
          for k, v in ev.items():
            ev_map[f'{ds}_{k}'] = v
          pbar.update()
        tqdm.write('{first} & {provider_name}'
              ' & {dl19_kendall:.3f} & {dl19_spearman:.3f} & {dl19_rbo:.3f} & {dl19_fnr:.3f} & {dl19_fpr:.3f}'  # noqa: E501
              ' & {dl20_kendall:.3f} & {dl20_spearman:.3f} & {dl20_rbo:.3f} & {dl20_fnr:.3f} & {dl20_fpr:.3f}'  # noqa: E501
              ' & {dl21_kendall:.3f} & {dl21_spearman:.3f} & {dl21_rbo:.3f} & {dl21_fnr:.3f} & {dl21_fpr:.3f}'  # noqa: E501
              ' \\\\'.format(first=first, provider_name=provider_name, **ev_map))
      tqdm.write('\\midrule')


if __name__ == '__main__':
  main()
