from typing import List
import pandas as pd
import ir_datasets
import gzip
import json
from pathlib import Path
from autoqrels import Labeler
from warnings import warn


class ZeroShotLabeler(Labeler):
    def __init__(self, cache_path=None):
        super().__init__()
        self._cache = {}
        self._cache_dirty = False
        self.cache_path = cache_path
        # pre-load cache from disk if enabled and it exists
        if self.cache_path is not None:
            # The format of the cache file for zero-shot labelers looks like:
            # {query_id: {unk_doc_id: score}}
            self.cache_path = Path(self.cache_path)
            if self.cache_path.exists():
                with gzip.open(self.cache_path, 'rt') as fin:
                    self._cache = json.load(fin)

    def infer_qrels(self, run: pd.DataFrame, qrels: pd.DataFrame = None) -> pd.DataFrame:
        if qrels is not None and len(qrels) > 0:
            warn("zeroshot labelers do not use qrels; they will be ignored")
        run = dict(iter(run.groupby('query_id')))
        result = []
        for query_id, run_group in run.items():
            unk_doc_ids = list(run_group['doc_id'])
            inferred_rel = self.infer_zeroshot(query_id, unk_doc_ids)
            result.extend(((query_id, did, rel, '1') for did, rel in zip(unk_doc_ids, inferred_rel))) # iteration=1 means inferred qrel
        self.flush_cache() # flush if cache is dirty
        return pd.DataFrame(result, columns=['query_id', 'doc_id', 'relevance', 'iteration'])

    def infer_zeroshot(self, query_id: str, unk_doc_ids: List[str]) -> List[float]:
        # get results from cache
        unk_doc_ids = list(unk_doc_ids)
        c = self._cache.setdefault(query_id, {})
        res = [c.get(did) for did in unk_doc_ids]

        # calculate missing values (if any)
        uncached_idxs = [idx for idx, r in enumerate(res) if r is None]
        if uncached_idxs:
            uncached_dids = [unk_doc_ids[idx] for idx in uncached_idxs]
            uncached_res = self._infer_zeroshot(query_id, uncached_dids)
            for did, idx, score in zip(uncached_dids, uncached_idxs, uncached_res):
                res[idx] = score
                c[did] = score
            # mark cache as dirty so it can be flushed later
            self._cache_dirty = True
        return res

    def flush_cache(self):
        if self._cache_dirty and self.cache_path is not None:
            with ir_datasets.util.finialized_file(str(self.cache_path), 'wb') as fout, \
                 gzip.open(fout, 'wt') as gz_fout:
                json.dump(self._cache, gz_fout)
            self._cache_dirty = False

    def _infer_zeroshot(self, query_id: str, unk_doc_ids: List[str]) -> List[float]:
        raise NotImplementedError() # implementation-specific
