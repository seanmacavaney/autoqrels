from typing import List
from . import OneShotLabeler


class MaxRep(OneShotLabeler):
    def __init__(self, corpus_graph, score_mode='recip_rank', cache_path=None):
        super().__init__(cache_path=cache_path)
        self.corpus_graph = corpus_graph
        self.score_mode = score_mode

    def __repr__(self):
        return 'autoqrels.oneshot.MaxRep'

    def _infer_oneshot(self, query_id: str, rel_doc_id: str, unk_doc_ids: List[str]) -> List[float]:
        idx_map = {did: i for i, did in enumerate(unk_doc_ids)}
        result = [0. for _ in idx_map]
        neighbours = list(self.corpus_graph.neighbours(rel_doc_id))
        for i, neighbour_id in enumerate(neighbours):
             if neighbour_id in idx_map:
                idx = idx_map[neighbour_id]
                if self.score_mode == 'recip_rank':
                    result[idx] = (len(neighbours)-i) / len(neighbours)
                else:
                    raise ValueError(f'unsupported score_mode: {repr(self.score_mode)}')
        return result
