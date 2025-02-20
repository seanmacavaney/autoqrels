import pyterrier as pt
from pyterrier_caching import Sqlite3KeyValueCache

import autoqrels._ir_measures_integration


class QrelsCache(Sqlite3KeyValueCache):
    """A cache for qrels, which is a key-value store with keys 'qid' and 'docno', and values 'label'.

    This class is intended to wrap a :class:`autoqrels.Labeler` to avoid re-computing qrels.
    """
    ARTIFACT_TYPE = 'qrels_cache'
    ARTIFACT_FORMAT = 'sqlite3'

    def __init__(self, path: str, labeler: pt.Transformer = None, verbose: bool = False):
        super().__init__(path, labeler, key=['qid', 'docno'], value=['label'], verbose=verbose)

    def __getattr__(self, attr: str):
        measure = autoqrels._ir_measures_integration.measure_factory(attr, self)
        if measure is not None:
            return measure
        return self.__getattribute__(attr)
