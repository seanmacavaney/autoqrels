import pandas as pd

import autoqrels._ir_measures_integration


SUPPORTED_MEASURES = {'SDCG', 'P', 'RBP'}


class Labeler:
    def infer_qrels(self, run: pd.DataFrame, qrels: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError() # implementation-specific

    def __getattr__(self, attr: str):
        measure = autoqrels._ir_measures_integration.measure_factory(attr, self)
        if measure is not None:
            return measure
        return self.__getattribute__(attr)


class DummyLabeler(Labeler):
    def infer_qrels(self, run: pd.DataFrame, qrels: pd.DataFrame) -> pd.DataFrame:
        return qrels
