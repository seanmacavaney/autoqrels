import ir_measures
from ir_measures.providers.cwl_eval import CwlEvaluator

SUPPORTED_MEASURES = {'SDCG', 'P', 'RBP'}


def measure_factory(attr: str, auto_qrels_provider: str):
    if attr in SUPPORTED_MEASURES:
        M = getattr(ir_measures.measures, attr)
        _SUPPORTED_PARAMS = dict(M.SUPPORTED_PARAMS)
        name = repr(auto_qrels_provider) + '.' + M.NAME
        class _RuntimeMeasure(ir_measures.measures.Measure):
            nonlocal _SUPPORTED_PARAMS
            SUPPORTED_PARAMS = _SUPPORTED_PARAMS
            NAME = M.NAME
            __name__ = name
            _autoqrels_provider = auto_qrels_provider
            _autoqrels_base_measure = M

            def runtime_impl(self, qrels, run):
                inf_qrels = self._autoqrels_provider.infer_qrels(run, qrels)
                evaluator = CwlEvaluator([self], inf_qrels, {(None, 0., 1.): [self]}, verify_gains=False)
                return evaluator.iter_calc(run)
        Measure = _RuntimeMeasure()
        if 'max_rel' in _SUPPORTED_PARAMS:
            Measure = Measure(max_rel=1)
        return Measure
