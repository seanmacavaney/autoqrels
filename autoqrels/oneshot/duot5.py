from typing import List
from functools import cached_property
import ir_datasets
import more_itertools
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
import autoqrels
from . import OneShotLabeler


logger = ir_datasets.log.easy()


class DuoT5(OneShotLabeler):
    def __init__(self, dataset, model_name='castorini/duot5-3b-msmarco', tokeniser_name='t5-3b', device=None, batch_size=8, verbose=False, query_field=None, doc_field=None, cache_path=None):
        super().__init__(cache_path=cache_path)
        self.model_name = model_name
        self.tokeniser = AutoTokenizer.from_pretrained(tokeniser_name, model_max_length=512)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.verbose = verbose
        self.REL = self.tokeniser.encode('true')[0]
        self.NREL = self.tokeniser.encode('false')[0]
        self.dataset = dataset
        self.query_field = query_field
        self.doc_field = doc_field

    @cached_property
    def model(self):
        return T5ForConditionalGeneration.from_pretrained(self.model_name).eval().to(self.device)

    def __repr__(self):
        return 'autoqrels.oneshot.DuoT5'

    def _infer_oneshot(self, query_id: str, rel_doc_id: str, unk_doc_ids: List[str]) -> List[float]:
        return self.infer_oneshot_text(
            autoqrels.text.query_text(self.dataset, query_id, self.query_field),
            autoqrels.text.doc_text(self.dataset, rel_doc_id, self.doc_field),
            autoqrels.text.doc_text(self.dataset, unk_doc_ids, self.doc_field))

    def infer_oneshot_text(self, query_text: str, rel_doc_text: str, unk_doc_texts: List[str]) -> List[float]:
        it = ((query_text, rel_doc_text, d) for d in unk_doc_texts)
        if self.verbose:
            it = logger.pbar(list(it), desc=repr(self), unit='d')
        result = []
        with torch.no_grad():
            for chunk in more_itertools.chunked(it, self.batch_size):
                batch = self.tokeniser.batch_encode_plus(
                    [f'Query: {q} Document0: {t2} Document1: {t1} Relevant:' for q, t1, t2 in chunk],
                    return_tensors='pt',
                    padding=True,
                    truncation=True)
                batch['decoder_input_ids'] = torch.full(
                    (batch['input_ids'].shape[0], 1),
                    self.model.config.decoder_start_token_id,
                    dtype=torch.long
                )
                res = self.model(**{k: v.to(self.device) for k, v in batch.items()})
                scores = res['logits'][:, 0, [self.REL, self.NREL]]
                scores = scores.softmax(dim=1)[:, 0].cpu().tolist()
                result.extend(scores)
        return result
