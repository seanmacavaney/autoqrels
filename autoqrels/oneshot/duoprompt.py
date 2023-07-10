from functools import cached_property
from typing import List
import ir_datasets
import more_itertools
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import smashed
import autoqrels
from . import OneShotLabeler


logger = ir_datasets.log.easy()


class DuoPrompt(OneShotLabeler):
    PROMPT = (
        "Determine if passage B is as relevant as passage A "
        "for the given query. "
        'Passage A: "...{{ rel_doc_text | replace("\\"", "\'") }}..." '
        'Passage B: "...{{ unk_doc_text | replace("\\"", "\'") }}..." '
        'Query: "{{ query_text }}" '
        "Is passage B as relevant as passage A? </s>"
    )

    def __init__(self, dataset, backbone='google/flan-t5-xl', device=None, batch_size=8, verbose=False, query_field=None, doc_field=None, max_src_len=330, cache_path=None):
        super().__init__(cache_path=cache_path)
        self.backbone = backbone
        self.tokeniser = AutoTokenizer.from_pretrained(backbone)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.verbose = verbose
        self.rel_token_id = self.tokeniser("Yes", add_special_tokens=False).input_ids[0]
        self.non_rel_token_id = self.tokeniser("No", add_special_tokens=False).input_ids[0]
        self.dataset = dataset
        self.query_field = query_field
        self.doc_field = doc_field
        fields = ['query_text', 'rel_doc_text', 'unk_doc_text']
        m_jinja = smashed.mappers.JinjaMapper(jinja=self.PROMPT)
        m_txt2word = smashed.mappers.TextToWordsMapper(
            fields=fields,
        )
        prompt_length = max_src_len - len(
            m_txt2word.splitter(m_jinja.template_text[0])
        )
        self.encode_mapper = (
            m_txt2word
            >> smashed.mappers.TruncateMultipleNestedFieldsMapper(
                fields_to_truncate=fields,
                max_length=prompt_length,
            )
            >> smashed.mappers.WordsToTextMapper(
                fields=fields,
            )
            >> m_jinja
            >> smashed.mappers.TokenizerMapper(
                tokenizer=self.tokeniser,
                input_field="source",
                add_special_tokens=False, # prompt itself has a </s> at the end!
                truncation=True,
            )
            >> smashed.mappers.ChangeFieldsMapper(
                keep_fields=["input_ids", "attention_mask"],
            )
            >> smashed.mappers.Python2TorchMapper()
            >> smashed.mappers.FixedBatchSizeMapper(
                batch_size=batch_size,
            )
            >> smashed.mappers.FromTokenizerTensorCollatorMapper(
                tokenizer=self.tokeniser,
                fields_pad_ids={"labels": -100},
            )
        )
        self.decode_mapper = (
            smashed.mappers.Torch2PythonMapper()
            >> smashed.mappers.UnpackingMapper(
                fields_to_unpack=["labels", "input_ids", "prob"],
                ignored_behavior="drop",
            )
            >> smashed.mappers.DecodingMapper(
                tokenizer=self.tokeniser,
                fields=["input_ids", "labels"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        )

    def __repr__(self):
        return 'autoqrels.oneshot.DuoPrompt'

    @cached_property
    def model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.backbone).eval().to(self.device)

    def _infer_oneshot(self, query_id: str, rel_doc_id: str, unk_doc_ids: List[str]) -> List[float]:
        return self.infer_oneshot_text(
            autoqrels.text.query_text(self.dataset, query_id, self.query_field),
            autoqrels.text.doc_text(self.dataset, rel_doc_id, self.doc_field),
            autoqrels.text.doc_text(self.dataset, unk_doc_ids, self.doc_field))

    def infer_oneshot_text(self, query_text: str, rel_doc_text: str, unk_doc_texts: List[str]) -> List[float]:
        prompt_data = self.encode_mapper.map([
            {'query_text': query_text, 'rel_doc_text': rel_doc_text, 'unk_doc_text': unk_doc_text}
            for unk_doc_text in unk_doc_texts
        ])
        preds: List[Dict[str, str]] = []

        if self.verbose:
            prompt_data = logger.pbar(prompt_data, desc=repr(self), unit='batch')

        with torch.no_grad():
            for batch in prompt_data:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                decoder_input_ids = torch.full(
                    (input_ids.size(0), 1),
                    int(self.model.config.decoder_start_token_id),
                    dtype=torch.long,
                    device=self.device,
                )
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )
                logits = output.logits[
                    :, -1, [self.non_rel_token_id, self.rel_token_id]
                ].to(torch.float32)
                probs = torch.softmax(logits, dim=-1)

                # decode the generated string
                batch_pred = self.decode_mapper.map(
                    [
                        {
                            **batch,
                            "labels": torch.argmax(output.logits, dim=-1),
                            "prob": probs[:, 1],
                        }
                    ]
                )
                preds.extend([r['prob'] for r in batch_pred])

        return preds
