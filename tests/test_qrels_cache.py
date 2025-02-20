import pandas as pd
import pyterrier as pt
import autoqrels
import unittest
import tempfile


class TestQrelsCache(unittest.TestCase):

    def test_qrels_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            one_labeler = pt.apply.label(lambda x: 1)
            two_labeler = pt.apply.label(lambda x: 2)
            with autoqrels.QrelsCache(f'{tmpdir}/cache', one_labeler) as cache:
                result = cache(pd.DataFrame([
                    {'qid': '1', 'docno': 'A'},
                ]))
                self.assertEqual(len(result), 1)
                self.assertEqual(result['label'][0], 1)

            with autoqrels.QrelsCache(f'{tmpdir}/cache', two_labeler) as cache:
                result = cache(pd.DataFrame([
                    {'qid': '1', 'docno': 'A'},
                    {'qid': '1', 'docno': 'B'},
                ]))
                self.assertEqual(len(result), 2)
                self.assertEqual(result['label'][0], 1)
                self.assertEqual(result['label'][1], 2)

            with autoqrels.QrelsCache(f'{tmpdir}/cache') as cache:
                result = cache(pd.DataFrame([
                    {'qid': '1', 'docno': 'A'},
                    {'qid': '1', 'docno': 'B'},
                ]))
                self.assertEqual(len(result), 2)
                self.assertEqual(result['label'][0], 1)
                self.assertEqual(result['label'][1], 2)
