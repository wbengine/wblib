import os
import unittest
import base

testdata = os.path.join(os.path.dirname(__file__), 'testdata')
outdir = base.mkdir('/Users/lls/tmp/wblibTestOutput')


class BaseTest(unittest.TestCase):
    def setUp(self):
        self.wp_tools_dir = '/Users/lls/tools/subword-nmt'
        self.wp_bpe_codes = testdata + '/trans.bpe.codes'
        self.wp_vocabulary = testdata + '/vocab.wp'

    def test_generate_wp(self):
        self.assertEqual(
            base.generate_wp(self.wp_tools_dir,
                             ['this is a cat', ''],
                             self.wp_bpe_codes,
                             self.wp_vocabulary),
            ['this is a ca@@ t',
             '']
        )

        self.assertEqual(
            base.generate_wp(self.wp_tools_dir,
                             ['', '', ''],
                             self.wp_bpe_codes,
                             self.wp_vocabulary),
            ['', '', '']
        )
