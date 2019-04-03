import unittest
import os, sys

import base
import wer


class Test(unittest.TestCase):
    def setUp(self):
        pass

    def test_wer(self):
        # refer = 'if you choose the subject about finance <?> in the future after you after you graduate from the high from the'
        # hypos = 'if you choose the subject about finance you will get you will be you will easily get a good job in the future after your graduate from the high from the'

        refer = 'A <?> C'
        hypos = 'A B C D C'

        res = base.TxtScore(hypos.split(), refer.split())
        print('err={} word={}'.format(res['err'], res['word']))
        print('{ins} {del} {rep}'.format(**res))

        res = wer.wer(hypos, refer, '<?>')
        print('err={} word={}'.format(res['err'], res['word']))
        print('{ins} {del} {rep}'.format(**res))
        print('refer = {}'.format(' '.join(res['refer'])))
        print('hypos = {}'.format(' '.join(res['hypos'])))



unittest.main()
