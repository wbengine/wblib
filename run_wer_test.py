import os
import unittest
import base
import remote


class Test(unittest.TestCase):
    def setUp(self):
        remote_dir = '/riseml/workspace/engine/exp/lls-new-data-v3-try1-wp500-adaptive-on-telis-chatbot/expr1-mwer/1pass_recog/chatbot_test_v3-lp0.10-beam16'

        local_dir = base.mkdir('/Users/lls/tmp/run_wer_test')

        remote.download(remote.ali_ssh, remote_dir + '.*', local_dir)

        name = base.path_file_name(remote_dir)
        self.nbest = local_dir + '/' + name + '.nbest'
        self.refer = local_dir + '/' + name + '.refer'
        self.best = local_dir + '/' + name + '.best'
        self.log = local_dir + '/' + name + '.log'

        base.GetBest(self.nbest, None, self.best)

    def test(self):

        os.system('python ./run_wer.py --best {} --refer {} --filter ./filters/wer_hyp_filter > {}'.format(
            self.best, self.refer, self.log
        ))

        os.system('python ./run_wer.py --nbest {} --refer {} --filter ./filters/wer_hyp_filter > {}'.format(
            self.nbest, self.refer, self.log
        ))

if __name__ == '__main__':
    unittest.main()
