"""CPU checks for canonical greedy and tied-maximum reporting semantics."""

import unittest

import torch

from hf_check import greedy_match_masks


class HFTieReportingTests(unittest.TestCase):
    def test_noncanonical_tied_token_is_max_probability_but_not_argmax(self):
        logits = torch.tensor([[3., 0., 3., 1., 0.], [0., 4., 1., 1., 0.], [5., 4., 3., 0., 1.]])
        logprobs = logits.log_softmax(-1)
        chosen = torch.tensor([2, 1, 3])
        selected = logprobs.gather(1, chosen[:, None]).squeeze(1)
        best = logprobs.max(-1).values
        canonical, maximum, argmax_ids = greedy_match_masks(logprobs, chosen, selected, best)
        self.assertEqual(argmax_ids.tolist(), [0, 1, 0])
        self.assertEqual(canonical.tolist(), [False, True, False])
        self.assertEqual(maximum.tolist(), [True, True, False])
        self.assertEqual(int(canonical.sum()), 1)
        self.assertEqual(int(maximum.sum()), 2)


if __name__ == "__main__":
    unittest.main()
