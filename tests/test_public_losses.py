import os
from pathlib import Path
import unittest

import torch
from painter_rl.public_losses import PublicLosses


class LossContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cache = Path(os.environ.get("PUBLIC_LOSS_CACHE", "work/public_sources"))
        cls.losses = PublicLosses(cache, download=True)

    def test_same_canvas_grouping(self):
        a = self.losses.trajectory_advantages([0., 1., 100., 100.], ["A", "A", "B", "B"], 2)
        self.assertAlmostEqual(a[0].item(), -.707106, places=5)
        self.assertAlmostEqual(a[1].item(), .707106, places=5)
        self.assertTrue(torch.equal(a[2:], torch.zeros(2)))

    def test_constant_group_no_division_nan(self):
        a = self.losses.trajectory_advantages([.5] * 8, ["A"] * 8, 8)
        self.assertTrue(torch.equal(a, torch.zeros(8)))

    def test_incomplete_group_rejected(self):
        with self.assertRaises(ValueError):
            self.losses.trajectory_advantages([0, 1, 2], ["A", "A", "B"], 2)

    def test_scalar_tensor_group_ids_rejected(self):
        with self.assertRaises(TypeError):
            self.losses.trajectory_advantages([0, 1], [torch.tensor(0), torch.tensor(0)], 2)

    def test_observation_tokens_have_no_gradient(self):
        student = torch.tensor([[-1., -2., -3., -4.]], requires_grad=True)
        teacher = torch.tensor([[-.5, -2., -1., -3.]], requires_grad=True)
        old = student.detach().clone().requires_grad_(True)
        reference = old.detach().clone().requires_grad_(True)
        mask = torch.tensor([[0., 1., 0., 1.]])
        loss, _ = self.losses.combined(student, old, teacher, reference,
                                       torch.ones_like(student), mask)
        loss.backward()
        self.assertEqual(student.grad[0, 0].item(), 0)
        self.assertEqual(student.grad[0, 2].item(), 0)
        self.assertNotEqual(student.grad[0, 1].item(), 0)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(old.grad)
        self.assertIsNone(reference.grad)

    def test_positive_advantage_increases_sampled_token_probability(self):
        for advantage, sign in [(1., -1), (-1., 1)]:
            student = torch.full((1, 2), -1., requires_grad=True)
            loss, _ = self.losses.combined(
                student, student.detach(), student.detach(), student.detach(),
                torch.full_like(student, advantage), torch.ones_like(student),
                sdar_coef=0., ref_kl_coef=0., entropy_coef=0.)
            loss.backward()
            self.assertTrue(torch.all(sign * student.grad > 0))

    def test_sdar_beta_five_is_teacher_dependent_and_detached(self):
        gradients = []
        for teacher_score in (-.1, -3.):
            student = torch.tensor([[-1.]], requires_grad=True)
            teacher = torch.tensor([[teacher_score]], requires_grad=True)
            loss, _ = self.losses.functions["compute_sdar_loss"](
                student, teacher, torch.ones_like(student), 5., "token-mean")
            loss.backward()
            self.assertIsNone(teacher.grad)
            gradients.append(student.grad.item())
        self.assertLess(gradients[0], -0.98)
        self.assertGreater(gradients[1], -0.001)

    def test_microbatch_scaling_matches_global_token_mean(self):
        # A training adapter must weight each microbatch by its GENERATED token
        # count, not by number of turns, to preserve this public token-mean.
        s = torch.tensor([[-1., -2.], [-3., -4.]], requires_grad=True)
        mask = torch.tensor([[1., 0.], [1., 1.]])
        t = s.detach() + .2
        a = torch.tensor([[1., 1.], [-1., -1.]])
        loss, _ = self.losses.combined(s, s.detach(), t, s.detach(), a, mask)
        target = torch.autograd.grad(loss, s)[0]
        s2 = s.detach().clone().requires_grad_(True)
        total = 0.
        for i in range(2):
            part, _ = self.losses.combined(s2[i:i+1], s[i:i+1].detach(),
                    t[i:i+1], s[i:i+1].detach(), a[i:i+1], mask[i:i+1])
            total = total + part * mask[i].sum() / mask.sum()
        actual = torch.autograd.grad(total, s2)[0]
        torch.testing.assert_close(actual, target)


if __name__ == "__main__":
    unittest.main()
