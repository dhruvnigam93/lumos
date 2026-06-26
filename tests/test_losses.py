import torch

from lumos.training.losses import MultiTaskLoss, DirectLearnedWeightedLoss


class TestMultiTaskLoss:
    def test_forward(self):
        is_reg = torch.tensor([False, True, False])
        loss_fn = MultiTaskLoss(is_regression=is_reg, reduction="sum")
        losses = torch.tensor([0.5, 1.0, 0.3], requires_grad=True)
        total = loss_fn(losses)
        total.backward()
        assert total.item() > 0
        assert losses.grad is not None

    def test_reduction_mean(self):
        is_reg = torch.tensor([False, True])
        loss_fn = MultiTaskLoss(is_regression=is_reg, reduction="mean")
        losses = torch.tensor([0.5, 1.0], requires_grad=True)
        total = loss_fn(losses)
        assert total.item() > 0

    def test_learnable_parameters(self):
        loss_fn = MultiTaskLoss(is_regression=torch.tensor([False, True]), reduction="sum")
        params = list(loss_fn.parameters())
        assert len(params) == 1
        assert params[0].shape == (2,)


class TestDirectLearnedWeightedLoss:
    def test_forward(self):
        loss_fn = DirectLearnedWeightedLoss(n_tasks=3)
        losses = torch.tensor([0.5, 1.0, 0.3], requires_grad=True)
        total = loss_fn(losses)
        total.backward()
        assert total.item() > 0
        assert losses.grad is not None

    def test_initial_weights_equal(self):
        loss_fn = DirectLearnedWeightedLoss(n_tasks=3)
        losses = torch.tensor([1.0, 1.0, 1.0])
        total = loss_fn(losses)
        assert abs(total.item() - 3.0) < 1e-5
