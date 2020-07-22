import torch
import torch.nn as nn

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05, num_classes=4):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = torch.eye(self.num_classes)[target]
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -logprobs.to('cuda') * target.to('cuda')
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)