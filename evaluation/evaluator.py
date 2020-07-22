import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions
from pytorch_pfn_extras import reporting
from pytorch_pfn_extras.training.extensions.evaluator import IterationStatus
from pytorch_pfn_extras.training.extensions.evaluator import _IteratorProgressBar
from pytorch_pfn_extras.training.extensions.evaluator import _in_eval_mode
import numpy as np


class ALASKAEvaluator(extensions.Evaluator):
    def __init__(self, iterator, target, eval_hook=None, eval_func=None,
                 loss_criterion=None, auc_criterion=None, device=None, scheduler=None,
                 metric_learning=False):
        super(ALASKAEvaluator, self).__init__(
            iterator, target
        )
        self._eval_init()
        self.loss_criterion = loss_criterion
        self.auc_criterion = auc_criterion
        self.device = device
        self.scheduler = scheduler
        self.metric_learning = metric_learning
        #self._eval_func = eval_func

    def _eval_init(self):
        self.loss = 0
        self.y = []
        self.pred = []
        self.auc = 0

    def evaluate(self):
        self._eval_init()
        iterator = self._iterators['main']
        if self.eval_hook:
            self.eval_hook(self)

        #summary = reporting.DictSummary()

        updater = IterationStatus(len(iterator))
        if self._progress_bar:
            pbar = _IteratorProgressBar(iterator=updater)

        with _in_eval_mode(self._targets.values()):
            for idx, batch in enumerate(iterator):
                x, t = batch
                x, t = x.to(self.device), t.to(self.device)
                updater.current_position = idx
                if self.metric_learning:
                    pred = self.eval_func(x)
                else:
                    pred = self.eval_func(x)
                loss = self.loss_criterion(pred, t)
                self.loss += loss.item() * len(t)
                self.pred.extend(pred.detach().cpu().numpy())
                self.y.extend(t.cpu().numpy().astype(int))
                #observation = {}
                #with reporting.report_scope(observation):
                #    if isinstance(batch, (tuple, list)):
                #        self.eval_func(*batch)
                #    elif isinstance(batch, dict):
                #        self.eval_func(**batch)
                #    else:
                #        self.eval_func(batch)
                #summary.add(observation)

                if self._progress_bar:
                    pbar.update()
            self.auc = self.auc_criterion(self.y, self.pred)
            if self.scheduler is not None:
                self.scheduler.step(metrics=self.loss / len(self.y))
        if self._progress_bar:
            pbar.close()

        observation = {}
        with reporting.report_scope(observation):
            ppe.reporting.report({'validation/loss': self.loss / len(self.y)})
            ppe.reporting.report({'validation/auc': self.auc})

        return observation

        #return summary.compute_mean()


        #model.eval()
        #with torch.no_grad():
        #    for x, t in val_loader:
        #        x, t = x.to(self.device), t.to(self.device)
        #        loss = self.criterion(model(x), t).item()
        #        self.y.extend(t.cpu().numpy().astype(int))
        #        self.pred.extend()
