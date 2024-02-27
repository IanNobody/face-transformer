import pytorch_lightning as L
import torch
from sklearn.metrics import roc_curve, f1_score
import numpy as np
from torch.optim import lr_scheduler, AdamW, SGD
import pytorch_warmup as warmup

def _similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

class LightningWrapper(L.LightningModule):
    def __init__(self, model, max_model_lr, min_model_lr,
                 criterion, max_crit_lr, min_cri_lr,
                 warmup_period, total_batches, config):
        super(LightningWrapper, self).__init__()

        self.model = model
        self.max_model_lr = max_model_lr
        self.min_model_lr = min_model_lr
        self.model_warmup = None

        self.criterion = criterion
        self.max_crit_lr = max_crit_lr
        self.min_crit_lr = min_cri_lr
        self.criterion_warmup = None

        self.warmup_period = warmup_period
        self.config = config
        self.total_batches = total_batches

        self.val_sims = []
        self.val_gts = []

        self.automatic_optimization = False

    def forward(self, x):
        if self.config.model_name == "dat":
            return self.model(x)[0]
        else:
            return self.model(x)

    def training_step(self, batch, batch_idx):
        model_opt, crit_opt = self.optimizers()
        model_sched, crit_sched = self.lr_schedulers()

        model_opt.zero_grad()
        crit_opt.zero_grad()

        x = batch["image"]
        gt = batch["entity"]

        out = self(x)

        loss = self.criterion(out, gt)
        self.manual_backward(loss)

        model_opt.step()
        crit_opt.step()

        with self.model_warmup.dampening():
            if self.model_warmup.last_step + 1 >= self.warmup_period:
                model_sched.step()

        with self.criterion_warmup.dampening():
            if self.criterion_warmup.last_step + 1 >= self.warmup_period:
                crit_sched.step()

        self.log("train_loss", loss, prog_bar=True)
        self.log("model_lr", model_sched.get_last_lr()[0], prog_bar=True)
        self.log("crit_lr", crit_sched.get_last_lr()[0], prog_bar=True)

    def validation_step(self, batch, _):
        img1 = batch["img1"]
        img2 = batch["img2"]
        label = batch["label"].cpu()

        embedding1 = self(img1).cpu()
        embedding2 = self(img2).cpu()

        similarity = [_similarity(em1, em2) for em1, em2 in zip(embedding1, embedding2)]

        self.val_sims.extend(similarity)
        self.val_gts.extend(label)

    def on_validation_epoch_end(self):
        all_outputs = self.val_sims
        all_targets = self.val_gts

        fpr, tpr, thresholds = roc_curve(all_targets, all_outputs)

        if all(np.isnan(x) for x in fpr) or all(np.isnan(x) for x in tpr):
            threshold = thresholds[0]
        else:
            threshold = thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]

        all_outputs = [1 if x > threshold else 0 for x in all_outputs]

        self.val_sims = []
        self.val_gts = []

        self.log('valid_acc_epoch', f1_score(all_targets, all_outputs), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        model_optimizer = AdamW(self.model.parameters(), lr=self.max_model_lr)
        model_scheduler = lr_scheduler.CosineAnnealingLR(model_optimizer,
                                                         self.total_batches // 5,
                                                         self.min_model_lr)
        criterion_optimizer = SGD(self.criterion.parameters(), lr=self.max_crit_lr)
        criterion_scheduler = lr_scheduler.CosineAnnealingLR(criterion_optimizer,
                                                             self.total_batches // 5,
                                                             self.min_crit_lr)

        self.model_warmup = warmup.LinearWarmup(model_optimizer, warmup_period=self.warmup_period)
        self.criterion_warmup = warmup.LinearWarmup(criterion_optimizer, warmup_period=self.warmup_period)

        return [model_optimizer, criterion_optimizer], [
            {
                'scheduler': model_scheduler,
                'interval': 'step'
            },
            {
                'scheduler': criterion_scheduler,
                'interval': 'step'
            }
        ]