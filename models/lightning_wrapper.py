import pytorch_lightning as L
import torch
from torch.optim import lr_scheduler, AdamW, SGD
from torchmetrics import ROC
from pytorch_metric_learning.losses import ArcFaceLoss, CosFaceLoss
import torch.distributed as dist
from sklearn.metrics import f1_score


def _similarity(x, y):
    return torch.dot(x, y) / (torch.linalg.norm(x) * torch.linalg.norm(y))


class LightningWrapper(L.LightningModule):
    def __init__(self, model, config, num_classes=0, num_batches=0):
        super(LightningWrapper, self).__init__()

        self.config = config
        self.num_batches = num_batches
        self.num_classes = num_classes

        self.model = model
        self._configure_criterions()
        self.automatic_optimization = False

        self.val_sims = []
        self.val_gts = []
        self.roc = ROC(task="binary")

    def _configure_criterions(self):
        if "multitask" in self.config.model_name:
            self.task_criterions = {
                "gender_fc": torch.nn.BCEWithLogitsLoss(),
                "hair_fc": torch.nn.CrossEntropyLoss(),
                "glasses_fc": torch.nn.CrossEntropyLoss(),
                "mustache_fc": torch.nn.BCEWithLogitsLoss(),
                "hat_fc": torch.nn.BCEWithLogitsLoss(),
                "open_mouth_fc": torch.nn.BCEWithLogitsLoss(),
                "long_hair_fc": torch.nn.BCEWithLogitsLoss()
            }

        self.embed_criterion = CosFaceLoss(self.num_classes, self.config.embedding_size)
        self.class_criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, text_prompt=None):
        if "multitask" in self.config.model_name:
            return self.model(x, text_prompt)
        else:
            return self.model(x)

    def switch_random_task(self):
        rand = self.task_rng.random()
        for task in self.task_weights:
            if rand > self.task_weights[task]:
                rand -= self.task_weights[task]
            else:
                self.switch_task_by_layer_name(task)
                break

    def training_step(self, batch, batch_idx):
        # CAUTION: Do not use this function until fixed.
        self.model.switch_random_task()

        model_opt, crit_opt = self.optimizers()
        model_sched, crit_sched = self.lr_schedulers()

        x = batch["data"]
        gt = batch["annotation"]

        if "multitask" in self.config.model_name:
            out = self(x["image"], x["textual_prompt"])
        loss = self._loss(out, gt)
        self.manual_backward(loss)

        model_opt.step()
        crit_opt.step()
        model_opt.zero_grad()
        crit_opt.zero_grad()

        model_sched.step()
        crit_sched.step()

        self.log("loss", loss, prog_bar=True)
        self.log("lrm", model_opt.param_groups[0]['lr'], prog_bar=True)
        self.log("lrc", crit_opt.param_groups[0]['lr'], prog_bar=True)

        return loss

    def _loss(self, out, gt):
        loss = self.config.embedding_loss_rate * self.embed_criterion(out["embedding"], gt["class"])

        if "multitask" in self.config.model_name:
            task_out = out[self.model.active_task_layer[:-3]]
            task_gt = gt[self.model.active_task_layer[:-3]]
            task_rate = 1 - self.config.embedding_loss_rate

            if self.model.active_task_layer == "class_fc":
                loss += task_rate * self.class_criterion(task_out, task_gt)
            else:
                loss += task_rate * self.task_criterions[self.model.active_task_layer](task_out, task_gt)
        else:
            loss += (1 - self.config.embedding_loss_rate) * self.class_criterion(out["class"], gt["class"])

        return loss

    def validation_step(self, batch, _):
        img1 = batch["img1"]
        img2 = batch["img2"]
        label = batch["label"].int()

        embedding1 = self(img1)["embedding"]
        embedding2 = self(img2)["embedding"]

        similarity = torch.stack([_similarity(em1, em2) for em1, em2 in zip(embedding1, embedding2)])

        self.val_sims.extend(similarity)
        self.val_gts.extend(label)

    def gather_results(self):
        val_sims_tensor = torch.tensor(self.val_sims).to(self.device)
        val_gts_tensor = torch.tensor(self.val_gts).to(self.device)

        all_outputs = None
        all_targets = None

        if dist.get_rank() == 0:
            all_outputs = [torch.empty_like(val_sims_tensor) for _ in range(dist.get_world_size())]
            all_targets = [torch.empty_like(val_gts_tensor) for _ in range(dist.get_world_size())]

        dist.gather(val_sims_tensor, gather_list=all_outputs, dst=0)
        dist.gather(val_gts_tensor, gather_list=all_targets, dst=0)

        if dist.get_rank() == 0:
            all_outputs = torch.cat(all_outputs)
            all_targets = torch.cat(all_targets).int()

        return all_outputs, all_targets

    def on_validation_epoch_end(self):
        if dist.is_initialized():
            all_outputs, all_targets = self.gather_results()
        else:
            all_outputs = torch.tensor(self.val_sims).to(self.device)
            all_targets = torch.tensor(self.val_gts).to(self.device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            fpr, tpr, thresholds = self.roc(all_outputs, all_targets)

            indexes = torch.tensor([True if rate <= 1e-3 else False for rate in fpr])
            best_index = torch.argmax(tpr[indexes])

            if all(torch.isnan(x) for x in fpr) or all(torch.isnan(x) for x in tpr):
                threshold = thresholds[0].to(self.device)
            else:
                threshold = thresholds[torch.argmin(torch.abs(fpr - (1 - tpr)))].to(self.device)

            all_outputs = torch.tensor([1. if x > threshold else 0. for x in all_outputs])
            f1 = f1_score(all_targets.cpu(), all_outputs.cpu())
            tpr = tpr[indexes][best_index].to(self.device)

            self.log('acc', f1, prog_bar=True, rank_zero_only=True)
            self.log('th', threshold, prog_bar=True, rank_zero_only=True)
            self.log('tar@far', tpr, prog_bar=True, rank_zero_only=True)

        self.val_sims = []
        self.val_gts = []

    def configure_optimizers(self):
        model_optimizer = AdamW(self.model.parameters(), lr=self.config.max_model_lr)
        criterion_optimizer = AdamW(self.embed_criterion.parameters(), lr=self.config.max_crit_lr)

        total_batches = self.config.num_of_epoch * self.num_batches
        warmup_period = int(max(self.config.warmup_epochs * self.num_batches, 1))

        model_warmup = lr_scheduler.LinearLR(model_optimizer, start_factor=1.0, end_factor=1.0, total_iters=warmup_period)
        criterion_warmup = lr_scheduler.LinearLR(criterion_optimizer, start_factor=1.0, end_factor=1.0, total_iters=warmup_period)
        model_scheduler = lr_scheduler.CosineAnnealingLR(model_optimizer, total_batches // 5, self.config.min_model_lr)
        criterion_scheduler = lr_scheduler.CosineAnnealingLR(criterion_optimizer, total_batches // 5, self.config.min_crit_lr)

        combined_model_scheduler = lr_scheduler.SequentialLR(
            model_optimizer,
            [model_warmup, model_scheduler],
            [warmup_period]
        )

        combined_criterion_scheduler = lr_scheduler.SequentialLR(
            criterion_optimizer,
            [criterion_warmup, criterion_scheduler],
            [warmup_period]
        )

        return ([model_optimizer, criterion_optimizer],
                [combined_model_scheduler, combined_criterion_scheduler])
