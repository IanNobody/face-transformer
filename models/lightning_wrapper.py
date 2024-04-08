import pytorch_lightning as L
import torch
from torch.optim import lr_scheduler, AdamW, SGD
import random
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
        self._configure_custom_criterions()

        self.automatic_optimization = False
        self.current_objective = "embed_fc"

        self.task_codes = {"head": 0,
                           "gender_fc": 10, "hair_fc": 11, "glasses_fc": 12, "mustache_fc": 13, "hat_fc": 14,
                           "open_mouth_fc": 15, "long_hair_fc": 16}
        self.task_weights = {"head": 1.0,
                             "gender_fc": 0., "hair_fc": 0., "glasses_fc": 0., "mustache_fc": 0.,
                             "hat_fc": 0.0, "open_mouth_fc": 0., "long_hair_fc": 0.}
        self.task_rng = random.Random(412)

        self.val_sims = []
        self.val_gts = []
        self.roc = ROC(task="binary")

    def _configure_custom_criterions(self):
        if "multitask" in self.config.model_name:
            self.hair_color_criterion = torch.nn.CrossEntropyLoss()
            self.glasses_criterion = torch.nn.CrossEntropyLoss()
            self.gender_criterion = torch.nn.BCEWithLogitsLoss()
            self.mustache_criterion = torch.nn.BCEWithLogitsLoss()
            self.hat_criterion = torch.nn.BCEWithLogitsLoss()
            self.open_mouth_criterion = torch.nn.BCEWithLogitsLoss()
            self.long_hair_criterion = torch.nn.BCEWithLogitsLoss()

        self.embed_criterion = CosFaceLoss(self.num_classes, self.config.embedding_size)
        self.classification_criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def switch_task_by_layer_name(self, layer_name):
        self.current_objective = layer_name
        self.unfreeze_layer([layer_name])
        self.log("obj", self.task_codes[layer_name], prog_bar=True)

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
        # self.switch_random_task()

        model_opt, crit_opt = self.optimizers()
        model_sched, crit_sched = self.lr_schedulers()

        x = batch["data"]
        gt = batch["annotation"]

        out = self(x)
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
        if self.current_objective == "gender_fc":
            loss = self.gender_criterion(out["gender"], gt["gender"])
        elif self.current_objective == "hair_fc":
            loss = self.hair_color_criterion(out["hair"], gt["hair"])
        elif self.current_objective == "glasses_fc":
            loss = self.glasses_criterion(out["glasses"], gt["glasses"])
        elif self.current_objective == "mustache_fc":
            loss = self.mustache_criterion(out["mustache"], gt["mustache"])
        elif self.current_objective == "hat_fc":
            loss = self.hat_criterion(out["hat"], gt["hat"])
        elif self.current_objective == "open_mouth_fc":
            loss = self.open_mouth_criterion(out["open_mouth"], gt["open_mouth"])
        elif self.current_objective == "long_hair_fc":
            loss = self.long_hair_criterion(out["long_hair"], gt["long_hair"])
        elif self.current_objective == "class_fc":
            loss = self.classification_criterion(out["class"], gt["class"])
        else:
            loss = self.config.embedding_loss_rate * self.embed_criterion(out["embedding"], gt["class"])
            loss += (1 - self.config.embedding_loss_rate) * self.classification_criterion(out["class"], gt["class"])

        return loss

    def unfreeze_layer(self, target):
        layers = {name: module for name, module in self.model.named_modules() if '.' not in name or 'head.' in name}
        layers.pop('backbone')
        layers.pop('head')
        layers.pop('')

        for name in layers:
            if name in target:
                print("Turning on", name)
                layers[name].weight.requires_grad_(True)
            else:
                print("Turning off", name)
                layers[name].weight.requires_grad_(False)

        if "head" in target:
            print("Turning on head")
            self.embed_criterion.W.requires_grad_(True)
        else:
            print("Turning off head")
            self.embed_criterion.W.requires_grad_(False)

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

        f1, threshold, tpr = 0., 0., 0.

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
