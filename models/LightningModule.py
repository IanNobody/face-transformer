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
                 warmup_epochs, num_batches, config):
        super(LightningWrapper, self).__init__()

        self.model = model
        self.max_model_lr = max_model_lr
        self.min_model_lr = min_model_lr
        self.model_warmup = None

        self._configure_custom_criterions()
        self.criterion = criterion
        self.max_crit_lr = max_crit_lr
        self.min_crit_lr = min_cri_lr
        self.criterion_warmup = None

        self.warmup_epochs = warmup_epochs

        self.config = config
        self.num_batches = num_batches

        self.val_sims = []
        self.val_gts = []

        self.automatic_optimization = False
        self.current_objective = "embed_fc"

    def _configure_custom_criterions(self):
        self.classification_criterion = torch.nn.CrossEntropyLoss()
        self.hair_color_criterion = torch.nn.CrossEntropyLoss()
        self.glasses_criterion = torch.nn.CrossEntropyLoss()
        self.gender_criterion = torch.nn.BCEWithLogitsLoss()
        self.mustache_criterion = torch.nn.BCEWithLogitsLoss()
        self.hat_criterion = torch.nn.BCEWithLogitsLoss()
        self.open_mouth_criterion = torch.nn.BCEWithLogitsLoss()
        self.long_hair_criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x, txt=None):
        if self.config.model_name == "dat":
            return self.model(x)[0]
        elif self.config.model_name == "multitask_openclip":
            return self.model(x, txt)
        else:
            return self.model(x)

    def on_train_epoch_start(self):
        if self.trainer.current_epoch < self.warmup_epochs:
            self.unfreeze_layer(["embed_fc", "class_fc"])
            self.current_objective = "recognition"
        elif (self.trainer.current_epoch - self.warmup_epochs) % 20 < 15:
            epoch = (self.trainer.current_epoch - self.warmup_epochs) % 14
            if epoch % 14 == 0:
                self.unfreeze_layer(["gender_fc"])
                self.current_objective = "gender"
            elif epoch % 14 in [1, 3, 5, 7, 9, 11, 13]:
                self.unfreeze_layer(["embed_fc", "class_fc"])
                self.current_objective = "recognition"
            elif epoch % 14 == 2:
                self.unfreeze_layer(["hair_fc"])
                self.current_objective = "hair_color"
            elif epoch % 14 == 4:
                self.unfreeze_layer(["glasses_fc"])
                self.current_objective = "glasses"
            elif epoch % 14 == 6:
                self.unfreeze_layer(["mustache_fc"])
                self.current_objective = "mustache"
            elif epoch % 14 == 8:
                self.unfreeze_layer(["hat_fc"])
                self.current_objective = "hat"
            elif epoch % 14 == 10:
                self.unfreeze_layer(["open_mouth_fc"])
                self.current_objective = "open_mouth"
            elif epoch % 14 == 12:
                self.unfreeze_layer(["long_hair_fc"])
                self.current_objective = "long_hair"
        else:
            self.unfreeze_layer(["embed_fc", "class_fc"])
            self.current_objective = "recognition"

    def training_step(self, batch, batch_idx):
        model_opt, crit_opt = self.optimizers()
        model_sched, crit_sched = self.lr_schedulers()

        model_opt.zero_grad()
        crit_opt.zero_grad()

        x = batch["image"]
        txt = batch["textual_prompt"]
        gt = batch["annotation"]

        out = self(x, txt)
        loss = self._custom_loss_call(out, gt)
        self.manual_backward(loss)

        # before = self.model.embed_fc.weight.clone()
        model_opt.step()
        crit_opt.step()
        # after = self.model.embed_fc.weight.clone()

        # compare all elements of the two tensors
        # if torch.equal(before.data, after.data):
        #     print("The weights didnt change")
        # else:
        #     print("The weights changed by average: ", torch.mean(torch.abs(before - after)))

        with self.model_warmup.dampening():
            if self.model_warmup.last_step + 1 >= (self.num_batches * self.warmup_epochs):
                model_sched.step()

        with self.criterion_warmup.dampening():
            if self.criterion_warmup.last_step + 1 >= (self.num_batches * self.warmup_epochs):
                crit_sched.step()

        self.log("loss", loss, prog_bar=True)
        self.log("lr", model_opt.param_groups[0]['lr'], prog_bar=True)

    def _custom_loss_call(self, out, gt):
        loss = self.criterion(out["embedding"], gt["class"]) * 0.79
        loss += self.classification_criterion(out["class"], gt["class"]) * 0.1
        loss += self.gender_criterion(out["gender"], gt["gender"]) * 0.05
        loss += self.hair_color_criterion(out["hair"], gt["hair"]) * 0.01
        loss += self.glasses_criterion(out["glasses"], gt["glasses"]) * 0.01
        loss += self.mustache_criterion(out["mustache"], gt["mustache"]) * 0.01
        loss += self.hat_criterion(out["hat"], gt["hat"]) * 0.01
        loss += self.open_mouth_criterion(out["open_mouth"], gt["open_mouth"]) * 0.01
        loss += self.long_hair_criterion(out["long_hair"], gt["long_hair"]) * 0.01
        return loss

    # def _custom_loss_call(self, out, gt):
    #     if self.current_objective == "gender":
    #         loss = self.gender_criterion(out["gender"], gt["gender"])
    #     elif self.current_objective == "hair_color":
    #         loss = self.hair_color_criterion(out["hair"], gt["hair"])
    #     elif self.current_objective == "glasses":
    #         loss = self.glasses_criterion(out["glasses"], gt["glasses"])
    #     elif self.current_objective == "mustache":
    #         loss = self.mustache_criterion(out["mustache"], gt["mustache"])
    #     elif self.current_objective == "hat":
    #         loss = self.hat_criterion(out["hat"], gt["hat"])
    #     elif self.current_objective == "open_mouth":
    #         loss = self.open_mouth_criterion(out["open_mouth"], gt["open_mouth"])
    #     elif self.current_objective == "long_hair":
    #         loss = self.long_hair_criterion(out["long_hair"], gt["long_hair"])
    #     else:
    #         loss = self.criterion(out["embedding"], gt["class"]) * 0.8
    #         loss += self.classification_criterion(out["class"], gt["class"]) * 0.2
    #
    #     return loss

    def unfreeze_layer(self, target):
        pass
        # layers = {name: module for name, module in self.model.named_modules() if '.' not in name}
        # layers.pop('model')
        # layers.pop('')
        #
        # if "embed_fc" in target:
        #     self.criterion.W.requires_grad_(True)
        # else:
        #     self.criterion.W.requires_grad_(False)
        #
        # for name in layers:
        #     if name in target:
        #         layers[name].weight.requires_grad_(True)
        #     else:
        #         layers[name].weight.requires_grad_(False)

    def validation_step(self, batch, _):
        img1 = batch["img1"]
        img2 = batch["img2"]
        label = batch["label"].cpu()

        embedding1 = self(img1)["embedding"].cpu()
        embedding2 = self(img2)["embedding"].cpu()

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

        self.log('acc', f1_score(all_targets, all_outputs), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        total_batches = self.config.num_of_epoch * self.num_batches
        model_optimizer = AdamW(self.model.parameters(), lr=self.max_model_lr)
        model_scheduler = lr_scheduler.CosineAnnealingLR(model_optimizer,
                                                         total_batches // 5,
                                                         self.min_model_lr)
        criterion_optimizer = SGD(self.criterion.parameters(), lr=self.max_crit_lr)
        criterion_scheduler = lr_scheduler.CosineAnnealingLR(criterion_optimizer,
                                                             total_batches // 5,
                                                             self.min_crit_lr)

        warmup_period = int(self.warmup_epochs * self.num_batches)
        self.model_warmup = warmup.LinearWarmup(model_optimizer, warmup_period=warmup_period)
        self.criterion_warmup = warmup.LinearWarmup(criterion_optimizer, warmup_period=warmup_period)

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