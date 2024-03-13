import pytorch_lightning as L
import torch
from torch.optim import lr_scheduler, AdamW, SGD
import pytorch_warmup as warmup
import random
from torchmetrics import ROC, F1Score

def _similarity(x, y):
    return torch.dot(x, y) / (torch.linalg.norm(x) * torch.linalg.norm(y))

class LightningWrapper(L.LightningModule):
    def __init__(self, model, config, max_model_lr = 0, min_model_lr = 0,
                 criterion = None, max_crit_lr = 0, min_cri_lr = 0,
                 warmup_epochs = 0, num_batches = 0):
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

        self.task_codes = {"embed_fc": 0, "class_fc": 1,
                           "gender_fc": 10, "hair_fc": 11, "glasses_fc": 12, "mustache_fc": 13, "hat_fc": 14,
                           "open_mouth_fc": 15, "long_hair_fc": 16}
        # self.task_weights = {"embed_fc": 0.5, "class_fc": 0.25,
        #                      "gender_fc": 0.1, "hair_fc": 0.025, "glasses_fc": 0.025, "mustache_fc": 0.025,
        #                      "hat_fc": 0.025, "open_mouth_fc": 0.025, "long_hair_fc": 0.025}
        self.task_weights = {"embed_fc": 0.7, "class_fc": 0.3,
                             "gender_fc": 0., "hair_fc": 0., "glasses_fc": 0., "mustache_fc": 0.,
                             "hat_fc": 0.0, "open_mouth_fc": 0., "long_hair_fc": 0.}
        # self.task_weights = {"embed_fc": 0.2, "class_fc": 0.1,
        #                      "gender_fc": 0.1, "hair_fc": 0.1, "glasses_fc": 0.1, "mustache_fc": 0.1,
        #                      "hat_fc": 0.1, "open_mouth_fc": 0.1, "long_hair_fc": 0.1}
        self.task_rng = random.Random(412)

        self.f1_score = F1Score(task="binary")
        self.roc = ROC(task="binary")

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
        self.switch_random_task()

        model_opt, crit_opt = self.optimizers()
        model_sched, crit_sched = self.lr_schedulers()

        x = batch["image"]
        txt = batch["textual_prompt"]
        gt = batch["annotation"]

        out = self(x, txt)
        loss = self._custom_loss_call(out, gt)

        self.manual_backward(loss)

        # before_model = copy.deepcopy(self.model.model.state_dict())
        # before_criterion = copy.deepcopy(self.criterion.state_dict().copy())
        # before_fc = copy.deepcopy(self.model.embed_fc.state_dict().copy())
        if (batch_idx + 1) % 2 == 0:
            model_opt.step()
            crit_opt.step()
            model_opt.zero_grad()
            crit_opt.zero_grad()
        # after_model = copy.deepcopy(self.model.model.state_dict().copy())
        # after_criterion = copy.deepcopy(self.criterion.state_dict().copy())
        # after_fc = copy.deepcopy(self.model.embed_fc.state_dict().copy())

        # self.check_weights_changed(before_model.items(), after_model.items(), "MODEL")
        # self.check_weights_changed(before_criterion.items(), after_criterion.items(), "CRITERION")
        #
        # if self.current_objective == "embed_fc":
        #     self.check_weights_changed(before_fc.items(), after_fc.items(), "EMBEDDING LAYER")

        with self.model_warmup.dampening():
            if self.model_warmup.last_step + 1 >= (self.num_batches * self.warmup_epochs):
                model_sched.step()

        with self.criterion_warmup.dampening():
            if self.criterion_warmup.last_step + 1 >= (self.num_batches * self.warmup_epochs):
                crit_sched.step()

        if batch_idx == self.num_batches - 1:
            self.losses.append(loss.item())

        self.log("loss", loss, prog_bar=True)
        self.log("lr", model_opt.param_groups[0]['lr'], prog_bar=True)

        return loss

    def check_weights_changed(self, before, after, tgt):
        same = 0
        different = 0
        total_change = 0

        for name_b, param_b in before:
            for name_a, param_a in after:
                if name_b == name_a:
                    if torch.equal(param_b, param_a):
                        print(tgt, " ", name_a)
                        same += 1
                    else:
                        different += 1
                        total_change += torch.sum(torch.abs(param_b - param_a)).item()

        avg_change = total_change / different if different > 0 else 0
        print(different, " had average change ", avg_change, " and ", same, " stayed the same in ", tgt, ".")

    def _custom_loss_call(self, out, gt):
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
            loss = self.criterion(out["embedding"], gt["class"])

        return loss

    def unfreeze_layer(self, target):
        layers = {name: module for name, module in self.model.named_modules() if '.' not in name}
        layers.pop('model')
        layers.pop('')

        if "embed_fc" in target:
            self.criterion.W.requires_grad_(True)
        else:
            self.criterion.W.requires_grad_(False)

        for name in layers:
            if name in target:
                layers[name].weight.requires_grad_(True)
            else:
                layers[name].weight.requires_grad_(False)

    def validation_step(self, batch, _):
        img1 = batch["img1"]
        img2 = batch["img2"]
        label = batch["label"].int()

        embedding1 = self(img1)["embedding"]
        embedding2 = self(img2)["embedding"]

        similarity = torch.stack([_similarity(em1, em2) for em1, em2 in zip(embedding1, embedding2)])

        self.val_sims.extend(similarity)
        self.val_gts.extend(label)

    def on_validation_epoch_end(self):
        all_outputs = torch.tensor(self.val_sims)
        all_targets = torch.tensor(self.val_gts).int()

        fpr, tpr, thresholds = self.roc(all_outputs, all_targets)

        if all(torch.isnan(x) for x in fpr) or all(torch.isnan(x) for x in tpr):
            threshold = thresholds[0]
        else:
            threshold = thresholds[torch.argmin(torch.abs(fpr - (1 - tpr)))]

        all_outputs = torch.tensor([1. if x > threshold else 0. for x in all_outputs])

        self.val_sims = []
        self.val_gts = []

        f1 = self.f1_score(all_outputs, all_targets)
        print("F1: ", f1, " from device ", self.device)
        self.log('acc', f1, prog_bar=True, sync_dist=True)

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

        criterion_optimizer.zero_grad()
        model_optimizer.zero_grad()

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