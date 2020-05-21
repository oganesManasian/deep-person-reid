from __future__ import division, print_function, absolute_import

import torch

from torchreid import metrics
from torchreid.losses import ContrastiveLoss, CrossEntropyLoss

from ..engine import Engine


class ImageContrastiveEngine(Engine):
    r"""Contrastive-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for contrastive loss. Default is 0.3.
        weight_c (float, optional): weight for contrastive loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageContrastiveEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_c=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """

    def __init__(
            self,
            datamanager,
            model,
            optimizer,
            margin=0.3,
            weight_c=1,
            weight_x=1,
            scheduler=None,
            use_gpu=True,
            label_smooth=True
    ):
        super(ImageContrastiveEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        self.weight_c = weight_c
        self.weight_x = weight_x

        self.criterion_c = ContrastiveLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def forward_backward(self, data):
        # batch_imgs, batch_pids = self.parse_data_for_train(data)
        #
        # if self.use_gpu:
        #     for i in [0, 1]:
        #         batch_imgs[i] = batch_imgs[i].cuda()
        #         batch_pids[i] = batch_pids[i].cuda()
        #
        # batch_outputs = []
        # batch_features = []
        # for i in [0, 1]:
        #     outputs, features = self.model(batch_imgs[i])
        #     batch_outputs.append(outputs)
        #     batch_features.append(features)
        #
        # loss_c = self.compute_loss(self.criterion_c, batch_features, batch_pids)
        # batch_loss_x = [self.compute_loss(self.criterion_x, batch_outputs[i], batch_pids[i]) for i in [0, 1]]
        # loss = self.weight_c * loss_c + self.weight_x * (batch_loss_x[0] + batch_loss_x[1])

        for i in [0, 1]:
            data[i] = torch.cat((data[i][0], data[i][1]))
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        outputs, features = self.model(imgs)
        loss_c = self.compute_loss(self.criterion_c, features, pids)
        loss_x = self.compute_loss(self.criterion_x, outputs, pids)
        loss = self.weight_c * loss_c + self.weight_x * loss_x

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_summary = {
            'loss_c': loss_c.item(),
            'loss_x': loss_x.item(),
            'acc': metrics.accuracy(outputs, pids)[0].item()
        }

        return loss_summary
