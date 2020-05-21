import torchreid
import torch

torch.cuda.set_device(0)
DATASET_DIR = '../../adaimi'
LOSS = 'contrastive'  # Options softmax, triplet, contrastive
LOG_DIR = f'log/resnet50_{LOSS}'

datamanager = torchreid.data.ImageDataManager(
    # root='reid-data',
    root=DATASET_DIR,
    sources='msmt17',
    targets='msmt17',
    height=256,
    width=128,
    batch_size_train=32,
    # batch_size_train=5,
    batch_size_test=100,
    loss=LOSS
)

model = torchreid.models.build_model(
    name='resnet50',
    num_classes=datamanager.num_train_pids,
    loss=LOSS,
    pretrained=True,
    use_gpu=True
)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

if LOSS == "softmax":
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True,
    )
elif LOSS == "triplet":
    engine = torchreid.engine.ImageTripletEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True,
        margin=10,
        weight_x=0
    )
elif LOSS == "contrastive":
    engine = torchreid.engine.ImageContrastiveEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True,
        margin=10,
        weight_x=0
    )
else:
    raise NotImplementedError

engine.run(
    save_dir=LOG_DIR,
    max_epoch=100,
    eval_freq=10,
    print_freq=100,
    test_only=False
)
