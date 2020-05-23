import torchreid
import torch

torch.cuda.set_device(0)
LOSS = 'contrastive'  # Options softmax, triplet, contrastive
MODEL_NAME = 'resnet50'
PRETRAINED = True
EPOCHS = 50
DATASET_DIR = '../../adaimi'

LOG_DIR = f'log/{MODEL_NAME}_{LOSS}_pretrained:{PRETRAINED}'
if LOSS == 'softmax':
    train_sampler = 'RandomSampler'
elif LOSS == 'triplet' or LOSS == 'contrastive':
    train_sampler = 'RandomIdentitySampler'
else:
    raise NotImplementedError


datamanager = torchreid.data.ImageDataManager(
    root=DATASET_DIR,
    sources='msmt17',
    targets='msmt17',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    train_sampler=train_sampler
)

model = torchreid.models.build_model(
    name=MODEL_NAME,
    num_classes=datamanager.num_train_pids,
    loss=LOSS,
    pretrained=PRETRAINED,
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
        margin=0.3,
        weight_x=0
    )
elif LOSS == "contrastive":
    engine = torchreid.engine.ImageContrastiveEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True,
        margin=0.3,
        weight_x=0
    )
else:
    raise NotImplementedError

engine.run(
    save_dir=LOG_DIR,
    max_epoch=EPOCHS,
    eval_freq=10,
    print_freq=300,
    test_only=False
)
