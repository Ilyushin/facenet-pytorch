from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
from pysistant import helpers
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import time

import numpy as np
from sklearn.model_selection import train_test_split

# data_dir = '../data/test_images'
# data_dir = '/mnt/vmk/datasets/faces/casia_webface'

train_data_dir = '/mnt/vmk/datasets/faces/vgg_face_2/data/train_cropped'
test_data_dir = '/mnt/vmk/datasets/faces/vgg_face_2/data/test_cropped'

saved_model_dir = '/mnt/vmk/projects/ilyushin/ai-s/facenet_pytorch/results/models'

helpers.create_dir(saved_model_dir)

batch_size = 512
epochs = 7
workers = 0 if os.name == 'nt' else 8
gpu_ids = [0, 1, 2, 3]

dist.init_process_group(backend='nccl', init_method='env://')
local_rank = 0
torch.cuda.set_device(local_rank)
global_rank = dist.get_rank()


class VGGDataset(torch.utils.data.Dataset):
    def __init__(self, path_list, transform):
        super(VGGDataset, self).__init__()
        self.path_list = path_list
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path_to_img = self.path_list[idx]
        with open(path_to_img, 'rb') as f:
            img = Image.open(f)

        return self.transform(img.convert('RGB'))


def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size


class Logger(object):
    def __init__(self, mode, length, calculate_mean=False):
        self.mode = mode
        self.length = length
        self.calculate_mean = calculate_mean
        if self.calculate_mean:
            self.fn = lambda x, i: x / (i + 1)
        else:
            self.fn = lambda x, i: x

    def __call__(self, loss, metrics, i):
        track_str = '\r{} | {:5d}/{:<5d}| '.format(self.mode, i + 1, self.length)
        loss_str = 'loss: {:9.4f} | '.format(self.fn(loss, i))
        metric_str = ' | '.join('{}: {:9.4f}'.format(k, self.fn(v, i)) for k, v in metrics.items())
        print(track_str + loss_str + metric_str + '   ', end='')
        if i + 1 == self.length:
            print('')


class BatchTimer(object):
    """Batch timing class.
    Use this class for tracking training and testing time/rate per batch or per sample.

    Keyword Arguments:
        rate {bool} -- Whether to report a rate (batches or samples per second) or a time (seconds
            per batch or sample). (default: {True})
        per_sample {bool} -- Whether to report times or rates per sample or per batch.
            (default: {True})
    """

    def __init__(self, rate=True, per_sample=True):
        self.start = time.time()
        self.end = None
        self.rate = rate
        self.per_sample = per_sample

    def __call__(self, y_pred, y):
        self.end = time.time()
        elapsed = self.end - self.start
        self.start = self.end
        self.end = None

        if self.per_sample:
            elapsed /= len(y_pred)
        if self.rate:
            elapsed = 1 / elapsed

        return torch.tensor(elapsed)


def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()


def pass_epoch(
        model, loss_fn, loader, optimizer=None, scheduler=None,
        batch_metrics={'time': BatchTimer()}, show_running=True,
        device='cpu', writer=None
):
    """Train or evaluate over a data epoch.

    Arguments:
        model {torch.nn.Module} -- Pytorch model.
        loss_fn {callable} -- A function to compute (scalar) loss.
        loader {torch.utils.data.DataLoader} -- A pytorch data loader.

    Keyword Arguments:
        optimizer {torch.optim.Optimizer} -- A pytorch optimizer.
        scheduler {torch.optim.lr_scheduler._LRScheduler} -- LR scheduler (default: {None})
        batch_metrics {dict} -- Dictionary of metric functions to call on each batch. The default
            is a simple timer. A progressive average of these metrics, along with the average
            loss, is printed every batch. (default: {{'time': iter_timer()}})
        show_running {bool} -- Whether or not to print losses and metrics for the current batch
            or rolling averages. (default: {False})
        device {str or torch.device} -- Device for pytorch to use. (default: {'cpu'})
        writer {torch.utils.tensorboard.SummaryWriter} -- Tensorboard SummaryWriter. (default: {None})

    Returns:
        tuple(torch.Tensor, dict) -- A tuple of the average loss and a dictionary of average
            metric values across the epoch.
    """

    mode = 'Train' if model.training else 'Valid'
    logger = Logger(mode, length=len(loader), calculate_mean=show_running)
    loss = 0
    metrics = {}

    for i_batch, (x, y) in enumerate(loader):
        x = x.cuda()
        y = y.cuda()
        y_pred = model(x)
        loss_batch = loss_fn(y_pred, y)

        if model.training:
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()

        metrics_batch = {}
        for metric_name, metric_fn in batch_metrics.items():
            metrics_batch[metric_name] = metric_fn(y_pred, y).detach().cpu()
            metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]
            reduce_loss(metrics.get(metric_name, 0), global_rank, len(gpu_ids))

        reduce_loss(loss_batch, global_rank, len(gpu_ids))

        if i_batch % 10 == 0 and global_rank == 0:
            logger(loss_batch.item(), metrics_batch, i_batch)
            # print('Epoch: {} step: {} loss: {}'.format(e, idx, loss.item()))

        # if show_running:
        #     logger(loss.item(), metrics, i_batch)
        # else:
        #     logger(loss_batch, metrics_batch, i_batch)

    if model.training and scheduler is not None:
        scheduler.step()

    loss = loss / (i_batch + 1)
    metrics = {k: v / (i_batch + 1) for k, v in metrics.items()}

    return loss, metrics


train_source_files = [item for item in helpers.find_files(train_data_dir, pattern=['.jpg'])]
train_x, train_y = [], []

for file_path in train_source_files:
    train_x.append(file_path)
    train_y.append(file_path.split('/')[-2])

train_x = np.array(train_x)
train_y = np.array(train_y)

X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y)

print('X_train, X_val - ', len(X_train), len(X_val))

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

train_dataset = VGGDataset(X_train, transform=trans)
val_dataset = VGGDataset(X_val, transform=trans)

sampler = DistributedSampler(train_dataset)

train_loader = DataLoader(
    train_dataset,
    shuffle=False,
    pin_memory=True,
    sampler=sampler
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True
)

resnet = InceptionResnetV1(
    classify=True,
    num_classes=len(train_dataset.class_to_idx)
)

resnet.cuda()
resnet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(resnet)
resnet = DDP(resnet, device_ids=gpu_ids, output_device=local_rank)

optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])

loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': BatchTimer(),
    'acc': accuracy
}

writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

# print('\n\nInitial')
# print('-' * 10)
# resnet.eval()
# training.pass_epoch(
#     resnet, loss_fn, val_loader,
#     batch_metrics=metrics,
#     show_running=True,
#     device=device,
#     writer=writer
# )

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    sampler.set_epoch(epoch)

    resnet.train()
    training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True,
        writer=writer
    )

    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

writer.close()

# Save
torch.save(resnet, os.path.join(saved_model_dir, 'resnet.pt'))
