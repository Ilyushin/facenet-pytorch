import argparse
import os
# from distutils.version import LooseVersion
# from filelock import FileLock

import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import horovod.torch as hvd

from pysistant import helpers
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, training
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--use-mixed-precision', action='store_true', default=False,
                    help='use mixed precision for training')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--data-dir',
                    help='location of the training dataset in the local filesystem (will be downloaded if needed)')


class VGGDataset(torch.utils.data.Dataset):
    def __init__(self, file_names, transform):
        super(VGGDataset, self).__init__()
        self.file_names = file_names
        self.transform = transform

        classes, class_to_idx = self.find_classes()
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path_to_img = self.file_names[idx]
        # print('path_to_img - ', path_to_img)
        raw_image = Image.open(path_to_img)
        x = self.transform(raw_image.convert('RGB'))
        y = path_to_img.split('/')[-2]

        return x, self.class_to_idx[y]

    def find_classes(self):
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        classes = set(sorted(file_path.split('/')[-2] for file_path in self.file_names))
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class in the list of path.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


def accuracy(output, target):
    winners = output.argmax(dim=1)
    corrects = (winners == target)
    accuracy = corrects.cpu().sum().float() / float(target.size(0))

    return accuracy


def get_datasets():
    train_source_files = [item for item in helpers.find_files(train_data_dir, pattern=['.jpg'])]
    train_x, train_y = [], []

    for file_path in train_source_files:
        train_x.append(file_path)
        train_y.append(file_path.split('/')[-2])

    x_train = np.array(train_x)
    x_val = np.array(train_y)

    x_train, x_val, y_train, y_val = train_test_split(x_train, x_val, test_size=0.2, stratify=x_val)

    # print('x_train, x_val - ', len(x_train), len(x_val))

    trans = transforms.Compose([
        # np.float32,
        # transforms.Resize(image_size),
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    train_dataset = VGGDataset(x_train, transform=trans)
    val_dataset = VGGDataset(x_val, transform=trans)

    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs
    )

    # Horovod: use DistributedSampler to partition the test data.
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler, **kwargs
    )

    return train_dataset, train_sampler, val_sampler, train_loader, val_loader


def train(epoch, train_sampler, train_loader, verbose):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        train_accuracy.update(accuracy(output, target))
        loss = loss_fn(output, target)
        train_loss.update(loss)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            if hvd.rank() == 0:
                print(
                    f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_sampler)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {train_loss.avg.item():.2f}\tAcc: {train_accuracy.avg.item():.2f}')

        # t.set_postfix({'loss': train_loss.avg.item(),
        #                'accuracy': 100. * train_accuracy.avg.item()})
        # t.update(1)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def validate(val_loader, verbose):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            val_loss.update(loss_fn(output, target))
            # get the index of the max log-probability
            val_accuracy.update(accuracy(output, target))

            if batch_idx % args.log_interval == 0:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                print(
                    f'Val Epoch: {epoch} [{batch_idx * len(data)}/{len(val_sampler)} ({100. * batch_idx / len(val_loader):.0f}%)]\tLoss: {val_loss.avg.item():.2f}\tAcc: {val_accuracy.avg.item():.2f}')

            # t.set_postfix({'loss': val_loss.avg.item(),
            #                'accuracy': 100. * val_accuracy.avg.item()})
            # t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # train_data_dir = '/mnt/vmk/datasets/faces/vgg_face_2/data/train_cropped'
    # test_data_dir = '/mnt/vmk/datasets/faces/vgg_face_2/data/test_cropped'

    train_data_dir = '/mnt/vmk/datasets/faces/vgg_face_2/data/train_cropped'
    test_data_dir = '/mnt/vmk/datasets/faces/vgg_face_2/data/test_cropped'

    image_size = (256, 256)

    # Horovod: initialize library.
    hvd.init()
    torch.manual_seed(args.seed)

    # Horovod: write TensorBoard logs on first worker.
    log_writer = SummaryWriter('/mnt/vmk/projects/ilyushin/ai-s/facenet_pytorch/logs') if hvd.rank() == 0 else None

    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    train_dataset, train_sampler, val_sampler, train_loader, val_loader = get_datasets()

    model = InceptionResnetV1(
        classify=True,
        num_classes=len(train_dataset.class_to_idx)
    )

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not args.use_adasum else 1

    # Move model to GPU.
    model.cuda()
    # If using GPU Adasum allreduce, scale learning rate by local_size.
    if args.use_adasum and hvd.nccl_built():
        lr_scaler = hvd.local_size()

    # Horovod: scale learning rate by lr_scaler.
    optimizer = optim.Adam(model.parameters(), lr=args.lr * lr_scaler)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if args.use_adasum else hvd.Average,
                                         gradient_predivide_factor=args.gradient_predivide_factor)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    for epoch in range(1, args.epochs + 1):
        train(epoch, train_sampler, train_loader, verbose)
        # Keep test in full precision since computation is relatively light.
        validate(val_loader, verbose)
