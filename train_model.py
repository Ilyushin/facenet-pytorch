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


# data_dir = '../data/test_images'
# data_dir = '/mnt/vmk/datasets/faces/casia_webface'

train_data_dir = '/mnt/vmk/datasets/faces/vgg_face_2/data/train_cropped'
test_data_dir = '/mnt/vmk/datasets/faces/vgg_face_2/data/test_cropped'

saved_model_dir = '/mnt/vmk/projects/ilyushin/ai-s/facenet_pytorch/results_plain/models'
saved_checkpoints_dir = '/mnt/vmk/projects/ilyushin/ai-s/facenet_pytorch/results_plain/checkpoints'

helpers.create_dir(saved_model_dir)
helpers.create_dir(saved_checkpoints_dir)


batch_size = 512
epochs = 70
workers = 0 if os.name == 'nt' else 8


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])
dataset = datasets.ImageFolder(train_data_dir, transform=trans)
img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]



train_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)


resnet = InceptionResnetV1(
    classify=True,
    num_classes=len(dataset.class_to_idx)
).to(device)

optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])

loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}

writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

print('\n\nInitial')
print('-' * 10)
resnet.eval()
training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics,
    show_running=True,
    device=device,
    writer=writer
)

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()
    training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    torch.save({
        'epoch': epoch,
        'model_state_dict': resnet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'loss': loss,
    }, os.path.join(saved_checkpoints_dir, f'epoch_{epoch}.tar'))

    # Save
    torch.save(resnet, os.path.join(saved_model_dir, 'resnet.pt'))

writer.close()

# Save
torch.save(resnet, os.path.join(saved_model_dir, 'resnet.pt'))

