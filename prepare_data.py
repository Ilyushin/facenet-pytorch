from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


# data_dir = '../data/test_images'
# data_dir = '/mnt/vmk/datasets/faces/casia_webface'

# train_data_dir = '/mnt/vmk/datasets/faces/vgg_face_2/data/train'
# test_data_dir = '/mnt/vmk/datasets/faces/vgg_face_2/data/test'

train_data_dir = '/home/datasets/faces/vgg_face_2/data/train'
test_data_dir = '/home/datasets/faces/vgg_face_2/data/test'


batch_size = 2048
epochs = 8
workers = 0 if os.name == 'nt' else 8


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# train_dataset = datasets.ImageFolder(train_data_dir, transform=transforms.Resize((512, 512)))
# train_dataset.samples = [
#     (p, p.replace(train_data_dir, train_data_dir + '_cropped'))
#     for p, _ in train_dataset.samples
# ]
#
# train_loader = DataLoader(
#     train_dataset,
#     num_workers=workers,
#     batch_size=batch_size,
#     collate_fn=training.collate_pil
# )
#
# for i, (x, y) in enumerate(train_loader):
#     mtcnn(x, save_path=y)
#     print('\rBatch {} of {}'.format(i + 1, len(train_loader)), end='')

test_dataset = datasets.ImageFolder(test_data_dir, transform=transforms.Resize((512, 512)))
test_dataset.samples = [
    (p, p.replace(test_data_dir, test_data_dir + '_cropped'))
    for p, _ in test_dataset.samples
]

test_loader = DataLoader(
    test_dataset,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)

for i, (x, y) in enumerate(test_loader):
    mtcnn(x, save_path=y)
    print('\rBatch {} of {}'.format(i + 1, len(test_loader)), end='')

print('\rFinished data cropping')

# Remove mtcnn to reduce GPU memory usage
del mtcnn
