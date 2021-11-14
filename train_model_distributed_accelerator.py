from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import os
from pysistant import helpers
from PIL import Image
from accelerate import Accelerator

import numpy as np
from sklearn.model_selection import train_test_split

# data_dir = '../data/test_images'
# data_dir = '/mnt/vmk/datasets/faces/casia_webface'

train_data_dir = '/mnt/vmk/datasets/faces/vgg_face_2/data/train_cropped'
test_data_dir = '/mnt/vmk/datasets/faces/vgg_face_2/data/test_cropped'

saved_model_dir = '/mnt/vmk/projects/ilyushin/ai-s/facenet_pytorch/results_distr_acc/models'
saved_checkpoints_dir = '/mnt/vmk/projects/ilyushin/ai-s/facenet_pytorch/results_distr_acc/checkpoints'

helpers.create_dir(saved_model_dir)
helpers.create_dir(saved_checkpoints_dir)

batch_size = 700
image_size = (256, 256)
epochs = 70
num_workers = 0 if os.name == 'nt' else 8

accelerator = Accelerator()


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


def get_datasets():
    train_source_files = [item for item in helpers.find_files(train_data_dir, pattern=['.jpg'])]
    train_x, train_y = [], []

    for file_path in train_source_files:
        train_x.append(file_path)
        train_y.append(file_path.split('/')[-2])

    x_train = np.array(train_x)
    train_y = np.array(train_y)

    x_train, x_val, y_train, y_val = train_test_split(x_train, train_y, test_size=0.2, stratify=train_y)

    trans = transforms.Compose([
        # np.float32,
        # transforms.Resize(image_size),
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    train_dataset = VGGDataset(x_train, transform=trans)
    val_dataset = VGGDataset(x_val, transform=trans)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_dataset, train_loader, val_loader


# def accuracy(logits, y):
#     _, preds = torch.max(logits, 1)
#     return (preds == y).float().mean()


train_dataset, train_loader, val_loader = get_datasets()

resnet = InceptionResnetV1(
    classify=True,
    num_classes=len(train_dataset.class_to_idx)
)

optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])

loss_fn = torch.nn.CrossEntropyLoss()

resnet, optimizer, train_loader, val_loader = accelerator.prepare(resnet, optimizer, train_loader, val_loader)
steps = len(train_dataset) // (batch_size*8)

for epoch in range(epochs):
    accelerator.print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    accelerator.print('-' * 10)

    accuracy = 0
    num_elems = 0
    resnet.train()
    for step, (x, y) in enumerate(train_loader):
        # x = x.to(accelerator.device)
        # y = y.to(accelerator.device)
        y_pred = resnet(x)
        loss_batch = loss_fn(y_pred, y)
        accelerator.backward(loss_batch)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # loss_batch = loss_batch.detach().cpu()
        #
        # predictions = y_pred.argmax(dim=-1)
        # accurate_preds = accelerator.gather(predictions) == accelerator.gather(y)
        # num_elems += accurate_preds.shape[0]
        # accuracy += accurate_preds.long().sum()

        # if step % 100 == 0:
        #     accelerator.print(
        #         f"Train epoch {epoch}/{epochs}, step {step}/{steps}: loss {loss_batch.item():.4f}, accuracy {100 * accuracy.item() / num_elems:.2f}"
        #     )

    resnet.eval()
    loss = 0
    accuracy = 0
    num_elems = 0
    for step, (x, y) in enumerate(val_loader):
        with torch.no_grad():
            y_pred = resnet(x)

        predictions = y_pred.argmax(dim=-1)
        accuracy_preds = accelerator.gather(predictions) == accelerator.gather(y)
        num_elems += accuracy_preds.shape[0]
        accuracy += accuracy_preds.long().sum()

    eval_metric = accuracy.item() / num_elems
    # eval_loss = loss.item() / num_elems
    # Use accelerator.print to print only on the main process.
    accelerator.print('-' * 10)
    # accelerator.print(f"Eval epoch {epoch} from {epochs}: loss {eval_loss:.4f}, accuracy {100 * eval_metric:.2f}")
    accelerator.print(f"Eval epoch {epoch+1} from {epochs}: accuracy {100 * eval_metric:.2f}")

    accelerator.save({
        'epoch': epoch,
        'model_state_dict': resnet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(saved_checkpoints_dir, f'epoch_{epoch}.tar'))

# Save
accelerator.save(resnet, os.path.join(saved_model_dir, 'resnet.pt'))
