import sys
import os
import math

# --- path setup (supports either layout) ---
# Layout A (this repo): CKAN-Executions-main/scripts/*.py
# Layout B (spec): ROOT/scripts/*.py and ROOT/CKAN-Executions-main/*.py
current_script_folder = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_script_folder)

_candidate_project_folders = [
    workspace_root,
    os.path.join(workspace_root, "CKAN-Executions-main"),
]

project_folder = None
for _p in _candidate_project_folders:
    if os.path.exists(os.path.join(_p, "KANConv.py")):
        project_folder = _p
        sys.path.append(project_folder)
        break

if project_folder is None:
    raise FileNotFoundError(
        "Could not locate 'KANConv.py'. Expected it in either the workspace root or in 'CKAN-Executions-main/'."
    )

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from KANConv import KAN_Convolutional_Layer


class KANC_CIFAR(nn.Module):
    def __init__(self, grid_size=3, num_classes=10):
        super().__init__()
        ## block 1
        self.conv1 = KAN_Convolutional_Layer(
            in_channels=3,
            out_channels=8,
            kernel_size=(3, 3),
            padding=(1, 1),
            grid_size=grid_size,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        ## block 2
        self.conv2 = KAN_Convolutional_Layer(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            padding=(1, 1),
            grid_size=grid_size,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        ## block 3
        self.conv3 = KAN_Convolutional_Layer(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            grid_size=grid_size,
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.flat(x)
        x = self.linear1(x)
        return x


def get_antiblur_transform(epoch, total_epochs):
    ## im doing reverse curriculum: sharp to blurry
    halfway = total_epochs / 2

    if epoch < halfway:
        sigma = 0.0
    else:
        progress = (epoch - halfway) / halfway
        sigma = 4.0 * progress

    t_list = []
    if sigma > 0.1:
        k = int(math.ceil(4 * sigma))
        if k % 2 == 0:
            k += 1
        t_list.append(transforms.GaussianBlur(kernel_size=k, sigma=sigma))

    t_list.append(transforms.ToTensor())
    t_list.append(
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    )
    return transforms.Compose(t_list)


def main():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"SUCCESS: Found GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CRITICAL ERROR: No NVIDIA GPU found!")
        sys.exit()

    data_path = os.path.join(project_folder, "data1")

    ## loading data
    train_set = datasets.CIFAR10(
        data_path, train=True, download=True, transform=transforms.ToTensor()
    )

    model = KANC_CIFAR(grid_size=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("starting anti-biomimetic training (Reverse Curriculum Group)...")
    epochs = 20

    for epoch in range(epochs):
        ## reverse curriculum: sharp then blurred
        train_set.transform = get_antiblur_transform(epoch, epochs)
        loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)

        model.train()
        total_loss = 0

        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(
                    f"epoch {epoch+1} | batch {i+1} | loss: {round(loss.item(), 4)} | reverse curriculum"
                )

        print(f"--- EPOCH {epoch+1} FINISHED | AVG LOSS: {round(total_loss/len(loader), 4)} ---")

    save_file = os.path.join(data_path, "model_c_antibiomimetic.pth")
    torch.save(model.state_dict(), save_file)
    print("saved anti-biomimetic model to:", save_file)


if __name__ == "__main__":
    main()
