import sys
import os

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
        ## block 1 setup
        self.conv1 = KAN_Convolutional_Layer(
            in_channels=3, out_channels=8, kernel_size=(3,3), padding=(1,1), grid_size=grid_size
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        
        ## block 2 setup
        self.conv2 = KAN_Convolutional_Layer(
            in_channels=8, out_channels=16, kernel_size=(3,3), padding=(1,1), grid_size=grid_size
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        
        ## block 3 setup
        self.conv3 = KAN_Convolutional_Layer(
            in_channels=16, out_channels=32, kernel_size=(3,3), padding=(1,1), grid_size=grid_size
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.flat = nn.Flatten()
        ## 32 channels * 4 * 4 size = 512 inputs
        self.linear1 = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.flat(x)
        x = self.linear1(x)
        return x

def main():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"SUCCESS: Found GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CRITICAL ERROR: No NVIDIA GPU found! Training will be impossible.")
        sys.exit()

    data_path = os.path.join(project_folder, "data1")
    if not os.path.exists(data_path): os.makedirs(data_path)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_set = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)

    model = KANC_CIFAR(grid_size=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("starting standard training (Control Group)...")

    for epoch in range(20): 
        model.train()
        total_loss = 0
        
        ## enumerating batches
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            ## printing every 50 batches
            if (i + 1) % 50 == 0:
                print(f"epoch {epoch+1} | batch {i+1} | current loss: {round(loss.item(), 4)}")
        
        print(f"--- EPOCH {epoch+1} FINISHED | AVG LOSS: {round(total_loss/len(loader), 4)} ---")

    save_file = os.path.join(data_path, "model_a_standard.pth")
    torch.save(model.state_dict(), save_file)
    print("saved control model to:", save_file)

if __name__ == "__main__":
    main()