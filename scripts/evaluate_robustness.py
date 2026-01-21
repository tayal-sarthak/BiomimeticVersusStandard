import sys
import os
import numpy as np

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
        "Could not locate 'KANConv.py'. Expected it in either the workspace root or in CKAN-Executions-main."
    )

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from KANConv import KAN_Convolutional_Layer

class KANC_CIFAR(nn.Module):
    def __init__(self, grid_size=3, num_classes=10):
        super().__init__()
        # block 1 (fixed padding)
        self.conv1 = KAN_Convolutional_Layer(
            in_channels=3, out_channels=8, kernel_size=(3,3), padding=(1,1), grid_size=grid_size
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        
        # block 2 (fixed padding)
        self.conv2 = KAN_Convolutional_Layer(
            in_channels=8, out_channels=16, kernel_size=(3,3), padding=(1,1), grid_size=grid_size
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        
        # block 3 (fixed padding)
        self.conv3 = KAN_Convolutional_Layer(
            in_channels=16, out_channels=32, kernel_size=(3,3), padding=(1,1), grid_size=grid_size
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.flat(x)
        x = self.linear1(x)
        return x

class CifarCDataset(Dataset):
    def __init__(self, root_dir, corruption, severity=3):
        # looks in data1/CIFAR-10-C/
        data_file = os.path.join(root_dir, corruption + '.npy')
        labels_file = os.path.join(root_dir, 'labels.npy')
        
        data = np.load(data_file)
        labels = np.load(labels_file)
        
        start = (severity - 1) * 10000
        end = severity * 10000
        
        self.data = data[start:end]
        self.labels = labels[start:end]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        img = self.transform(img)
        return img, self.labels[idx]

def evaluate(model, device, data_path, corruption):
    dataset = CifarCDataset(data_path, corruption, severity=3)
    loader = DataLoader(dataset, batch_size=100, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            
            output = model(imgs)
            pred = output.argmax(dim=1)
            
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
    return 100 * correct / total

def main():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"SUCCESS: Found GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU found. Testing will be slow.")
        device = "cpu"

    data_path = os.path.join(project_folder, "data1")
    cifar_c_path = os.path.join(data_path, "CIFAR-10-C")
    
    model_a_file = os.path.join(data_path, "model_a_standard.pth")
    model_b_file = os.path.join(data_path, "model_b_biomimetic.pth")
    model_c_file = os.path.join(data_path, "model_c_antibiomimetic.pth")
    
    if not os.path.exists(cifar_c_path):
        print("ERROR: Could not find CIFAR-10-C folder at:", cifar_c_path)
        return

    required_files = ["labels.npy", "snow.npy", "glass_blur.npy", "defocus_blur.npy", "fog.npy"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(cifar_c_path, f))]
    if missing:
        print("ERROR: Missing CIFAR-10-C files:", ", ".join(missing))
        return

    model_a = KANC_CIFAR(grid_size=3).to(device)
    model_b = KANC_CIFAR(grid_size=3).to(device)
    model_c = None

    try:
        model_a.load_state_dict(torch.load(model_a_file, map_location=device), strict=True)
        model_b.load_state_dict(torch.load(model_b_file, map_location=device), strict=True)
        if os.path.exists(model_c_file):
            model_c = KANC_CIFAR(grid_size=3).to(device)
            model_c.load_state_dict(torch.load(model_c_file, map_location=device), strict=True)
    except FileNotFoundError:
        print("ERROR: Could not find the trained model files in 'data1'.")
        print("Expected:", model_a_file)
        print("Expected:", model_b_file)
        return
    except RuntimeError as e:
        print("ERROR: Model weights did not match the architecture.")
        print("Details:", str(e))
        return
    
    model_a.eval()
    model_b.eval()
    if model_c is not None:
        model_c.eval()

    tests = ['snow', 'glass_blur', 'defocus_blur', 'fog']
    
    print("\n--- ROBUSTNESS EXAM RESULTS ---")
    if model_c is None:
        print("corruption      | standard   | biomimetic")
    else:
        print("corruption      | standard   | biomimetic | anti-biomimetic")
    print("-" * 45)

    rows = []
    for c in tests:
        acc_a = evaluate(model_a, device, cifar_c_path, c)
        acc_b = evaluate(model_b, device, cifar_c_path, c)
        if model_c is None:
            rows.append((c, acc_a, acc_b))
        else:
            acc_c = evaluate(model_c, device, cifar_c_path, c)
            rows.append((c, acc_a, acc_b, acc_c))

        name = c.ljust(15)
        if model_c is None:
            print(f"{name} | {round(acc_a, 1)}%     | {round(acc_b, 1)}%")
        else:
            print(f"{name} | {round(acc_a, 1)}%     | {round(acc_b, 1)}%     | {round(acc_c, 1)}%")

    mean_a = sum(r[1] for r in rows) / len(rows)
    mean_b = sum(r[2] for r in rows) / len(rows)
    mean_c = None
    if model_c is not None:
        mean_c = sum(r[3] for r in rows) / len(rows)
    print("-" * 45)
    if model_c is None:
        print(f"mean (4 corrupt) | {round(mean_a, 2)}%    | {round(mean_b, 2)}%")
    else:
        print(f"mean (4 corrupt) | {round(mean_a, 2)}%    | {round(mean_b, 2)}%    | {round(mean_c, 2)}%")

    if model_c is None:
        if mean_a > mean_b:
            winner = "Standard (Twin A)"
        elif mean_b > mean_a:
            winner = "Biomimetic (Twin B)"
        else:
            winner = "Tie"
    else:
        means = {
            "Standard (Twin A)": mean_a,
            "Biomimetic (Twin B)": mean_b,
            "Anti-biomimetic (Twin C)": mean_c,
        }
        max_mean = max(means.values())
        winners = [k for k, v in means.items() if v == max_mean]
        winner = winners[0] if len(winners) == 1 else "Tie"

    print("WINNER:", winner)

if __name__ == "__main__":
    main()