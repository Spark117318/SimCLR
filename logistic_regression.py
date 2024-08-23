import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os

# Define the SimCLR model
class SimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(SimCLR, self).__init__()
        self.encoder = base_model
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_dim)
        )
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z  # Return both the feature and the projection

# Load the pre-trained SimCLR model
base_model = models.resnet18(weights=None)
model = SimCLR(base_model, out_dim=128).cuda()
checkpoint = torch.load('checkpoints/simclr_epoch_100.pth')
print("Checkpoint keys:", checkpoint.keys())
model.load_state_dict(checkpoint)
model.eval()

# Define the Logistic Regression modelpickle
class LogisticRegression(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# Load CIFAR10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
train_dataset = datasets.CIFAR10(root='../../DJSCC_Pytorch/data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='../../DJSCC_Pytorch/data', train=False, transform=transform, download=True)

# Reduce the size of the training dataset
def get_smaller_dataset(dataset, num_samples_per_class):
    targets = torch.tensor(dataset.targets)
    indices = []
    for class_idx in range(10):  # CIFAR10 has 10 classes
        class_indices = (targets == class_idx).nonzero(as_tuple=True)[0]
        indices.extend(class_indices[:num_samples_per_class].tolist())
    return Subset(dataset, indices)

# Specify the number of samples per class
num_samples_per_class = 50
train_dataset = get_smaller_dataset(train_dataset, num_samples_per_class)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
# Extract features
def extract_features(loader, model):
    features = []
    labels = []
    with torch.no_grad():
        for images, target in tqdm(loader):
            images = images.cuda()
            h, _ = model(images)
            features.append(h.cpu())
            labels.append(target)
    return torch.cat(features), torch.cat(labels)

train_features, train_labels = extract_features(train_loader, model)
test_features, test_labels = extract_features(test_loader, model)

# Train Logistic Regression
logistic_model = LogisticRegression(feature_dim=train_features.shape[1], num_classes=10).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(logistic_model.parameters(), lr=3e-3, weight_decay=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)

num_epochs = 1
for epoch in range(num_epochs):
    logistic_model.train()
    optimizer.zero_grad()
    outputs = logistic_model(train_features.cuda())
    loss = criterion(outputs, train_labels.cuda())
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
logistic_model.eval()
with torch.no_grad():
    outputs = logistic_model(test_features.cuda())
    _, predicted = torch.max(outputs.data, 1)
    total = test_labels.size(0)
    correct = (predicted.cpu() == test_labels).sum().item()
    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')