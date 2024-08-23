import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import os
import logging
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from data_aug.view_generator import ContrastiveLearningViewGenerator
import matplotlib.pyplot as plt
import numpy as np
import torchvision

# Set the CUDA_VISIBLE_DEVICES environment variable to '1' to select the second device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Function to show an image
def imshow(img):
    #img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show(block=True)
    plt.pause(0.1)

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
        return z

def calculate_info_nce_loss(features, temperature=0.07, batch_size=256):
    batch_size = batch_size
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    # z_i = nn.functional.normalize(z_i, dim=1)
    # z_j = nn.functional.normalize(z_j, dim=1)
    representations = nn.functional.normalize(features, dim=1)
    # representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(similarity_matrix.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(similarity_matrix.device)
    
    return logits, labels

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    # Data augmentation for SimCLR
    # transform_ = transforms.Compose([
    #     transforms.RandomResizedCrop(size=32),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
    #     transforms.RandomGrayscale(p=0.2),
    #     transforms.ToTensor()
    # ])

    transform = ContrastiveLearningViewGenerator(ContrastiveLearningDataset.get_simclr_pipeline_transform(32), 2)

    # Load CIFAR10 dataset
    train_dataset = datasets.CIFAR10(root='../../DJSCC_Pytorch/data', train=True, transform=transform, download=True)
    train_loader_i = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)
    # train_dataset = datasets.CIFAR10(root='../../../data', train=True, transform=transform, download=True)
    # train_loader_j = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)
    # dataset = ContrastiveLearningDataset(root_folder='../../DJSCC_Pytorch/data')
    # train_dataset = dataset.get_dataset('cifar10', 2)
    # train_loader_i = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)
    # Initialize model, optimizer
    base_model = models.resnet18(weights=None)
    model = SimCLR(base_model, out_dim=128).cuda()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    critirion = nn.CrossEntropyLoss().cuda()
    # Get some random training images
    # dataiter = iter(train_loader_i)
    # images, labels = next(dataiter)
    # print(images.shape)

    # Show images
    # imshow(torchvision.utils.make_grid(images))
    # plt.figure()
    # imshow(torchvision.utils.make_grid(images[1]))

    # Tensorboard writer
    writer = SummaryWriter()

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(writer.log_dir, 'training.log')))
    logger.info("Start SimCLR training for 100 epochs.")
    logger.info(f"Training with gpu: {torch.cuda.is_available()}.")

    # Training loop
    scaler = GradScaler()
    num_epochs = 100
    n_iter = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, _ in tqdm(train_loader_i):
            images = torch.cat(images, dim=0)
            images = images.cuda()

            with autocast(enabled=False):
                features = model(images)
                logits, labels = calculate_info_nce_loss(features)  
                loss = critirion(logits, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if n_iter % 10 == 0:
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                writer.add_scalar('loss', loss, global_step=n_iter)
                writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step=n_iter)

            n_iter += 1

        avg_loss = total_loss / len(train_loader_i)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
        writer.add_scalar('Loss/train', avg_loss, epoch)

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join('checkpoints', f'simclr_epoch_{epoch+1}.pth')
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

    writer.close()
    logger.info("Training has finished.")