import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import RGCNN_Cls, RGCNN_Seg


SEG_CLASSES = {
    "Earphone": [16, 17, 18],
    "Motorbike": [30, 31, 32, 33, 34, 35],
    "Rocket": [41, 42, 43],
    "Car": [8, 9, 10, 11],
    "Laptop": [28, 29],
    "Cap": [6, 7],
    "Skateboard": [44, 45, 46],
    "Mug": [36, 37],
    "Guitar": [19, 20, 21],
    "Bag": [4, 5],
    "Lamp": [24, 25, 26, 27],
    "Table": [47, 48, 49],
    "Airplane": [0, 1, 2, 3],
    "Pistol": [38, 39, 40],
    "Chair": [12, 13, 14, 15],
    "Knife": [22, 23],
}


def build_segmentation_categories(labels):
    seg = {}
    i = 0
    for _, values in sorted(SEG_CLASSES.items()):
        for value in values:
            seg[value] = i
        i += 1
    cat = np.zeros((labels.shape[0],), dtype=np.int64)
    for idx in range(labels.shape[0]):
        cat[idx] = seg[int(labels[idx][0])]
    return cat


class SegmentationDataset(Dataset):
    def __init__(self, data, labels, categories):
        self.data = data
        self.labels = labels
        self.categories = categories

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.data[idx]).float(),
            torch.from_numpy(self.labels[idx]).long(),
            torch.tensor(self.categories[idx]).long(),
        )


class ClassificationDataset(Dataset):
    def __init__(self, data, labels, num_points=None, resample=False):
        self.data = data
        self.labels = labels
        self.num_points = num_points
        self.resample = resample

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        pts = self.data[idx]
        if self.resample and self.num_points is not None:
            pts = resample_points(pts, self.num_points, np.random)
        return torch.from_numpy(pts).float(), torch.tensor(self.labels[idx]).long()


def load_numpy_split(data_dir, split, limit=None):
    data_path = os.path.join(data_dir, "data_{}.npy".format(split))
    label_path = os.path.join(data_dir, "label_{}.npy".format(split))
    data = np.load(data_path)
    labels = np.load(label_path)
    if limit is not None:
        data = data[:limit]
        labels = labels[:limit]
    return data, labels


def prepare_segmentation_data(data_dir, split, limit=None):
    data, labels = load_numpy_split(data_dir, split, limit=limit)
    categories = build_segmentation_categories(labels)
    return data, labels, categories


def prepare_classification_data(data_dir, split, limit=None):
    data, labels = load_numpy_split(data_dir, split, limit=limit)
    if labels.ndim > 1:
        labels = labels.reshape(-1)
    if labels.shape[0] != data.shape[0]:
        raise ValueError(
            "Expected label_{}.npy to contain one class id per sample.".format(split)
        )
    return data, labels


def evaluate_segmentation(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_points = 0
    with torch.no_grad():
        for points, labels, cats in loader:
            points = points.to(device)
            labels = labels.to(device)
            cats = cats.to(device)
            logits = model(points, cats)
            loss = loss_fn(logits.permute(0, 2, 1), labels)
            preds = logits.argmax(dim=2)
            total_loss += loss.item() * points.size(0)
            total_correct += (preds == labels).sum().item()
            total_points += labels.numel()
    avg_loss = total_loss / max(1, len(loader.dataset))
    accuracy = total_correct / max(1, total_points)
    return avg_loss, accuracy


def evaluate_classification(model, loader, device, loss_fn, reg_weight):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device)
            labels = labels.to(device)
            logits, reg_losses = model(points, None)
            loss = loss_fn(logits, labels)
            if reg_losses and reg_weight:
                reg_term = torch.stack(reg_losses).mean()
                loss = loss + reg_weight * reg_term
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * points.size(0)
            total_correct += (preds == labels).sum().item()
    avg_loss = total_loss / max(1, len(loader.dataset))
    accuracy = total_correct / max(1, len(loader.dataset))
    return avg_loss, accuracy


def train_segmentation(args):
    train_data, train_labels, train_cat = prepare_segmentation_data(
        args.data_dir, "train", limit=args.limit
    )
    val_data, val_labels, val_cat = prepare_segmentation_data(
        args.data_dir, "val", limit=args.limit
    )
    train_loader = DataLoader(
        SegmentationDataset(train_data, train_labels, train_cat),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        SegmentationDataset(val_data, val_labels, val_cat),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = RGCNN_Seg(
        args.num_points,
        F=args.filters,
        K=args.orders,
        M=args.fc_layers,
        regularization=args.regularization,
        dropout=args.dropout,
        batch_size=args.batch_size,
        eval_frequency=args.eval_frequency,
        input_dim=args.input_dim,
        cat_dim=args.cat_dim,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for points, labels, cats in train_loader:
            points = points.to(device)
            labels = labels.to(device)
            cats = cats.to(device)
            optimizer.zero_grad()
            logits = model(points, cats)
            loss = loss_fn(logits.permute(0, 2, 1), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * points.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))
        val_loss, val_acc = evaluate_segmentation(model, val_loader, device, loss_fn)
        elapsed = time.time() - start_time
        print(
            "epoch {}/{} - loss {:.4f} - val_loss {:.4f} - val_acc {:.4f} - {:.1f}s".format(
                epoch, args.epochs, train_loss, val_loss, val_acc, elapsed
            )
        )


def train_classification(args):
    train_data, train_labels = prepare_classification_data(args.data_dir, "train", limit=args.limit)
    val_data, val_labels = prepare_classification_data(args.data_dir, "val", limit=args.limit)
    test_data, test_labels = prepare_classification_data(args.data_dir, "test", limit=args.limit)
    train_loader = DataLoader(
        ClassificationDataset(
            train_data,
            train_labels,
            num_points=args.num_points,
            resample=args.resample_input,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        ClassificationDataset(
            val_data,
            val_labels,
            num_points=args.num_points,
            resample=args.resample_input,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        ClassificationDataset(
            test_data,
            test_labels,
            num_points=args.num_points,
            resample=False,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = RGCNN_Cls(
        args.num_points,
        F=args.filters,
        K=args.orders,
        M=args.fc_layers,
        regularization=args.regularization,
        dropout=args.dropout,
        batch_size=args.batch_size,
        eval_frequency=args.eval_frequency,
        input_dim=args.input_dim,
        cat_dim=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for points, labels in train_loader:
            points = points.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, reg_losses = model(points, None)
            loss = loss_fn(logits, labels)
            if reg_losses and args.reg_weight:
                reg_term = torch.stack(reg_losses).mean()
                loss = loss + args.reg_weight * reg_term
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * points.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))
        val_loss, val_acc = evaluate_classification(model, val_loader, device, loss_fn, args.reg_weight)
        elapsed = time.time() - start_time
        print(
            "epoch {}/{} - loss {:.4f} - val_loss {:.4f} - val_acc {:.4f} - {:.1f}s".format(
                epoch, args.epochs, train_loss, val_loss, val_acc, elapsed
            )
        )
    test_loss, test_acc = evaluate_classification(model, test_loader, device, loss_fn, args.reg_weight)
    print("final test_loss {:.4f} - test_acc {:.4f}".format(test_loss, test_acc))


def parse_args():
    parser = argparse.ArgumentParser(description="Train RGCNN in PyTorch.")
    parser.add_argument("--task", choices=["seg", "cls"], default="seg")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=26)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-9)
    parser.add_argument("--regularization", type=float, default=1e-9)
    parser.add_argument("--dropout", type=float, default=1.0)
    parser.add_argument("--eval-frequency", type=int, default=30)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--input-dim", type=int, default=6)
    parser.add_argument("--cat-dim", type=int, default=16)
    parser.add_argument("--reg-weight", type=float, default=0.0)
    parser.add_argument("--resample-input", action="store_true",
                        help="Randomly resample points on-the-fly to --num-points each epoch.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.task == "seg":
        args.filters = [128, 512, 1024, 512, 128, 50]
        args.orders = [6, 5, 3, 1, 1, 1]
        args.fc_layers = [384, 16, 1]
        train_segmentation(args)
    else:
        args.filters = [128, 512, 1024, 512, 128]
        args.orders = [6, 5, 3, 1, 1]
        args.fc_layers = [512, 256, 40]
        train_classification(args)


if __name__ == "__main__":
    main()
