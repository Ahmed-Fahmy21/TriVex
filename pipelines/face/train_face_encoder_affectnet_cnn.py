import os
import json
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


LABEL_MAP = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6
}


class AffectNetImageDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None, max_samples=None):
        self.image_root = image_root
        self.transform = transform

        df = pd.read_csv(csv_path)

        if "pth" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV لازم يحتوي على الأعمدة: pth,label")

        df = df.dropna(subset=["pth", "label"]).reset_index(drop=True)
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df["pth"] = df["pth"].astype(str).str.strip().str.replace("\\", "/", regex=False)

        # استبعاد contempt
        df = df[df["label"].isin(LABEL_MAP.keys())].reset_index(drop=True)

        if max_samples is not None:
            df = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)

        samples = []
        for _, row in df.iterrows():
            img_path = os.path.join(image_root, row["pth"])
            if os.path.exists(img_path):
                samples.append((img_path, LABEL_MAP[row["label"]]))

        self.samples = samples
        print(f"[INFO] Loaded {len(self.samples)} valid samples from {image_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class FaceEncoderCNNModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.encoder = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )

        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        z = self.encoder(feats)
        logits = self.head(z)
        return z, logits


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            _, logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / max(total, 1)
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    f1 = f1_score(all_labels, all_preds, average="macro") if len(all_labels) > 0 else 0.0

    return avg_loss, acc, f1


def train(
    labels_csv="labels.csv",
    train_root="Train",
    test_root="Test",
    epochs=10,
    batch_size=32,
    lr=1e-4,
    max_train_samples=None,
    max_test_samples=None,
    encoder_out="face_encoder_128_affectnet_cnn.pth",
    model_out="face_encoder_classifier_affectnet_cnn.pth"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = AffectNetImageDataset(
        csv_path=labels_csv,
        image_root=train_root,
        transform=train_transform,
        max_samples=max_train_samples
    )

    test_dataset = AffectNetImageDataset(
        csv_path=labels_csv,
        image_root=test_root,
        transform=test_transform,
        max_samples=max_test_samples
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = FaceEncoderCNNModel(num_classes=7).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_test_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        all_preds = []
        all_labels = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            _, logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(y.detach().cpu().numpy().tolist())

        train_loss = total_loss / max(total, 1)
        train_acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        train_f1 = f1_score(all_labels, all_preds, average="macro") if len(all_labels) > 0 else 0.0

        test_loss, test_acc, test_f1 = evaluate(model, test_loader, device, criterion)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1": test_f1
        }
        history.append(record)

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"train_f1={train_f1:.4f} "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc:.4f} "
            f"test_f1={test_f1:.4f}"
        )

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1

            torch.save(model.state_dict(), model_out)
            torch.save(model.encoder.state_dict(), encoder_out)

            with open("train_history_affectnet_cnn.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            print(f"[INFO] Saved full model: {model_out}")
            print(f"[INFO] Saved encoder only: {encoder_out}")

    print(f"[DONE] Best test_f1 = {best_test_f1:.4f}")


if __name__ == "__main__":
    train(
        labels_csv="labels.csv",
        train_root="Train",
        test_root="Test",
        epochs=10,
        batch_size=32,
        lr=1e-4,
        max_train_samples=None,
        max_test_samples=None,
        encoder_out="face_encoder_128_affectnet_cnn.pth",
        model_out="face_encoder_classifier_affectnet_cnn.pth"
    )