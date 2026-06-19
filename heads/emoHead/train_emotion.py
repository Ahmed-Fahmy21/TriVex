import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# استدعاء المودل بتاعنا
from emoHead import EmotionModel

# ==========================================
# 1. DATASET CLASS (SMART LOAD & EXACT PATHS)
# ==========================================
class MultimodalDataset(Dataset):
    def __init__(self, base_dir, dataset_name, allowed_actors):
        self.base_dir = Path(base_dir)
        self.dataset_name = dataset_name
        
        # رجعنا المسارات الموحدة النظيفة بتاعتك اللي كانت شغالة
        self.face_dir = self.base_dir / "face" / dataset_name
        self.body_dir = self.base_dir / "body" / dataset_name
        self.voice_dir = self.base_dir / "voice" / dataset_name
        self.text_dir = self.base_dir / "text" / dataset_name
            
        self.base_names = []
        
        face_bases = {f.stem for f in self.face_dir.glob("*.pt")}
        body_bases = {f.stem for f in self.body_dir.glob("*.pt")}
        voice_bases = {f.stem for f in self.voice_dir.glob("*.pt")}
        text_bases = {f.stem.replace("_text", "") for f in self.text_dir.glob("*.pt")}
        
        common_bases = face_bases & body_bases & voice_bases & text_bases
        
        for base in common_bases:
            actor_id = self._extract_actor_id(base, dataset_name)
            if actor_id in allowed_actors:
                self.base_names.append(base)
                
        print(f"📂 {dataset_name}: Loaded {len(self.base_names)} complete multi-modal files for {len(allowed_actors)} actors.")
        
    def _extract_actor_id(self, base_name, dataset_name):
        if dataset_name == "CREMA_D":
            return int(base_name.split("_")[0])
        elif dataset_name == "RAVDESS":
            return int(base_name.split("-")[6])
        return -1

    def _get_tensor(self, data, expected_key):
        if torch.is_tensor(data):
            return data
        if expected_key in data:
            return data[expected_key]
        for key, val in data.items():
            if torch.is_tensor(val) and key not in ['label', 'silent_mask']:
                return val
        raise KeyError(f"Could not find any feature tensor. Available keys: {list(data.keys())}")

    def __len__(self):
        return len(self.base_names)

    def __getitem__(self, idx):
        base_name = self.base_names[idx]
        
        standard_file = f"{base_name}.pt"
        text_file = f"{base_name}_text.pt"
        
        face_data = torch.load(self.face_dir / standard_file, weights_only=False)
        face_tensor = self._get_tensor(face_data, 'face_feature')[:32, :1916].float()
        
        body_data = torch.load(self.body_dir / standard_file, weights_only=False)
        body_tensor = self._get_tensor(body_data, 'body_feature')[:32, :].float()
        
        voice_data = torch.load(self.voice_dir / standard_file, weights_only=False)
        voice_tensor = self._get_tensor(voice_data, 'voice_feature').float()
        
        text_path = self.text_dir / text_file
        if not text_path.exists():
            text_path = self.text_dir / standard_file
            
        text_data = torch.load(text_path, weights_only=False)
        text_tensor = self._get_tensor(text_data, 'text_feature').float()
        
        silent_mask = torch.tensor(False, dtype=torch.bool)
        if isinstance(text_data, dict) and 'silent_mask' in text_data:
            silent_mask = text_data['silent_mask']
        
        label = torch.tensor(face_data['label'], dtype=torch.long)
        
        return {
            'body': body_tensor,
            'face': face_tensor,
            'voice': voice_tensor,
            'text': text_tensor,
            'silent_mask': silent_mask,
            'label': label
        }

# ==========================================
# 2. TRAINING LOOP (SOLUTION 1: NATURAL LEARNING + SCHEDULER)
# ==========================================
def train_emotion_model(model, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Training initiated on device: {device}")
    
    model = model.to(device)
    
    # 💡 الحل الأول: رجعنا الـ Loss العادي من غير أوزان عشان المودل ميتجننش
    criterion = nn.CrossEntropyLoss()
    
    # احتفظنا بالـ L2 Regularization (weight_decay) عشان نمنع الـ Overfitting
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=2e-3) 
    
    # 💡 احتفظنا بالـ Scheduler (بيراقب الـ val_acc وبينزل الـ LR لو متحسنتش)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    
    best_val_acc = 0.0 
    target_names = ["neutral_calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
    
    for epoch in range(epochs):
        print(f"\n" + "="*50)
        print(f"🔄 Epoch {epoch+1}/{epochs}")
        print("="*50)
        
        # --- TRAIN ---
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        progress_bar = tqdm(train_loader, desc="[TRAIN]", leave=False)
        
        for batch in progress_bar:
            body = batch['body'].to(device)
            face = batch['face'].to(device)
            voice = batch['voice'].to(device)
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            mask = batch['silent_mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(body, face, voice, text, mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_acc = 100 * correct_train / total_train
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[VALID]", leave=False):
                body = batch['body'].to(device)
                face = batch['face'].to(device)
                voice = batch['voice'].to(device)
                text = batch['text'].to(device)
                labels = batch['label'].to(device)
                mask = batch['silent_mask'].to(device)
                
                outputs = model(body, face, voice, text, mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_acc = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"📈 Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"📉 Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        print("\n📊 Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=target_names, labels=[0, 1, 2, 3, 4, 5, 6], zero_division=0))
        
        print("🧮 Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4, 5, 6]))
        print("-" * 50)
        
        # 💡 تشغيل الـ Scheduler وطباعة الـ LR الحالي
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"⚙️ Current Learning Rate: {current_lr}")
        
        # التسييف على أعلى Accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"🌟 New Best Validation Accuracy ({best_val_acc:.2f}%)! Checkpoint saved as 'best_emotion_model.pth'")
            torch.save(model.state_dict(), "best_emotion_model.pth")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    BASE_DIRECTORY = r"D:\finals\TriVex-main\Data\emoHeadEF"
    
    CREMA_D_SPLITS = {
        "train": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070],
        "val": [1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080],
        "test": [1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091]
    }
    
    RAVDESS_SPLITS = {
        "train": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        "val": [17, 18, 19, 20],
        "test": [21, 22, 23, 24]
    }

    CURRENT_DATASET = "CREMA_D" 
    
    if CURRENT_DATASET == "CREMA_D":
        active_splits = CREMA_D_SPLITS
    else:
        active_splits = RAVDESS_SPLITS

    print("⏳ Initializing Actor-Wise Datasets...")
    
    train_dataset = MultimodalDataset(
        base_dir=BASE_DIRECTORY, 
        dataset_name=CURRENT_DATASET, 
        allowed_actors=active_splits["train"]
    )
    
    val_dataset = MultimodalDataset(
        base_dir=BASE_DIRECTORY, 
        dataset_name=CURRENT_DATASET, 
        allowed_actors=active_splits["val"]
    )
    
    # 💡 استخدمنا num_workers=0 عشان نمنع اختناق الهارد والوقت الطويل
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False) 
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
    
    print("\n🏗️ Building the Emotion Model Architecture...")
    emotion_model = EmotionModel()
    
    train_emotion_model(emotion_model, train_loader, val_loader, epochs=30)
    #(59.22%)