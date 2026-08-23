import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# 1. Root Directory & Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = ROOT_DIR / "data" / "processed_features"
MODEL_SAVE_PATH = ROOT_DIR / "models" / "temporal_behavior_model.pt"

LABEL_MAP = {"Fighting": 0, "Robbery": 1, "Stealing": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# 2. Simple LSTM / Temporal Model for Keypoint Sequences
class TemporalBehaviorClassifier(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_classes=3, num_layers=2):
        super(TemporalBehaviorClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, 34)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take last frame sequence output
        return out

def prepare_dataset(max_seq_len=30):
    X, y = [], []
    
    for category, label_id in LABEL_MAP.items():
        cat_dir = FEATURES_DIR / category
        if not cat_dir.exists():
            continue

        for feat_file in cat_dir.glob("*_features.json"):
            with open(feat_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            persons = data.get("persons", {})
            for p_id, p_data in persons.items():
                if len(p_data) < 5: # Ignore very short detection snippets
                    continue

                # Extract (x,y) keypoint arrays
                kpt_seq = [frame_data["keypoints"] for frame_data in p_data]
                
                # Pad or truncate sequence to fixed length (e.g., 30 frames)
                if len(kpt_seq) < max_seq_len:
                    padding = [[0.0] * 34] * (max_seq_len - len(kpt_seq))
                    kpt_seq.extend(padding)
                else:
                    kpt_seq = kpt_seq[:max_seq_len]

                X.append(kpt_seq)
                y.append(label_id)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def train_temporal_model():
    print("🚀 Loading Feature Sequences for Temporal Model Training...")
    X, y = prepare_dataset()

    if len(X) == 0:
        print("❌ No valid feature sequences found!")
        return

    print(f"📊 Dataset Shape: Sequences={X.shape[0]}, Frames={X.shape[1]}, Keypoints={X.shape[2]}")

    inputs = torch.tensor(X)
    labels = torch.tensor(y)

    model = TemporalBehaviorClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print("🧠 Training Temporal Behavior Classifier Baseline...")
    model.train()
    for epoch in range(1, 31):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"   • Epoch [{epoch}/30] - Loss: {loss.item():.4f}")

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Model Baseline Saved Successfully to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_temporal_model()