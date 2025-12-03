import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model import AnomalyClassifier
from torch.utils.data import TensorDataset, DataLoader
import os
import time
import json
from collections import Counter

# Import enhanced trainer if available
try:
    from src.training.enhanced_trainer import EnhancedTrainer
    from src.config_manager import config
    from src.logger import get_logger
    ENHANCED_MODE = True
    logger = get_logger(__name__)
except ImportError:
    ENHANCED_MODE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Wait for features.npz to exist before starting training
features_path = os.path.join(BASE_DIR, 'features.npz')
while not os.path.exists(features_path):
    print("Waiting for features.npz to be created...")
    time.sleep(10)  # Check every 10 seconds
print("features.npz found. Starting training...")

# Load features
try:
    data = np.load(features_path)
    visual = data['visual']
    audio = data['audio']
    labels = data['labels']
    print(f"Successfully loaded features from {features_path}")
    print(f"Visual features shape: {visual.shape}")
    print(f"Audio features shape: {audio.shape}")
    print(f"Labels shape: {labels.shape}")
except Exception as e:
    print(f"Error loading features from {features_path}: {e}")
    exit(1)

# Filter out any None or invalid labels
valid_indices = [i for i, label in enumerate(labels) if label is not None and str(label) != 'nan']
visual = visual[valid_indices]
audio = audio[valid_indices]
labels = labels[valid_indices]

print(f"After filtering: {len(labels)} valid samples")

# Dynamically map all unique labels
unique_labels = sorted(set(labels))
label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
print(f"Label mapping: {label_map}")
print(f"Number of classes: {len(unique_labels)}")

## Convert labels to indices
y = np.array([label_map[l] for l in labels])

# Train/test split
X_vis_train, X_vis_test, X_aud_train, X_aud_test, y_train, y_test = train_test_split(
    visual, audio, y, test_size=0.2, random_state=42, stratify=y)

# Remove duplicate samples between train and test
train_hashes = set([hash(tuple(row)) for row in np.concatenate([X_vis_train, X_aud_train], axis=1)])
test_hashes = set([hash(tuple(row)) for row in np.concatenate([X_vis_test, X_aud_test], axis=1)])
dupe_hashes = train_hashes.intersection(test_hashes)
if dupe_hashes:
    train_mask = [hash(tuple(row)) not in dupe_hashes for row in np.concatenate([X_vis_train, X_aud_train], axis=1)]
    test_mask = [hash(tuple(row)) not in dupe_hashes for row in np.concatenate([X_vis_test, X_aud_test], axis=1)]
    X_vis_train = X_vis_train[train_mask]
    X_aud_train = X_aud_train[train_mask]
    y_train = y_train[train_mask]
    X_vis_test = X_vis_test[test_mask]
    X_aud_test = X_aud_test[test_mask]
    y_test = y_test[test_mask]

# Convert to torch tensors
X_vis_train = torch.tensor(X_vis_train, dtype=torch.float32)
X_vis_test = torch.tensor(X_vis_test, dtype=torch.float32)
X_aud_train = torch.tensor(X_aud_train, dtype=torch.float32)
X_aud_test = torch.tensor(X_aud_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Concatenate visual and audio features
X_train = torch.cat([X_vis_train, X_aud_train], dim=1)
X_test = torch.cat([X_vis_test, X_aud_test], dim=1)

# Oversample minority classes in training set
class_counts = Counter(y_train.numpy())
max_count = max(class_counts.values())
indices = []
for cls, count in class_counts.items():
    cls_indices = np.where(y_train.numpy() == cls)[0]
    if count < max_count:
        oversampled = np.random.choice(cls_indices, max_count - count, replace=True)
        indices.extend(cls_indices.tolist() + oversampled.tolist())
    else:
        indices.extend(cls_indices.tolist())
X_train = X_train[indices]
y_train = y_train[indices]

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model, loss, optimizer
num_classes = len(unique_labels)

if ENHANCED_MODE:
    logger.info("Using enhanced training pipeline")
    model = AnomalyClassifier(visual_dim=X_vis_train.shape[1], audio_dim=X_aud_train.shape[1], hidden_dim=256, num_classes=num_classes)
    trainer = EnhancedTrainer(model)
    
    # Define paths
    model_save_path = os.path.join(BASE_DIR, 'model_weights.pth')
    label_map_path = os.path.join(BASE_DIR, 'label_map.json')
    
    # Use enhanced training
    results = trainer.train(features_path, model_save_path, label_map_path)
    
    # Save training plots
    trainer.save_training_plots()
    
    # Generate confusion matrix
    final_metrics = results['final_metrics']
    trainer.generate_confusion_matrix(
        final_metrics['predictions'], 
        final_metrics['true_labels'], 
        results['inv_label_map']
    )
    
    print(f"Enhanced training completed!")
    print(f"Final Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final Precision: {final_metrics['precision']:.4f}")
    print(f"Final Recall: {final_metrics['recall']:.4f}")
    print(f"Final F1-score: {final_metrics['f1']:.4f}")
    print(f"Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
    exit(0)  # Exit after enhanced training
    
else:
    print("Using basic training pipeline")
    model = AnomalyClassifier(visual_dim=X_vis_train.shape[1], audio_dim=X_aud_train.shape[1], hidden_dim=256, num_classes=num_classes)
    # Use class weights for imbalanced data
    class_weights = torch.tensor([1.0 / class_counts.get(i, 1) for i in range(num_classes)], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Add weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Print feature shapes and label distribution
print(f"Visual feature shape: {visual.shape}")
print(f"Audio feature shape: {audio.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Train labels distribution: {np.bincount(y_train.numpy())}")
print(f"Test labels distribution: {np.bincount(y_test.numpy())}")

# Check for duplicate samples between train and test sets
train_hashes = set([hash(tuple(row)) for row in X_train.numpy()])
test_hashes = set([hash(tuple(row)) for row in X_test.numpy()])
dupes = train_hashes.intersection(test_hashes)
print(f"Number of duplicate samples between train and test: {len(dupes)}")

# Training loop with better progress tracking
num_epochs = 500  # Increased for better training
print(f"Starting training for {num_epochs} epochs...")

# Early stopping parameters
patience = 20
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)

    # Evaluate on validation set every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.eval()
        val_preds = []
        val_true = []
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())

        val_loss = val_loss / len(test_loader.dataset)
        val_accuracy = accuracy_score(val_true, val_preds)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        model.train()
    else:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

# Restore best model
if best_model_state:
    model.load_state_dict(best_model_state)


# Evaluation
model.eval()
preds = []
true = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        preds.extend(predicted.cpu().numpy())
        true.extend(batch_y.cpu().numpy())

# Print predicted class names for first 10 test samples
inv_label_map = {v: k for k, v in label_map.items()}
print("Sample predictions:")
for i in range(min(10, len(preds))):
    print(f"True: {inv_label_map[true[i]]}, Predicted: {inv_label_map[preds[i]]}")

accuracy = accuracy_score(true, preds)
precision = precision_score(true, preds, average='weighted')
recall = recall_score(true, preds, average='weighted')
f1 = f1_score(true, preds, average='weighted')
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")


# Misclassification analysis
cm = confusion_matrix(true, preds)
print("Confusion Matrix:")
print(cm)
misclassified = [(t, p) for t, p in zip(true, preds) if t != p]
print(f"Number of misclassified samples: {len(misclassified)}")
if misclassified:
    print("Misclassified examples (True, Pred):")
    for i in range(min(10, len(misclassified))):
        print(f"True: {inv_label_map[misclassified[i][0]]}, Predicted: {inv_label_map[misclassified[i][1]]}")

# Save metrics to CSV
import csv
metrics_path = os.path.join(BASE_DIR, 'performance_metrics.csv')
with open(metrics_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['Accuracy', accuracy])
    writer.writerow(['Precision', precision])
    writer.writerow(['Recall', recall])
    writer.writerow(['F1-score', f1])
    writer.writerow([])
    writer.writerow(['Confusion Matrix'])
    for row in cm:
        writer.writerow(row)
print(f"Performance metrics saved to {metrics_path}")

# Save trained model weights for inference
model_save_path = os.path.join(BASE_DIR, 'model_weights.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Model weights saved to {model_save_path}")

# Save label_map for inference
label_map_path = os.path.join(BASE_DIR, 'label_map.json')
# Convert keys to int for JSON compatibility
label_map_json = {int(k): int(v) if isinstance(v, (np.integer, int)) else v for k, v in label_map.items()}
with open(label_map_path, 'w') as f:
    json.dump(label_map_json, f)
print(f"Label map saved to {label_map_path}")


# --- Hyperparameter tuning hooks ---
# You can easily tune: batch_size, learning rate, weight_decay, patience, hidden_dim, dropout rates in model.py
# Example: try different values and compare metrics above
