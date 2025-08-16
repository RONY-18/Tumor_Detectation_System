import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from tqdm import tqdm
import pickle
import json

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class BrainMRIDataset(Dataset):
    """Custom Dataset for Multi-class Brain MRI Images"""
    
    def __init__(self, root_dir, transform=None, split='Training'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.images = []
        self.labels = []
        
        # Define classes (make sure the order matches your dataset folders)
        self.classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load images and labels from the specified split (Training/Testing)
        split_path = os.path.join(root_dir, split)
        
        if not os.path.exists(split_path):
            raise ValueError(f"Split directory {split_path} does not exist!")
        
        for class_name in self.classes:
            class_path = os.path.join(split_path, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        img_path = os.path.join(class_path, img_name)
                        self.images.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])
        
        print(f"Found {len(self.images)} images in {split} set")
        
        # Print class distribution
        for class_name in self.classes:
            count = self.labels.count(self.class_to_idx[class_name])
            print(f"{class_name}: {count} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image as grayscale (or RGB if you prefer)
        image = Image.open(img_path).convert('RGB')  # Using RGB for better feature extraction
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class MultiClassBrainTumorCNN(nn.Module):
    """Enhanced CNN Architecture for Multi-class Brain Tumor Classification"""
    
    def __init__(self, num_classes=4):
        super(MultiClassBrainTumorCNN, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # Increased filters
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Second Convolutional Block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Third Convolutional Block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.3)
        
        # Fourth Convolutional Block
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.3)
        
        # Fifth Convolutional Block
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.dropout5 = nn.Dropout2d(0.3)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(1024, 512)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout_fc2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 128)
        self.dropout_fc3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Fourth block
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Fifth block
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        x = self.dropout5(x)
        
        # Global Average Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout_fc3(x)
        x = self.fc4(x)
        
        return x

class MultiClassBrainTumorClassifier:
    """Multi-class Brain Tumor Classifier using PyTorch"""
    
    def __init__(self, img_size=(224, 224), batch_size=32, device=None):
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"Using device: {self.device}")
        print(f"Classes: {self.class_names}")
    
    def get_transforms(self, train=True):
        """Get image transforms for training and validation"""
        if train:
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.3),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def prepare_data(self, data_path, validation_split=0.2, use_test_set=True):
        """Prepare training, validation, and test data loaders"""
        # Load training dataset
        train_dataset = BrainMRIDataset(data_path, transform=self.get_transforms(train=True), 
                                       split='Training')
        
        # Split training data into train and validation
        total_size = len(train_dataset)
        val_size = int(validation_split * total_size)
        train_size = total_size - val_size
        
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Apply different transforms to validation set
        val_dataset = BrainMRIDataset(data_path, transform=self.get_transforms(train=False), 
                                     split='Training')
        val_subset.dataset = val_dataset
        val_subset.indices = val_subset.indices  # Keep the same indices
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_subset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_subset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Load test dataset if available
        if use_test_set:
            try:
                test_dataset = BrainMRIDataset(data_path, transform=self.get_transforms(train=False), 
                                             split='Testing')
                self.test_loader = DataLoader(
                    test_dataset, 
                    batch_size=self.batch_size, 
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True if self.device.type == 'cuda' else False
                )
                print(f"Test samples: {len(test_dataset)}")
            except:
                print("No test set found or error loading test set")
                self.test_loader = None
        
        print(f"Training samples: {len(train_subset)}")
        print(f"Validation samples: {len(val_subset)}")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def build_model(self):
        """Build the CNN model"""
        self.model = MultiClassBrainTumorCNN(num_classes=4).to(self.device)
        
        # Print model architecture
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train_model(self, epochs=100, learning_rate=0.001, patience=15):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        # Define loss function with class weights for imbalanced data
        criterion = nn.CrossEntropyLoss()
        
        # Use different optimizers and schedulers
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                        patience=7, verbose=True)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        print("Starting training...")
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(self.model, self.train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(self.model, self.val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'epoch': epoch,
                    'class_names': self.class_names,
                    'img_size': self.img_size,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_accuracies': self.train_accuracies,
                    'val_accuracies': self.val_accuracies
                }, 'best_multiclass_brain_tumor_model.pth')
                print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {patience} epochs without improvement')
                    break
        
        # Load best model
        checkpoint = torch.load('best_multiclass_brain_tumor_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
    
    def evaluate_model(self, use_test_set=True):
        """Evaluate the model on validation or test set"""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Choose which dataset to evaluate on
        if use_test_set and self.test_loader is not None:
            eval_loader = self.test_loader
            dataset_name = "Test"
        else:
            eval_loader = self.val_loader
            dataset_name = "Validation"
        
        self.model.eval()
        all_preds = []
        all_targets = []
        
        print(f"Evaluating on {dataset_name} set...")
        
        with torch.no_grad():
            for data, targets in tqdm(eval_loader, desc='Evaluating'):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        print(f"{dataset_name} Accuracy: {accuracy:.4f}")
        
        # Classification report
        print(f"\n{dataset_name} Classification Report:")
        print(classification_report(all_targets, all_preds, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'{dataset_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Per-class accuracy
        print(f"\nPer-class Accuracy:")
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name}: {class_accuracies[i]:.4f}")
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.train_losses:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(self.train_losses, label='Training Loss', color='blue')
        axes[0].plot(self.val_losses, label='Validation Loss', color='red')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        axes[1].plot(self.train_accuracies, label='Training Accuracy', color='blue')
        axes[1].plot(self.val_accuracies, label='Validation Accuracy', color='red')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_image(self, image_path):
        """Preprocess a single image for prediction"""
        transform = self.get_transforms(train=False)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor
    
    def predict_single_image(self, image_path, show_image=True):
        """Predict tumor type in a single image"""
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = self.class_names[predicted.item()]
            confidence_score = confidence.item()
        
        # Display results
        if show_image:
            plt.figure(figsize=(15, 6))
            
            # Original image
            plt.subplot(1, 2, 1)
            img = Image.open(image_path)
            plt.imshow(img)
            plt.title('MRI Scan')
            plt.axis('off')
            
            # Prediction probabilities
            plt.subplot(1, 2, 2)
            probs = probabilities[0].cpu().numpy()
            colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow']
            bars = plt.bar(self.class_names, probs, color=colors)
            plt.title(f'Prediction: {predicted_class}\nConfidence: {confidence_score:.2%}')
            plt.ylabel('Probability')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Highlight the predicted class
            bars[predicted.item()].set_color('orange')
            
            plt.tight_layout()
            plt.show()
        
        print(f"Prediction: {predicted_class}")
        print(f"Confidence: {confidence_score:.2%}")
        print("Probabilities:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}: {probabilities[0][i]:.4f}")
        
        return predicted_class, confidence_score, probabilities[0].cpu().numpy()
    
    def predict_batch_images(self, image_folder_path):
        """Predict on multiple images in a folder"""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        image_files = [f for f in os.listdir(image_folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print("No valid image files found in the folder!")
            return
        
        results = []
        print(f"Processing {len(image_files)} images...")
        
        for img_file in tqdm(image_files):
            img_path = os.path.join(image_folder_path, img_file)
            predicted_class, confidence, _ = self.predict_single_image(img_path, show_image=False)
            results.append({
                'image': img_file,
                'prediction': predicted_class,
                'confidence': confidence
            })
        
        # Display results
        print("\nBatch Prediction Results:")
        print("-" * 70)
        print(f"{'Image Name':<30} | {'Prediction':<12} | {'Confidence'}")
        print("-" * 70)
        for result in results:
            print(f"{result['image']:<30} | {result['prediction']:<12} | {result['confidence']:.2%}")
        
        return results
    
    def save_model(self, filepath='multiclass_brain_tumor_classifier.pth'):
        """Save the trained model"""
        if self.model is None:
            print("No model to save!")
            return
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'img_size': self.img_size,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='multiclass_brain_tumor_classifier.pth'):
        """Load a saved model"""
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found!")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model = MultiClassBrainTumorCNN(num_classes=4).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load class names with fallback to default
        self.class_names = checkpoint.get('class_names', ['glioma', 'meningioma', 'no_tumor', 'pituitary'])
        
        # Load image size with fallback to default
        self.img_size = checkpoint.get('img_size', (224, 224))
        
        # Load training history if available
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        print(f"Model loaded from {filepath}")
        print(f"Classes: {self.class_names}")
        print(f"Image size: {self.img_size}")
        return True

def main_menu():
    """Main menu for the multi-class brain tumor classifier"""
    classifier = MultiClassBrainTumorClassifier(img_size=(224, 224), batch_size=16)  # Reduced batch size for stability
    
    # Check if a trained model exists
    model_exists = os.path.exists('best_multiclass_brain_tumor_model.pth')
    
    while True:
        print("\n" + "="*70)
        print("       MULTI-CLASS BRAIN TUMOR MRI CLASSIFIER")
        print("="*70)
        print("1. Train new model")
        print("2. Continue training existing model")
        print("3. Predict single image")
        print("4. Predict batch of images")
        print("5. Evaluate model (Validation set)")
        print("6. Evaluate model (Test set)")
        print("7. Show training history")
        print("8. Save model")
        print("9. Load model")
        print("10. Exit")
        print("="*70)
        
        if model_exists:
            print("✓ Trained model found: best_multiclass_brain_tumor_model.pth")
        else:
            print("✗ No trained model found")
            
        print(f"Current classes: {classifier.class_names}")
        choice = input("\nEnter your choice (1-10): ").strip()
        
        if choice == '1':
            # Train new model
            dataset_path = input("Enter path to your dataset folder (containing Training and Testing folders): ").strip()
            if not os.path.exists(dataset_path):
                print("Dataset path does not exist!")
                continue
            
            epochs = int(input("Enter number of epochs (default 100): ") or "100")
            learning_rate = float(input("Enter learning rate (default 0.001): ") or "0.001")
            
            print("Preparing data...")
            classifier.prepare_data(dataset_path)
            print("Building model...")
            classifier.build_model()
            print("Starting training...")
            classifier.train_model(epochs=epochs, learning_rate=learning_rate)
            classifier.plot_training_history()
            classifier.evaluate_model(use_test_set=False)  # Evaluate on validation first
            model_exists = True
            
        elif choice == '2':
            # Continue training
            if not model_exists:
                print("No existing model found! Please train a new model first.")
                continue
                
            dataset_path = input("Enter path to your dataset folder: ").strip()
            if not os.path.exists(dataset_path):
                print("Dataset path does not exist!")
                continue
                
            epochs = int(input("Enter additional epochs (default 50): ") or "50")
            learning_rate = float(input("Enter learning rate (default 0.0001): ") or "0.0001")
            
            classifier.load_model('best_multiclass_brain_tumor_model.pth')
            classifier.prepare_data(dataset_path)
            classifier.train_model(epochs=epochs, learning_rate=learning_rate)
            classifier.plot_training_history()
            classifier.evaluate_model(use_test_set=False)
            
        # Continuation of the main_menu() function - the missing part

        elif choice == '3':
            # Predict single image
            if not model_exists:
                print("No trained model found! Please train a model first.")
                continue
            
            # Load model if not already loaded
            if classifier.model is None:
                classifier.load_model('best_multiclass_brain_tumor_model.pth')
            
            image_path = input("Enter path to the MRI image: ").strip()
            if not os.path.exists(image_path):
                print("Image path does not exist!")
                continue
            
            try:
                classifier.predict_single_image(image_path, show_image=True)
            except Exception as e:
                print(f"Error predicting image: {e}")
            
        elif choice == '4':
            # Predict batch of images
            if not model_exists:
                print("No trained model found! Please train a model first.")
                continue
            
            # Load model if not already loaded
            if classifier.model is None:
                classifier.load_model('best_multiclass_brain_tumor_model.pth')
            
            folder_path = input("Enter path to folder containing MRI images: ").strip()
            if not os.path.exists(folder_path):
                print("Folder path does not exist!")
                continue
            
            try:
                results = classifier.predict_batch_images(folder_path)
                
                # Save results to CSV
                save_results = input("Save results to CSV? (y/n): ").strip().lower()
                if save_results == 'y':
                    import pandas as pd
                    df = pd.DataFrame(results)
                    csv_filename = f"batch_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    df.to_csv(csv_filename, index=False)
                    print(f"Results saved to {csv_filename}")
                    
            except Exception as e:
                print(f"Error predicting batch: {e}")
            
        elif choice == '5':
            # Evaluate on validation set
            if not model_exists:
                print("No trained model found! Please train a model first.")
                continue
            
            # Load model if not already loaded
            if classifier.model is None:
                classifier.load_model('best_multiclass_brain_tumor_model.pth')
                
            if classifier.val_loader is None:
                dataset_path = input("Enter path to your dataset folder: ").strip()
                if os.path.exists(dataset_path):
                    classifier.prepare_data(dataset_path)
                else:
                    print("Dataset path does not exist!")
                    continue
            
            try:
                classifier.evaluate_model(use_test_set=False)
            except Exception as e:
                print(f"Error evaluating model: {e}")
            
        elif choice == '6':
            # Evaluate on test set
            if not model_exists:
                print("No trained model found! Please train a model first.")
                continue
            
            # Load model if not already loaded
            if classifier.model is None:
                classifier.load_model('best_multiclass_brain_tumor_model.pth')
                
            if classifier.test_loader is None:
                dataset_path = input("Enter path to your dataset folder: ").strip()
                if os.path.exists(dataset_path):
                    classifier.prepare_data(dataset_path)
                else:
                    print("Dataset path does not exist!")
                    continue
            
            try:
                classifier.evaluate_model(use_test_set=True)
            except Exception as e:
                print(f"Error evaluating model: {e}")
            
        elif choice == '7':
            # Show training history
            if not model_exists:
                print("No trained model found! Please train a model first.")
                continue
            
            # Load model if not already loaded
            if classifier.model is None:
                classifier.load_model('best_multiclass_brain_tumor_model.pth')
            
            try:
                classifier.plot_training_history()
            except Exception as e:
                print(f"Error plotting training history: {e}")
            
        elif choice == '8':
            # Save model
            if classifier.model is None:
                print("No model to save! Please train a model first.")
                continue
            
            filename = input("Enter filename to save model (default: multiclass_brain_tumor_classifier.pth): ").strip()
            if not filename:
                filename = 'multiclass_brain_tumor_classifier.pth'
            
            try:
                classifier.save_model(filename)
            except Exception as e:
                print(f"Error saving model: {e}")
            
        elif choice == '9':
            # Load model
            filename = input("Enter filename to load model (default: best_multiclass_brain_tumor_model.pth): ").strip()
            if not filename:
                filename = 'best_multiclass_brain_tumor_model.pth'
            
            try:
                if classifier.load_model(filename):
                    model_exists = True
                    print("Model loaded successfully!")
                else:
                    print("Failed to load model!")
            except Exception as e:
                print(f"Error loading model: {e}")
            
        elif choice == '10':
            print("Thank you for using the Multi-class Brain Tumor Classifier!")
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice! Please enter a number between 1-10.")
        
        input("\nPress Enter to continue...")

def demo_usage():
    """Demo function showing how to use the classifier"""
    print("Demo: Multi-class Brain Tumor Classification")
    print("-" * 50)
    
    # Initialize classifier
    classifier = MultiClassBrainTumorClassifier(img_size=(224, 224), batch_size=16)
    
    # Example dataset path (replace with your actual path)
    dataset_path = "path/to/your/brain_tumor_dataset"
    
    print("1. Preparing data...")
    # classifier.prepare_data(dataset_path)
    
    print("2. Building model...")
    # classifier.build_model()
    
    print("3. Training model...")
    # classifier.train_model(epochs=50, learning_rate=0.001)
    
    print("4. Evaluating model...")
    # classifier.evaluate_model(use_test_set=True)
    
    print("5. Making predictions...")
    # classifier.predict_single_image("path/to/test/image.jpg")
    
    print("Demo completed! Use main_menu() for interactive experience.")

def quick_test(dataset_path, test_image_path=None):
    """Quick test function for fast prototyping"""
    classifier = MultiClassBrainTumorClassifier(img_size=(224, 224), batch_size=8)
    
    print("Loading data...")
    classifier.prepare_data(dataset_path)
    
    print("Building and training model...")
    classifier.build_model()
    classifier.train_model(epochs=5, learning_rate=0.001)  # Quick training for testing
    
    print("Evaluating...")
    classifier.evaluate_model(use_test_set=False)
    
    if test_image_path and os.path.exists(test_image_path):
        print("Testing prediction...")
        classifier.predict_single_image(test_image_path)
    
    return classifier

# Additional utility functions
def analyze_dataset(dataset_path):
    """Analyze the dataset structure and class distribution"""
    print("Dataset Analysis")
    print("=" * 50)
    
    for split in ['Training', 'Testing']:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            print(f"\n{split} Set:")
            total_images = 0
            
            for class_name in ['glioma', 'meningioma', 'no_tumor', 'pituitary']:
                class_path = os.path.join(split_path, class_name)
                if os.path.exists(class_path):
                    count = len([f for f in os.listdir(class_path) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
                    print(f"  {class_name}: {count} images")
                    total_images += count
                else:
                    print(f"  {class_name}: Directory not found")
            
            print(f"  Total: {total_images} images")
        else:
            print(f"\n{split} Set: Directory not found")

def create_sample_predictions_report(classifier, test_images_folder, output_file="predictions_report.html"):
    """Create an HTML report with sample predictions"""
    if classifier.model is None:
        print("Model not loaded!")
        return
     
    import base64
    from io import BytesIO
    
    # Get some sample images
    image_files = [f for f in os.listdir(test_images_folder)[:10]  # First 10 images
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Brain Tumor Classification Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .prediction { border: 1px solid #ddd; margin: 20px 0; padding: 15px; }
            .image { max-width: 300px; }
            .confident { background-color: #e8f5e8; }
            .uncertain { background-color: #fff5e6; }
        </style>
    </head>
    <body>
        <h1>Brain Tumor Classification Report</h1>
        <p>Generated on: """ + str(pd.Timestamp.now()) + """</p>
    """
    
    for img_file in image_files:
        img_path = os.path.join(test_images_folder, img_file)
        try:
            predicted_class, confidence, probabilities = classifier.predict_single_image(img_path, show_image=False)
            
            # Convert image to base64 for embedding
            with open(img_path, "rb") as img_file_obj:
                img_base64 = base64.b64encode(img_file_obj.read()).decode()
            
            confidence_class = "confident" if confidence > 0.8 else "uncertain"
            
            html_content += f"""
            <div class="prediction {confidence_class}">
                <h3>{img_file}</h3>
                <img src="data:image/jpeg;base64,{img_base64}" class="image" alt="{img_file}">
                <p><strong>Prediction:</strong> {predicted_class}</p>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                <p><strong>All Probabilities:</strong></p>
                <ul>
            """
            
            for i, class_name in enumerate(classifier.class_names):
                html_content += f"<li>{class_name}: {probabilities[i]:.4f}</li>"
            
            html_content += "</ul></div>"
            
        except Exception as e:
            html_content += f"<div class='prediction'><h3>{img_file}</h3><p>Error: {e}</p></div>"
    
    html_content += "</body></html>"
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Report saved to {output_file}")

# Main execution
if __name__ == "__main__":
    print("Multi-class Brain Tumor MRI Classifier")
    print("This classifier can distinguish between:")
    print("- Glioma tumors")
    print("- Meningioma tumors") 
    print("- Pituitary tumors")
    print("- No tumor (healthy)")
    print("\nMake sure your dataset has the following structure:")
    print("dataset/")
    print("├── Training/")
    print("│   ├── glioma/")
    print("│   ├── meningioma/")
    print("│   ├── no_tumor/")
    print("│   └── pituitary/")
    print("└── Testing/")
    print("    ├── glioma/")
    print("    ├── meningioma/")
    print("    ├── no_tumor/")
    print("    └── pituitary/")
    
    # You can also analyze your dataset first
    # analyze_dataset("path/to/your/dataset")
    
    # Start the main menu
    main_menu()



