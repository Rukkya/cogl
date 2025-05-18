import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import json
from utils import (
    get_data_loaders, create_model, 
    save_checkpoint, save_training_history
)

def train_model(data_dir, output_dir, num_epochs, learning_rate, batch_size, device_name="cuda"):
    # Setup device
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data loaders
    train_loader, val_loader, _, class_names = get_data_loaders(data_dir)
    num_classes = len(class_names)
    
    # Create model
    model = create_model(num_classes)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create learning rate scheduler with version-compatible parameters
    try:
        # Try with verbose parameter (newer PyTorch versions)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    except TypeError:
        # Fall back to without verbose parameter (older PyTorch versions)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        print("Note: Using ReduceLROnPlateau without verbose parameter due to PyTorch version compatibility")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Create progress bar with compatibility handling
        try:
            train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        except:
            print(f"Epoch {epoch+1}/{num_epochs} [Train] - Starting...")
            train_progress = train_loader
        for inputs, labels, _ in train_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            train_total += inputs.size(0)
            
            # Update progress bar if it's a tqdm object
            if hasattr(train_progress, 'set_postfix'):
                train_progress.set_postfix({
                    'loss': loss.item(), 
                    'acc': (train_correct / train_total).item()
                })
            else:
                # Simple progress indicator if tqdm not available
                if (len(train_loader) >= 10) and (i % (len(train_loader) // 10) == 0):
                    print(f"  Progress: {i}/{len(train_loader)} batches, loss: {loss.item():.4f}, acc: {(train_correct / train_total).item():.4f}")
        
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct.double() / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            # Create progress bar with compatibility handling
            try:
                val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            except:
                print(f"Epoch {epoch+1}/{num_epochs} [Val] - Starting...")
                val_progress = enumerate(val_loader)
                
            for i, (inputs, labels, _) in enumerate(val_progress):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                
                # Update progress bar if it's a tqdm object
                if hasattr(val_progress, 'set_postfix'):
                    val_progress.set_postfix({
                        'loss': loss.item(), 
                        'acc': (val_correct / len(val_loader.dataset)).item()
                    })
                else:
                    # Simple progress indicator if tqdm not available
                    if (len(val_loader) >= 5) and (i % (len(val_loader) // 5) == 0):
                        print(f"  Val Progress: {i}/{len(val_loader)} batches, loss: {loss.item():.4f}, acc: {(val_correct / len(val_loader.dataset)).item():.4f}")
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct.double() / len(val_loader.dataset)
        
        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print learning rate change if it happened
        if epoch > 0 and current_lr != history['learning_rates'][-1]:
            print(f"Learning rate updated to: {current_lr}")
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc.item())
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc.item())
        history['learning_rates'].append(current_lr)
        
        # Save checkpoint if validation accuracy improves
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            checkpoint_path = os.path.join(output_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, best_val_acc, checkpoint_path)
            print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        # Save current model state
        checkpoint_path = os.path.join(output_dir, 'last_model.pth')
        save_checkpoint(model, optimizer, epoch, epoch_val_acc, checkpoint_path)
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, history, class_names

if __name__ == "__main__":
    # Example usage
    data_dir = "path/to/dataset"
    output_dir = "output"
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 32
    
    model, history, class_names = train_model(
        data_dir=data_dir,
        output_dir=output_dir,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
