import torch
import math
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from models.model import TextClassifier
from data_loader import MedicalTextDataset, load_data
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs = 3):
    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, dim = 1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        train_accuracy = correct_predictions / total_predictions

        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = model.state_dict()

        # Save best model
        torch.save(best_model_state, "best_model.pth")

    return best_accuracy

def initialize_optimizer_scheduler(model, train_loader, learning_rate=2e-5, num_epochs=3):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    return optimizer, scheduler

def evaluate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)

    avg_loss = running_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy
