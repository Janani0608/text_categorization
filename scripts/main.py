import os
import torch
import math
from torch.utils.data import DataLoader, random_split
from transformers import DistilBertTokenizer
from models.model import TextClassifier
from data_loader import MedicalTextDataset, load_data
from train import train, initialize_optimizer_scheduler
from sklearn.preprocessing import LabelEncoder
from evaluate import plot_confusion_matrix, print_classification_report

def load_hyperparameters(file_path="best_hyperparameters.json"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        print("No saved hyperparameters found, using default values.")
        return None

def main():
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Load data
    csv_file = '/Users/jananikarthikeyan/Documents/Pet projects/text-categorization/data/raw/mtsamples.csv'
    data_frame = load_data(csv_file)

    # Sample a subset for faster training/evaluation
    data_frame = data_frame.sample(n=100, random_state=42)
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(data_frame['medical_specialty'])
    
    # Create dataset
    dataset = MedicalTextDataset(data_frame, tokenizer, label_encoder)
    
    # Split dataset: 80% train, 20% validation
    train_size = math.floor(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Load saved hyperparameters
    hyperparameters = load_hyperparameters()
    
    # If no saved hyperparameters, use default ones
    if hyperparameters is None:
        learning_rate = 2e-5
        num_epochs = 3
    else:
        learning_rate = hyperparameters.get("learning_rate", 2e-5)
        num_epochs = hyperparameters.get("num_epochs", 3)
    
    # Initialize model
    model = TextClassifier("distilbert-base-uncased", num_classes=len(label_encoder.classes_))
    model.to(device)
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Check if a trained model exists; if not, train the model.
    if os.path.exists("best_model.pth"):
        print("Loading best model from best_model.pth...")
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
    else:
        print("No saved model found. Training the model...")
        optimizer, scheduler = initialize_optimizer_scheduler(model, train_loader, learning_rate=learning_rate, num_epochs=num_epochs)
        best_accuracy = train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)
        print("Best validation accuracy during training:", best_accuracy)
    
    # Evaluate the model on validation data
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Call evaluation functions to generate confusion matrix and classification report
    plot_confusion_matrix(y_true, y_pred, label_encoder.classes_, save_path="confusion_matrix.png")
    print_classification_report(y_true, y_pred, label_encoder.classes_)

if __name__ == "__main__":
    main()
