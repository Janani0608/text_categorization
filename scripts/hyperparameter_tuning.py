import optuna
import torch
import math
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, DistilBertTokenizer
from models.model import TextClassifier
from train import initialize_optimizer_scheduler, train, evaluate
from data_loader import MedicalTextDataset, load_data
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    num_epochs = trial.suggest_int("num_epochs", 1, 3)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    csv_file = '/Users/jananikarthikeyan/Documents/Pet projects/text-categorization/data/raw/mtsamples.csv'
    data_frame = load_data(csv_file)  # Load CSV into DataFrame

    data_frame = data_frame.sample(n=100, random_state=42)

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(data_frame['medical_specialty'])

    # Create dataset
    dataset = MedicalTextDataset(data_frame, tokenizer, label_encoder)

    # Ensure train + val sizes match dataset length
    train_size = math.floor(0.8 * len(dataset))  # Use floor to avoid floating-point issues
    val_size = len(dataset) - train_size  # Ensure the sum is exact

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Load model
    model = TextClassifier("distilbert-base-uncased", num_classes=len(label_encoder.classes_))
    model.to(device)

    optimizer, scheduler = initialize_optimizer_scheduler(model, train_loader, learning_rate = lr, num_epochs = num_epochs)

    criterion = torch.nn.CrossEntropyLoss()

    val_accuracy = train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs = num_epochs)

    return val_accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials = 10)
    # Save the best hyperparameters to a JSON file
    best_trial = study.best_trial
    best_hyperparameters = best_trial.params
    with open("best_hyperparameters.json", "w") as f:
        json.dump(best_hyperparameters, f)

    print("Best trial:")
    print(" Validation Accuracy: {}".format(best_trial.value))
    print(" Best hyperparameters: {}".format(best_trial.params))

