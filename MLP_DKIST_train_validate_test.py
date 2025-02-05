import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.init as init

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from itertools import product
from joblib import Parallel, delayed
from collections import defaultdict
from collections import Counter

import numpy as np
from astropy.io import fits

class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

#--- paths ----#
output_labels_path = "/data/solar1/rjc/MLP/DKIST_production_combined.npy"
full_data_path = "/data/goose/rjc/MLP/DKIST_4sig_PCA_V_profiles.fits"
model_directory = "/data/solar1/rjc/MLP/DKIST_combined/"
random_split = 42

# Generate all hyperparameter combinations
activation_functions = [Swish(beta=1.0)]
hidden_sizes1 = [128,256,512]
hidden_sizes2 = [32,64,128]
dropout_rates1 = [0.0, 0.3]  # For the first hidden layer
dropout_rates2 = [0.0, 0.1]  # For the second hidden layer
learning_rates = [0.001, 0.005, 0.01, 0.05]
batch_sizes = [128,256,512]
num_epochs_range = range(30, 200, 5) #early stopping means upper range may not be reached in practice
num_iterations = 50
patience_limit = 30

# Load the full dataset
full_data = fits.open(full_data_path, memmap=True)[0].data

# Load the combined indices and labels
combined_data_path = "/data/goose/rjc/MLP/models/production/sets/DKIST_combined_indices_labels.fits"
combined_data = fits.open(combined_data_path)[1].data

# Extract indices and labels
combined_indices = np.array([item[0] for item in combined_data])
combined_labels_str = np.array([item[1] for item in combined_data])

# Encode labels
label_encoder = LabelEncoder()
combined_labels_encoded = label_encoder.fit_transform(combined_labels_str)

# --- Perform a stratified split (70/15/15) ---
# First split: 70% train, 30% remaining (val + test)
X_train_indices, X_temp_indices, y_train_encoded, y_temp_encoded = train_test_split(
    combined_indices, combined_labels_encoded, test_size=0.3, stratify=combined_labels_encoded, random_state=random_split
)

# Second split: 15% validation, 15% test
X_val_indices, X_test_indices, y_val_encoded, y_test_encoded = train_test_split(
    X_temp_indices, y_temp_encoded, test_size=1/3, stratify=y_temp_encoded, random_state=random_split
)

print(f"Combined dataset size: {len(combined_indices)}")
print(f"Training set size: {len(X_train_indices)}")
print(f"Validation set size: {len(X_val_indices)}")
print(f"Test set size: {len(X_test_indices)}")

# --- Load profiles for each split ---
X_train = full_data[X_train_indices, :]
X_val = full_data[X_val_indices, :]
X_test = full_data[X_test_indices, :]

# --- Convert to tensors ---
X_train_tensor = torch.from_numpy(X_train.astype(np.float32)).contiguous()
y_train_tensor = torch.LongTensor(y_train_encoded)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

X_val_tensor = torch.from_numpy(X_val.astype(np.float32)).contiguous()
y_val_tensor = torch.LongTensor(y_val_encoded)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

X_test_tensor = torch.from_numpy(X_test.astype(np.float32)).contiguous()
y_test_tensor = torch.LongTensor(y_test_encoded)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# --- Create DataLoaders --- training set is dynamically loaded so batch_size can be hyperparameter
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print("Loading complete")
    
# Convert label lists to sets
unique_classes_train = set(y_train_encoded)
unique_classes_val = set(y_val_encoded)

# Count the number of unique classes
num_unique_classes_train = len(unique_classes_train)
num_unique_classes_val = len(unique_classes_val)
print(f"Number of unique classes in training set: {num_unique_classes_train}")
print(f"Number of unique classes in validation set: {num_unique_classes_val}")
# Abort if the number of unique classes is not the same
if num_unique_classes_train != num_unique_classes_val:
    raise ValueError("Mismatch in the number of unique classes between training and validation sets. Aborting run.")
else:
    num_labels = num_unique_classes_train
    
input_shape = X_train.shape[1]
print(f"Input shape for the MLP: {input_shape}")
print(f"Data preparation complete. Training samples: {len(X_train_indices)}, Validation samples: {len(X_val_indices)}, Test samples: {len(X_test_indices)}.")

# MLP Architecture
class CustomMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, activation_fn, dropout_rate1, dropout_rate2):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = activation_fn
        self.dropout1 = nn.Dropout(dropout_rate1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = activation_fn
        self.dropout2 = nn.Dropout(dropout_rate2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def weight_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)

def train_and_validate(combination, N, patience):  # N is the number of iterations per hyperparameter set
    index, params = combination
    activation_fn, hidden_size1, hidden_size2, lr, batch_size, num_epochs, dropout_rate1, dropout_rate2 = params

    best_accuracy_val = 0.0
    best_f1_val = 0.0
    best_class_accuracies = {}
    best_model_path = None

    no_improvement_count = 0  # Tracks epochs without improvement

    for iteration in range(N):
        model = CustomMLP(input_shape, hidden_size1, hidden_size2, num_labels, activation_fn, dropout_rate1, dropout_rate2).to(device)
        model.apply(weight_init)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Create the train_loader dynamically
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            accuracy_val = accuracy_score(all_labels, all_preds)
            f1_val = f1_score(all_labels, all_preds, average='macro')

            # ~ print(f"Epoch {epoch + 1}: F1 Score = {f1_val:.4f}, Accuracy = {accuracy_val:.4f}")

            # Check for improvement
            if f1_val > best_f1_val:
                best_f1_val = f1_val
                best_accuracy_val = accuracy_val
                best_class_accuracies = defaultdict(int)
                best_model_path = model_directory + f"model_{index}_act_{activation_fn.__class__.__name__}_h1_{hidden_size1}_h2_{hidden_size2}_lr_{lr}_bs_{batch_size}_ep_{num_epochs}_dr1_{dropout_rate1}_dr2_{dropout_rate2}_f1_{best_f1_val:.4f}_acc_{best_accuracy_val:.4f}.pt"

                print("Saving new best model to:", best_model_path)
                torch.save(model.state_dict(), best_model_path)
                no_improvement_count = 0  # Reset the counter if there's an improvement
            else:
                no_improvement_count += 1

            # Early stopping condition
            if no_improvement_count >= patience:
                # ~ print(f"Early stopping at epoch {epoch + 1} due to no improvement in F1 score for {patience} consecutive epochs.")
                break

    return best_accuracy_val, best_f1_val, params, best_class_accuracies, best_model_path


full_param_combinations = list(enumerate(product(activation_functions, hidden_sizes1, hidden_sizes2, learning_rates, batch_sizes, num_epochs_range, dropout_rates1, dropout_rates2), 1))
indexed_combinations = list(enumerate(full_param_combinations, 1))  # Start indexing from 1
def parallel_execution():
    print("Starting parallel execution with joblib")
    results = Parallel(n_jobs=-1)(delayed(train_and_validate)(params,N=num_iterations,patience=patience_limit) for params in full_param_combinations)
    print("Parallel execution completed")
    return results

# Execute parallel training and validation
results = parallel_execution()

def find_best_hyperparameters(results):
    """Find the best hyperparameter combination based on F1 score."""
    best_f1, best_accuracy, best_model_path, best_params, best_class_accuracies_numeric = 0.0, 0.0, None, None, None
    for accuracy_val, f1_val, params, class_accuracies, model_path in results:
        if f1_val > best_f1:
            best_accuracy = accuracy_val
            best_f1 = f1_val
            best_params = params
            best_class_accuracies_numeric = class_accuracies
            best_model_path = model_path
    return best_f1, best_accuracy, best_model_path, best_params, best_class_accuracies_numeric


def print_best_hyperparameters(best_f1, best_accuracy, best_params, best_class_accuracies_numeric, label_encoder):
    """Print the best hyperparameter combination and class-wise accuracies."""
    print(f"\nBest Hyperparameter Combination:")
    print(f"Activation Function: {best_params[0].__class__.__name__}")
    print(f"Hidden Sizes: {best_params[1]}-{best_params[2]}")
    print(f"Learning Rate: {best_params[3]}")
    print(f"Batch Size: {best_params[4]}")
    print(f"Number of Epochs: {best_params[5]}")
    print(f"Dropout Rate 1: {best_params[6]}")
    print(f"Dropout Rate 2: {best_params[7]}")
    print(f"Best F1 Score: {best_f1:.2f}")
    print(f"Overall Accuracy: {best_accuracy:.2f}")

    if best_class_accuracies_numeric:
        print("Class-wise Accuracies:")
        best_class_accuracies = {
            label_encoder.inverse_transform([class_id])[0]: accuracy
            for class_id, accuracy in best_class_accuracies_numeric.items()
        }
        for class_label, accuracy in best_class_accuracies.items():
            print(f"Class '{class_label}': {accuracy:.2f}")


def predict_class_distribution(model, full_data, label_encoder):
    """Predict class distributions on the full dataset and print percentages."""
    print("Predicting class distributions on the full dataset...")
    model.eval()
    predicted_labels = []
    with torch.no_grad():
        for inputs in DataLoader(torch.from_numpy(full_data.astype(np.float32)).to(device), batch_size=64, pin_memory=True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predicted_labels.extend(preds.cpu().numpy())

    # Decode predictions into string labels
    predicted_labels_decoded = label_encoder.inverse_transform(predicted_labels)

    # Count occurrences of each class
    class_distribution = Counter(predicted_labels_decoded)
    total_predictions = sum(class_distribution.values())

    # Print class distribution percentages
    print("Class distribution percentages:")
    for class_label in label_encoder.classes_:
        class_percentage = (class_distribution.get(class_label, 0) / total_predictions) * 100
        print(f"{class_label}: {class_percentage:.2f}%")

    return predicted_labels_decoded


def save_predictions(predicted_labels, output_path):
    """Save predicted labels to a NumPy file."""
    labels_array = np.array(predicted_labels, dtype="S")
    np.save(output_path, labels_array)
    print(f"Saved predicted labels to {output_path}")
    
def evaluate_on_test_set(model, test_loader):
    """Evaluate the model on the test set and print accuracy and F1 score."""
    print("Evaluating on the test set...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy and F1 score
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    return test_accuracy, test_f1


# Find the best hyperparameters
best_f1, best_accuracy, best_model_path, best_params, best_class_accuracies_numeric = find_best_hyperparameters(results)

# Check if a best model was found
if best_model_path:
    print_best_hyperparameters(best_f1, best_accuracy, best_params, best_class_accuracies_numeric, label_encoder)

    # Load the best model
    best_activation_fn, best_hidden_size1, best_hidden_size2, best_lr, best_batch_size, best_num_epochs, best_dropout_rate1, best_dropout_rate2 = best_params
    model = CustomMLP(input_shape, best_hidden_size1, best_hidden_size2, num_labels, best_activation_fn, best_dropout_rate1, best_dropout_rate2)
    model.load_state_dict(torch.load(best_model_path))

    print("\nEvaluating the model on the test set...")
    evaluate_on_test_set(model, test_loader)

    print("\nClassifying the full dataset...")
    predicted_labels_decoded = predict_class_distribution(model, full_data, label_encoder)
    save_predictions(predicted_labels_decoded, output_labels_path)
else:
    print("No optimal model found.")
