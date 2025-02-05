
# MLP Stokes Classifier

This repository contains an MLP-based classifier written in Python for Stokes profiles from the solar telescopes. 

## Features
- Loads preprocessed spectropolarimetric data and a labelled training set from FITS files (note: these must both be prepared and provided by the user)
- Implements a Swish-activated MLP with dropout and Xavier initialization.
- Performs stratified train-validation-test splits (configued to be 70/15/15 split in this case).
- Hyperparameter tuning using over multiple parameters in a user-defined hyperparameter space (you should vary this!).
- Parallelised training/validation using `joblib`. Each hyperparameter combination can be trained N times (there is a stochastic element involved, so N should be > 1)
- Selects and saves the best model based on F1-score (saving the model based on accuracy instead is also possible, just comment/uncomment the necessary function).
- Evaluates on the test set to measure generalisation and ultimately classifies the full dataset.

## Installation
Ensure you have Python 3.x and install the dependencies:

pip install torch numpy astropy scikit-learn joblib


## Usage
### 1. **Prepare**
The version of the code provided was used to classify circular polarisation profiles from DKIST data, but the user can provide data from any telescope/instrument along with a labelled training set with any number of labels.

Update file paths in `MLP_DKIST_train_validate_test.py`:

output_labels_path = "/data/solar1/rjc/MLP/DKIST_production_combined.npy" #(output file) profiles labelled by the MLP
full_data_path = "/data/goose/rjc/MLP/DKIST_4sig_PCA_V_profiles.fits" (input file) profiles from the dataset, which you may have preprocessed to normalise, remove Doppler shifts, remove noise, etc.
model_directory = "/data/solar1/rjc/MLP/DKIST_combined/"
combined_data_path = "/data/goose/rjc/MLP/models/production/sets/DKIST_combined_indices_labels.fits" #(input file) labelled training set

### 2. **Run**
Execute the main script:

`python MLP_DKIST_train_validate_test.py`

## 3. **Results**
After training, the script finds the best hyperparameters and prints:
- Best F1 score & accuracy
- Class-wise accuracies
- Generalisation test scores
- Final model evaluation on the full dataset in `full_data_path`


## Model Architecture
- **Input**: Stokes profiles (i.e. two dimensional array, [number of wavelengths, number of profiles]).
- **Hidden layers**: Two fully connected layers with Swish activation.
- **Optimization**: Adam optimizer with CrossEntropy loss.
- **Output** same shape as the number of labels (i.e. determined automatically by counting the number of unique labels present in the training set)

## Citation
If you use this code, please cite appropriately.

## License
This project is open-source under the MIT License.


