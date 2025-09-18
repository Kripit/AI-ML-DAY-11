
# lstm_energy_prediction.py
# ADVANCED LSTM ENERGY PREDICTION TUTORIAL - EXPLAINED LINE BY LINE
# This is a comprehensive tutorial to learn LSTM from basic to advanced, with industry-relevant techniques

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os 
import warnings
import random 
from datetime import datetime, timedelta 

# creates time-based data for time series, generates timestamps for synthetic data

import itertools # for generating hyperparameter combinations, enables grid search for tuning, optimizes model parameters systematically 
import logging 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

logger.info("All libraries imported successfully!")  # Log successful imports.
logger.info(f"Using PyTorch version: {torch.__version__}")  # Log PyTorch version.
logger.info(f"CUDA available: {torch.cuda.is_available()}")  # Log GPU availability.

class Config:
    def __init__(self):
        # data setting 
        self.sequence_length = 24 # no of past hrs for predictions, captures daily patterns for lstm input 
        self.prediction_steps = 1 # predict one hr ahead 
        self.train_ratio = 0.7 
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
        # model arch
        self.input_features = 10 # feeds diverse data to model, no of input features (inc for complexity)
        self.hidden_size = 128
        self.num_layers = 2 # 2 lstm layers for stacked architecture, stacking layers captures deeper temporal dependencies, improves prediction for complex time series
        self.dropout_rate = 0.3
        self.attention_dim = 64
        
        # training settings 
        self.batch_size = 32
        self.epochs = 30
        self.learning_rate = 0.001 # controls weight update speed, balances learning speed and stability
        self.weight_decay = 1e-5
        # l2 reg, penalizes large weights to prevent overfitting 
        
        # hyper parameter - tuned manually before training to control how the model learns (like learning rate, num layers, max depths etc)
        # hyperparameter tuning = finding the best set of hyperparameters that give you the highest performance on your val/test data
        self.hyperparam_grid = {
            'hidden_size': [128, 256, 512],
            'sequence_length': [12, 24],
            'learning_rate': [0.001, 0.0001, 0.00005],
            'num_layers': [2, 3]
        }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        os.makedirs("models", exist_ok=True)
        
        logger.info(f"Configuration set! Using device: {self.device}")

config = Config()

# STEP 3: DATA GENERATION (Creating realistic synthetic data)
# =============================================================================
# WHY: Generates complex synthetic data to mimic real-world energy consumption.
# WHAT: Creates hourly data with daily/weekly patterns, noise, and simulated outliers.
# HOW: Uses sine functions, random noise, and outlier injection.
# USE CASE: Simulates challenging data for industry-relevant training.

def create_synthetic_energy_data(num_samples=10000):
    logger.info("Creating synthetic energy consumption data")
    
    start_date = datetime(2020, 1, 1)
    
    dates = [start_date + timedelta(hours=i) for i in range(num_samples)] # hourly timestamp
    
    hours = np.arange(num_samples)
    
    daily_pattern = 2.0 + 1.5 * np.sin(2 * np.pi * hours / 24) # 24hour cycle 
    
    # 2.0 = baseline (minimum demand), 1.5 = amplitude
    
    weekly_pattern = 0.5 * np.sin(2 * np.pi * hours / (24 * 7))
    
    noise = np.random.normal(0, 0.3, num_samples) # simulates random fluctuation
    
    outliers = np.random.choice([1, 0], size=num_samples, p=[0.02, 0.98]) * np.random.normal(5, 1, num_samples)
    
    base_consumption = 3.0  # Baseline energy in kW.
    
    energy_consumption = base_consumption + daily_pattern + weekly_pattern + noise + outliers
    
    energy_consumption = np.maximum(energy_consumption, 0.5)

    temperature = 20 + 10 * np.sin(2 * np.pi * hours / (24 * 365)) + np.random.normal(0, 2, num_samples)
        
    humidity = 50 + 20 * np.sin(2 * np.pi * hours / (24 * 365) + np.pi/2) + np.random.normal(0, 5, num_samples)
    
    humidity = np.clip(humidity, 10, 90)  
    
    voltage = 240 + np.random.normal(0, 3, num_samples)
    
    current = energy_consumption * 1000 / voltage
    
    power_factor = 0.85 + np.random.normal(0, 0.05, num_samples)
    
    power_factor = np.clip(power_factor, 0.7, 1.0)
    
    reactive_power = energy_consumption * np.tan(np.arccos(power_factor))
    
    day_of_week = np.array([d.weekday() for d in dates]) 
    
    hour_of_day = np.array([d.hour for d in dates])
    
    is_holiday = np.random.choice([0, 1], size=num_samples, p=[0.95, 0.05])  # 5% chance of holiday.
    
    data = pd.DataFrame({
        'datetime': dates,
        'energy_consumption': energy_consumption,
        'temperature': temperature,
        'humidity': humidity,
        'voltage': voltage,
        'current': current,
        'power_factor': power_factor,
        'reactive_power': reactive_power,
        'day_of_week': day_of_week,
        'hour_of_day': hour_of_day,
        'is_holiday': is_holiday
    })
    # USE CASE: Prepares data for advanced preprocessing and modeling.
    
    logger.info(f"Created {len(data)} samples of synthetic energy data")
    logger.info(f"Data columns: {list(data.columns)}")
    logger.info(f"Data shape: {data.shape}")
    
    return data 
        
data = create_synthetic_energy_data(10000) 

logger.info("\nFirst 5 rows of data:")
logger.info(f"\n{data.head()}")

def preprocess_data(data, config):
    logger.info("Starting data preprocessing...") 
    
    logger.info(f"Missing values before cleaning: {data.isnull().sum().sum()}")
    
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # ffill = forward fill 
    #0    1.0
    #1    1.0   # NaN replaced by previous value 1
    #2    1.0   # NaN replaced by previous value 1
    #3    4.0
    #4    4.0   # NaN replaced by previous value 4
    #5    6.0
    # bfill = backward fill, using both first forward fill if there is still NaN left then backward fill
    
    logger.info(f"Missing values after cleaning: {data.isnull().sum().sum()}")
    
    # outlier detection and capping 
    q1, q3 = data['energy_consumption'].quantile([0.25, 0.75])
    
    # IQR (Interquartile Range) = Q3 - Q1 it tells middle spread of data
    
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    data['energy_consumption'] = data['energy_consumption'].clip(lower=lower_bound, upper=upper_bound)
    
    logger.info(f"Outliers capped for energy_consumption: Lower={lower_bound:.2f}, Upper={upper_bound:.2f}")

    # feature engineering
    data['hour_sin'] = np.sin(2 * np.pi * data['hour_of_day'] / 24)
    
    data['hour_cos'] = np.cos(2 * np.pi * data['hour_of_day'] / 24)
    
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    
    data['energy_lag_1'] = data['energy_consumption'].shift(1)
    
    data['energy_lag_24'] = data['energy_consumption'].shift(24)
    
    data['energy_rolling_mean'] = data['energy_consumption'].rolling(window=24).mean()
    # 24-hour rolling mean.
    
    data = data.dropna() # Remove NaN rows from lag and rolling features
    
    logger.info(f"Data shape after feature engineering: {data.shape}")
    
    feature_columns = [
        'temperature', 'humidity', 'voltage', 'current', 'power_factor', 'reactive_power',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'energy_lag_1', 'energy_lag_24', 'energy_rolling_mean', 'is_holiday'
    ]
    
    logger.info(f"Selected {len(feature_columns)} features: {feature_columns}")
    
    n_samples = len(data)
    train_size = int(n_samples * config.train_ratio)
    val_size = int(n_samples * config.val_ratio)
    train_data = data[:train_size].copy()
    val_data = data[train_size:train_size + val_size].copy()
    test_data = data[train_size + val_size:].copy()
    logger.info(f"Data split - Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
    
    # Data normalization
    feature_scaler = StandardScaler() # Standardize features (mean=0, std=1)
    target_scaler = MinMaxScaler() # Scale target to [0,1]
    X_train = feature_scaler.fit_transform(train_data[feature_columns])
    y_train = target_scaler.fit_transform(train_data[['energy_consumption']]).flatten()
    
    X_val = feature_scaler.transform(val_data[feature_columns])
    y_val = target_scaler.transform(val_data[['energy_consumption']]).flatten()
    X_test = feature_scaler.transform(test_data[feature_columns])
    y_test = target_scaler.transform(test_data[['energy_consumption']]).flatten()
    
    logger.info("Data scaling completed")
    logger.info(f"Feature shape: {X_train.shape}, Target shape: {y_train.shape}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_columns': feature_columns
    }
    
processed_data = preprocess_data(data, config)   

# dataset class formatting data for pytorch 
class EnergyDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        self.sequences = []
        self.targets = []
        
        for i in range(len(X) - sequence_length):
            sequence = X[i:i + sequence_length]
            target = y[i + sequence_length]
            self.sequences.append(sequence)
            self.targets.append(target)
            
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        logger.info(f"Created {len(self.sequences)} sequences with length {sequence_length}")
        logger.info(f"Sequence shape: {self.sequences.shape}")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return sequence, target

class Attention(nn.Module):
    def __init__(self, hidden_size, attention_dim):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        
        self.attention = nn.Linear(hidden_size, attention_dim)
        self.value = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_size)
        attention_scores = self.attention(lstm_output)  # (batch_size, seq_len, attention_dim)
        attention_weights = self.value(attention_scores)  # (batch_size, seq_len, 1)
        attention_weights = self.softmax(attention_weights)  # (batch_size, seq_len, 1)
        context_vector = torch.bmm(attention_weights.transpose(1, 2), lstm_output).squeeze(1)  # (batch_size, hidden_size)
        return context_vector, attention_weights

class EnergyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, attention_dim):
        super(EnergyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # stacked LSTM 
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout_rate)
        
        self.attention = Attention(hidden_size, attention_dim)
        
        # dense layer
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        # WHY: Dense layers refine attention output for final prediction.
        
        logger.info(f"Advanced LSTM created: Input={input_size}, Hidden={hidden_size}, Layers={num_layers}, Attention_dim={attention_dim}")
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Total parameters: {total_params:,}")
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        context_vector, attention_weights = self.attention(lstm_out) 
        
        x = self.dropout(context_vector)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output, attention_weights
    
    def init_weights(self):
        # WHY: Initializes weights for stable training.
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for sequences, targets in train_loader:
        sequences, targets = sequences.to(device), targets.to(device)
        optimizer.zero_grad()
        predictions, _ = model(sequences)  # Ignore attention weights during training.
        predictions = predictions.squeeze()
        loss = criterion(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding gradients.
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    logger.info(f"Train Loss: {avg_loss:.6f}")
    return avg_loss

def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for sequences, targets in val_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            predictions, _ = model(sequences)
            predictions = predictions.squeeze()
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    logger.info(f"Validation Loss: {avg_loss:.6f}")
    return avg_loss               

def evaluate_model(model, test_loader, target_scaler, device):
    model.eval()
    predictions, actuals, attention_weights = [], [], []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs, attn_weights = model(sequences)
            predictions.extend(outputs.squeeze().cpu().numpy())
            actuals.extend(targets.cpu().numpy())
            attention_weights.extend(attn_weights.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    attention_weights = np.array(attention_weights)
    
    # Log attention weights stats
    logger.info(f"Attention weights mean: {np.mean(attention_weights):.4f}, std: {np.std(attention_weights):.4f}")
    
    # Convert to original scale
    predictions_original = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals_original = target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    mse = mean_squared_error(actuals_original, predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_original, predictions_original)
    mape = np.mean(np.abs((actuals_original - predictions_original) / actuals_original)) * 100
    r2 = r2_score(actuals_original, predictions_original)
    
    logger.info("\nMODEL PERFORMANCE METRICS")
    logger.info(f"Mean Squared Error (MSE):        {mse:.6f} kW²")
    logger.info(f"Root Mean Squared Error (RMSE):  {rmse:.6f} kW")
    logger.info(f"Mean Absolute Error (MAE):       {mae:.6f} kW")
    logger.info(f"Mean Absolute Percentage Error:  {mape:.2f}%")
    logger.info(f"R² Score:                        {r2:.6f}")
    
    return {
        'predictions': predictions_original,
        'actuals': actuals_original,
        'attention_weights': attention_weights,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }

def hyperparameter_tuning(processed_data, config):
    best_val_loss = float('inf')
    best_params = None
    best_model_state = None 
    best_results = None
    
    param_combinations = list(itertools.product(
        config.hyperparam_grid['hidden_size'],
        config.hyperparam_grid['sequence_length'],
        config.hyperparam_grid['learning_rate'],
        config.hyperparam_grid['num_layers']
    ))
    
    for hidden_size, sequence_length, learning_rate, num_layers in param_combinations:
        logger.info(f"\nTesting hidden_size={hidden_size}, sequence_length={sequence_length}, learning_rate={learning_rate}, num_layers={num_layers}")
        
        train_dataset = EnergyDataset(processed_data['X_train'], processed_data['y_train'], sequence_length)
        val_dataset = EnergyDataset(processed_data['X_val'], processed_data['y_val'], sequence_length)
        test_dataset = EnergyDataset(processed_data['X_test'], processed_data['y_test'], sequence_length) 
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
        
        # Initializing model
        model = EnergyLSTM(
            input_size=len(processed_data['feature_columns']),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=config.dropout_rate,
            attention_dim=config.attention_dim
        )
        
        model.init_weights()
        model = model.to(config.device)
        
        # defining loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config.weight_decay)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='min', factor=0.5, patience=3, min_lr=1e-6)
        
        train_losses = []
        val_losses = []
        patience_counter = 0
        early_stopping_patience = 15
        min_epochs = 10
        
        for epoch in range(config.epochs):
            logger.info(f"Epoch [{epoch + 1}/{config.epochs}]")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.device)
            val_loss = validate_one_epoch(model, val_loader, criterion, config.device)
            scheduler.step(val_loss)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if val_loss < best_val_loss and epoch >= min_epochs:
                best_val_loss = val_loss
                best_params = {'hidden_size': hidden_size, 'sequence_length': sequence_length, 'learning_rate': learning_rate, 'num_layers': num_layers}
                best_model_state = model.state_dict()
                test_results = evaluate_model(model, test_loader, processed_data['target_scaler'], config.device)
                best_results = test_results
                patience_counter = 0
                logger.info(f"✓ New best model! Validation loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience and epoch >= min_epochs:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            if config.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        logger.info(f"Completed training for hidden_size={hidden_size}, sequence_length={sequence_length}, learning_rate={learning_rate}, num_layers={num_layers}")
    
    torch.save({
        'model_state_dict': best_model_state,
        'params': best_params,
        'val_loss': best_val_loss
    }, 'models/best_energy_lstm_model.pth')
    logger.info(f"Best model saved with params: {best_params}, Val Loss: {best_val_loss:.6f}")
    
    return best_params, best_results, train_losses, val_losses

# Run hyperparameter tuning
best_params, test_results, train_losses, val_losses = hyperparameter_tuning(processed_data, config)
# WHY: Optimizes model for best performance.
# USE CASE: Ensures industry-grade accuracy.

def plot_losses(train_losses, val_losses):
    logger.info("Creating training and validation loss plot...")
    plt.figure(figsize=(10, 5))
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    logger.info("Loss plot saved: loss_curves.png")

# Plot training and validation losses
plot_losses(train_losses, val_losses)

model = EnergyLSTM(
    input_size=len(processed_data['feature_columns']),
    hidden_size=best_params['hidden_size'],
    num_layers=best_params['num_layers'],
    dropout_rate=config.dropout_rate,
    attention_dim=config.attention_dim
)
model.load_state_dict(torch.load('models/best_energy_lstm_model.pth', map_location=config.device)['model_state_dict'])
model = model.to(config.device)
logger.info("Best model reloaded for visualization")

test_dataset = EnergyDataset(processed_data['X_test'], processed_data['y_test'], best_params['sequence_length'])
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

# Re-evaluate for consistency
test_results = evaluate_model(model, test_loader, processed_data['target_scaler'], config.device)

def plot_predictions(results, sequence_length, num_samples=200):
    logger.info("Creating advanced prediction plot...")
    
    predictions = results['predictions'][:num_samples]
    actuals = results['actuals'][:num_samples]
    attention_weights = results['attention_weights'][:num_samples]
    
    plt.figure(figsize=(12, 6))  # Single plot for predictions and attention.
    plt.title('Advanced LSTM: Energy Consumption Predictions', fontsize=14, fontweight='bold')
    
    # Plot predictions vs actuals
    time_index = range(len(predictions))
    plt.plot(time_index, actuals, label='Actual', color='blue', linewidth=1.5)
    plt.plot(time_index, predictions, label='Predicted', color='red', linewidth=1.5)
    plt.xlabel('Time Step')
    plt.ylabel('Energy Consumption (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add metrics
    plt.text(0.02, 0.98, f'MAE: {results["mae"]:.3f} kW\nR²: {results["r2"]:.3f}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue'))
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot attention weights for first sample
    plt.figure(figsize=(10, 4))
    plt.title('Attention Weights for First Test Sample', fontsize=12, fontweight='bold')
    plt.plot(range(sequence_length), attention_weights[0].mean(axis=-1), color='green')
    plt.xlabel('Time Step in Sequence')
    plt.ylabel('Attention Weight')
    plt.grid(True, alpha=0.3)
    plt.savefig('attention_weights.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("Plots saved: prediction_analysis.png, attention_weights.png")

# Create visualization
plot_predictions(test_results, best_params['sequence_length'])

def predict_next_hour(model, recent_data, feature_scaler, target_scaler, device, sequence_length):
    logger.info("Making real-time prediction for next hour...")
    
    if len(recent_data) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} hours, got {len(recent_data)}")
    
    input_sequence = pd.DataFrame(recent_data, columns=processed_data['feature_columns'])
    input_scaled = feature_scaler.transform(input_sequence)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        prediction, _ = model(input_tensor)
        prediction = prediction.squeeze().cpu().numpy()
        
    prediction_original = target_scaler.inverse_transform([[prediction]])[0][0]
    logger.info(f"Predicted energy consumption: {prediction_original:.3f} kW")
    return prediction_original  

logger.info("\nDemonstrating real-time prediction...")
test_features = pd.DataFrame(processed_data['X_test'][-best_params['sequence_length']:], 
                            columns=processed_data['feature_columns'])
test_actual = processed_data['y_test'][-1]
actual_original = processed_data['target_scaler'].inverse_transform([[test_actual]])[0][0]
predicted = predict_next_hour(model, test_features, processed_data['feature_scaler'],
                             processed_data['target_scaler'], config.device, best_params['sequence_length'])
logger.info(f"Actual energy consumption: {actual_original:.3f} kW")
logger.info(f"Prediction error: {abs(predicted - actual_original):.3f} kW "
            f"({abs(predicted - actual_original)/actual_original*100:.1f}%)")

def generate_comprehensive_report(config, best_params, test_results):
    report = f"""
{'='*60}
                ADVANCED LSTM ENERGY PREDICTION REPORT
{'='*60}

1. PROJECT OVERVIEW
{'-'*40}
• Objective: Predict hourly energy consumption using advanced LSTM with attention
• Model Type: Stacked LSTM with Attention Mechanism
• Dataset: Synthetic energy data with outliers and complex features
• Target Variable: Energy consumption (kW)

2. DATA CHARACTERISTICS
{'-'*40}
• Total Samples: {len(data):,}
• Features: {len(processed_data['feature_columns'])}
• Best Sequence Length: {best_params['sequence_length']} hours
• Train/Val/Test Split: {config.train_ratio:.1%}/{config.val_ratio:.1%}/{config.test_ratio:.1%}

3. MODEL ARCHITECTURE
{'-'*40}
• Input Size: {len(processed_data['feature_columns'])} features
• Best Hidden Size: {best_params['hidden_size']} neurons
• LSTM Layers: {best_params['num_layers']}
• Attention Dimension: {config.attention_dim}
• Dropout Rate: {config.dropout_rate}

4. TRAINING CONFIGURATION
{'-'*40}
• Optimizer: AdamW
• Learning Rate: {best_params['learning_rate']}
• Batch Size: {config.batch_size}
• Epochs: {config.epochs}

5. TEST SET PERFORMANCE
{'-'*40}
• RMSE: {test_results['rmse']:.4f} kW
• MAE: {test_results['mae']:.4f} kW
• MAPE: {test_results['mape']:.2f}%
• R² Score: {test_results['r2']:.4f}

6. FILES GENERATED
{'-'*40}
• Model: models/best_energy_lstm_model.pth
• Plots: prediction_analysis.png, attention_weights.png, loss_curves.png
• Log: training.log
"""
    
    with open('comprehensive_report.txt', 'w') as f:
        f.write(report)
    
    logger.info(report)
    logger.info("Comprehensive report saved to 'comprehensive_report.txt'")

# Generate report
generate_comprehensive_report(config, best_params, test_results)

logger.info("\n🎉 CONGRATULATIONS! 🎉")
logger.info("You have completed an advanced LSTM energy prediction project!")
