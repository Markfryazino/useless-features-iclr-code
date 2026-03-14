import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformer_lens import HookedTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from datasets import load_dataset

class Config:
    def __init__(self):
        # Model settings
        self.model_name = "gemma-2-2b"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.layer = 15  # The layer we want to analyze
        
        # Dataset settings
        self.dataset_name = "NeelNanda/pile-10k"
        self.max_samples = 1000  # Number of sequences to process
        self.sequence_length = 512  # Fixed sequence length
        self.seed = 42
        
        # Training settings
        self.batch_size = 32
        self.train_ratio = 0.8
        self.learning_rate = 1e-4
        self.num_epochs = 10
        
        # Lookahead settings
        self.lookahead_steps = [1, 2, 3, 4, 5]  # Different distances to predict
        
        # Results directory
        self.results_dir = "INSERT YOURS"

def pad_or_truncate_sequence(tokens, target_length):
    """Pad or truncate a sequence to the target length"""
    if tokens.shape[1] > target_length:
        return tokens[:, :target_length]
    elif tokens.shape[1] < target_length:
        pad_length = target_length - tokens.shape[1]
        return F.pad(tokens, (0, pad_length), value=0)  # 0 is usually the padding token
    return tokens

def prepare_dataset(model, config):
    """Load and prepare the Pile dataset"""
    print("Loading Pile-10k dataset...")
    dataset = load_dataset(config.dataset_name, split="train")
    
    # Take only max_samples
    dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    
    all_tokens = []
    all_activations = []
    hook_point = f"blocks.{config.layer}.hook_resid_pre"
    
    print("Processing sequences...")
    for item in tqdm(dataset):
        # Tokenize text and ensure fixed length
        tokens = model.to_tokens(item['text'])
        tokens = pad_or_truncate_sequence(tokens, config.sequence_length)
        
        # Get activations for the sequence
        activations = []
        def store_hook(act, hook):
            activations.append(act.detach().clone())
            return act
        
        with torch.no_grad():
            model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_point, store_hook)]
            )
        
        # Store results
        all_tokens.append(tokens)
        all_activations.append(activations[0])  # Only one activation per sequence
        
        # Print shapes for debugging
        if len(all_tokens) == 1:
            print(f"First sequence shapes - Tokens: {tokens.shape}, Activations: {activations[0].shape}")
    
    # Stack all sequences and activations
    tokens = torch.cat(all_tokens, dim=0)
    activations = torch.cat(all_activations, dim=0)
    
    print(f"Final dataset size - Tokens: {tokens.shape}, Activations: {activations.shape}")
    return tokens, activations

class ActToTokenDataset(Dataset):
    def __init__(self, acts, tokens, lookahead_len=1):
        self.acts = acts
        self.tokens = tokens
        self.lookahead_len = lookahead_len
        
        # Create valid indices (excluding padded positions)
        self.valid_indices = []
        for seq_idx in range(acts.shape[0]):
            for pos_idx in range(acts.shape[1] - lookahead_len):
                if tokens[seq_idx, pos_idx + lookahead_len] != 0:  # Skip if target is padding
                    self.valid_indices.append((seq_idx, pos_idx))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        seq_idx, pos_idx = self.valid_indices[idx]
        act = self.acts[seq_idx, pos_idx]
        target_token = self.tokens[seq_idx, pos_idx + self.lookahead_len]
        return act, target_token

class LookaheadMapper(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.projection = nn.Linear(d_in, d_in, bias=True)
        
        # Initialize with identity matrix
        self.projection.weight = nn.Parameter(torch.eye(d_in, dtype=torch.float32))
    
    def forward(self, x, unembed):
        return unembed(self.projection(x))

def train_lookahead_mapper(model, train_dataset, test_dataset, config):
    """Train a lookahead mapper model"""
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False
    )
    
    # Initialize model
    d_in = train_dataset[0][0].shape[0]
    lookahead_mapper = LookaheadMapper(d_in).to(config.device)
    global_unembed = model.unembed
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lookahead_mapper.parameters(), lr=config.learning_rate)
    
    # Training loop
    train_losses = []
    test_losses = []
    
    for epoch in range(config.num_epochs):
        # Training phase
        lookahead_mapper.train()
        total_train_loss = 0.0
        
        for batch_acts, batch_target_tokens in train_loader:
            batch_acts = batch_acts.to(config.device)
            batch_target_tokens = batch_target_tokens.to(config.device)
            
            optimizer.zero_grad()
            outputs = lookahead_mapper(batch_acts, global_unembed)
            loss = criterion(outputs, batch_target_tokens)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        lookahead_mapper.eval()
        total_test_loss = 0.0
        
        with torch.no_grad():
            for batch_acts, batch_target_tokens in test_loader:
                batch_acts = batch_acts.to(config.device)
                batch_target_tokens = batch_target_tokens.to(config.device)
                
                outputs = lookahead_mapper(batch_acts, global_unembed)
                loss = criterion(outputs, batch_target_tokens)
                
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        print(f"Epoch [{epoch+1}/{config.num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
    
    return lookahead_mapper, {"train_loss": train_losses, "test_loss": test_losses}

def evaluate_mapper(model, lookahead_mapper, test_dataset, config):
    """Evaluate the lookahead mapper"""
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    global_unembed = model.unembed
    
    lookahead_mapper.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_acts, batch_target_tokens in test_loader:
            batch_acts = batch_acts.to(config.device)
            batch_target_tokens = batch_target_tokens.to(config.device)
            
            outputs = lookahead_mapper(batch_acts, global_unembed)
            _, predicted = torch.max(outputs, 1)
            
            total += batch_target_tokens.size(0)
            correct += (predicted == batch_target_tokens).sum().item()
    
    accuracy = correct / total
    return accuracy

def main():
    config = Config()
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Initialize model
    print("Loading model...")
    model = HookedTransformer.from_pretrained(
        config.model_name,
        device=config.device
    )
    
    # Load and prepare dataset
    print("Preparing dataset...")
    tokens, activations = prepare_dataset(model, config)
    
    # Train and evaluate mappers for different lookahead distances
    results = {}
    
    for lookahead in config.lookahead_steps:
        print(f"\nTraining mapper for {lookahead}-token lookahead")
        
        # Create dataset for this lookahead distance
        dataset = ActToTokenDataset(activations, tokens, lookahead_len=lookahead)
        
        # Split into train and test
        train_size = int(config.train_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        # Train mapper
        mapper, training_metrics = train_lookahead_mapper(
            model, train_dataset, test_dataset, config
        )
        
        # Evaluate mapper
        accuracy = evaluate_mapper(model, mapper, test_dataset, config)
        
        # Save results
        results[lookahead] = {
            "accuracy": accuracy,
            "train_loss": training_metrics["train_loss"],
            "test_loss": training_metrics["test_loss"]
        }
        
        # Save mapper
        torch.save(mapper.state_dict(), 
                  f"{config.results_dir}/mapper_lookahead_{lookahead}.pt")
        
        print(f"Lookahead {lookahead} Accuracy: {accuracy:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    lookaheads = list(results.keys())
    accuracies = [results[k]["accuracy"] for k in lookaheads]
    
    plt.plot(lookaheads, accuracies, 'o-')
    plt.xlabel("Lookahead Distance")
    plt.ylabel("Accuracy")
    plt.title("Lookahead Prediction Accuracy")
    plt.grid(True)
    
    plt.savefig(f"{config.results_dir}/lookahead_accuracies.png")
    plt.close()

if __name__ == "__main__":
    main()