import torch
import torch.nn as nn
import math
import warnings
import os

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Generate City Coordinates for an 8x8 Grid
def generate_city_coordinates(grid_size=8):
    cities = []
    for x in range(grid_size):
        for y in range(grid_size):
            cities.append((x, y))
    return torch.tensor(cities, dtype=torch.float32)  # Shape: (64, 2)

# 2. Define Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# 3. Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim=2, d_model=128, n_heads=8, num_layers=4):
        super(Encoder, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src):
        src = self.input_linear(src)  # Shape: (batch_size, 64, d_model)
        src = self.pos_encoder(src)    # Shape: (batch_size, 64, d_model)
        memory = self.transformer_encoder(src)  # Shape: (batch_size, 64, d_model)
        return memory

# 4. Define the Decoder
class Decoder(nn.Module):
    def __init__(self, d_model=128, n_heads=8, num_layers=4, k_salesmen=1):
        super(Decoder, self).__init__()
        self.k = k_salesmen
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        # For this deterministic approach, we don't use actual Transformer Decoders
        # Instead, we'll assign cities based on sorted distances

    def forward(self, memory, city_coords, depot=(0, 0)):
        batch_size = memory.size(0)
        num_cities = memory.size(1)
        routes = []

        for b in range(batch_size):
            # Compute distances from depot for all cities
            distances = torch.norm(city_coords[b] - torch.tensor(depot, dtype=torch.float32, device=device), dim=1)  # Shape: (64,)
            sorted_distances, sorted_indices = torch.sort(distances)  # Ascending order

            # Determine number of cities per salesman
            cities_per_salesman = num_cities // self.k
            remainder = num_cities % self.k

            start = 0
            for i in range(self.k):
                # Distribute the remainder among the first 'remainder' salesmen
                end = start + cities_per_salesman + (1 if i < remainder else 0)
                assigned_indices = sorted_indices[start:end]
                start = end

                # Convert indices to city coordinates
                assigned_cities = city_coords[b][assigned_indices].tolist()
                # Create route: depot -> assigned cities -> depot
                route = [depot] + assigned_cities + [depot]
                routes.append(route)

        return routes

# 5. Define the MTSP Model
class MTSPModel(nn.Module):
    def __init__(self, grid_size=8, d_model=128, n_heads=8, num_layers=4, k_salesmen=1):
        super(MTSPModel, self).__init__()
        self.encoder = Encoder(input_dim=2, d_model=d_model, n_heads=n_heads, num_layers=num_layers)
        self.decoder = Decoder(d_model=d_model, n_heads=n_heads, num_layers=num_layers, k_salesmen=k_salesmen)

    def forward(self, k, city_coords, depot=(0,0)):
        memory = self.encoder(city_coords)  # Shape: (batch_size, 64, d_model)
        routes = self.decoder(memory, city_coords, depot)
        return routes

# 6. Function to Solve MTSP
def solve_mtsp(k, grid_size=8):
    # Ensure k is less than the number of cities
    num_cities = grid_size * grid_size
    if k > num_cities:
        raise ValueError("Number of salesmen cannot exceed number of cities.")

    # Initialize Model
    model = MTSPModel(grid_size=grid_size, k_salesmen=k).to(device)
    model.eval()  # Set model to evaluation mode

    # Generate city coordinates and add batch dimension
    city_coords = generate_city_coordinates(grid_size).unsqueeze(0).to(device)  # Shape: (1, 64, 2)

    with torch.no_grad():
        routes = model(k, city_coords, depot=(0,0))

    # Convert routes to a readable format
    salesman_routes = []
    for i in range(k):
        route = routes[i]
        # Format cities as tuples
        formatted_route = [tuple(map(float, city)) for city in route]
        salesman_routes.append(formatted_route)

    return salesman_routes

# Calculate path distance
def calculate_path_distance(route):
    """Calculate the total distance of a path"""
    total_distance = 0.0
    for i in range(len(route) - 1):
        # Ensure points are properly handled as tensors with gradients
        point1 = route[i].float()
        point2 = route[i + 1].float()
        # Calculate Euclidean distance while maintaining gradients
        distance = torch.norm(point2 - point1)
        total_distance = total_distance + distance
    return total_distance

# Enhanced MTSP Model with learnable parameters
class EnhancedMTSPModel(nn.Module):
    def __init__(self, grid_size=8, d_model=128, n_heads=8, num_layers=4, k_salesmen=1):
        super(EnhancedMTSPModel, self).__init__()
        self.encoder = Encoder(input_dim=2, d_model=d_model, n_heads=n_heads, num_layers=num_layers)
        # Improved route scorer with deeper network
        self.route_scorer = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        self.k_salesmen = k_salesmen
        self.grid_size = grid_size
        # Initialize depot near center of grid for better starting point
        self.depot = nn.Parameter(torch.tensor([grid_size/2.0, grid_size/2.0], device=device))

    def forward(self, city_coords):
        batch_size = city_coords.size(0)
        memory = self.encoder(city_coords)
        
        # Score each city based on encoded features
        scores = self.route_scorer(memory).squeeze(-1)
        
        routes = []
        for b in range(batch_size):
            # Use softmax for smoother score distribution
            scores_prob = torch.softmax(scores[b], dim=0)
            sorted_scores, sorted_indices = torch.sort(scores_prob, descending=True)
            
            num_cities = len(sorted_indices)
            cities_per_salesman = num_cities // self.k_salesmen
            remainder = num_cities % self.k_salesmen
            
            start = 0
            batch_routes = []
            for i in range(self.k_salesmen):
                end = start + cities_per_salesman + (1 if i < remainder else 0)
                assigned_indices = sorted_indices[start:end]
                assigned_cities = city_coords[b][assigned_indices]
                
                route = torch.cat([
                    self.depot.unsqueeze(0),
                    assigned_cities,
                    self.depot.unsqueeze(0)
                ])
                batch_routes.append(route)
                start = end
            
            routes.append(batch_routes)
        
        return routes

# Training utilities
class MTSPTrainer:
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        # Initialize optimizer with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        # Add learning rate scheduler with smaller factor and patience
        self.scheduler = torch.optim.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.2,  # Reduced from 0.5 for finer control
            patience=3,   # Reduced from 5 to be more responsive
            verbose=True
        )
        self.best_loss = float('inf')
        self.best_max_distance = float('inf')
        self.metrics_history = {
            'loss': [], 'max_distance': [], 'avg_distance': [],
            'variance': [], 'learning_rate': []
        }
        # Adjust weights to focus heavily on max distance
        self.max_dist_weight = 5.0     # Increased from 1.0
        self.var_weight = 2.0          # Increased from 0.5
        self.smoothness_weight = 0.1
        self.balance_weight = 1.0      # New weight for path balance
    
    def calculate_loss(self, routes):
        """Enhanced loss calculation focusing on minimizing maximum path distance"""
        batch_size = len(routes)
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        max_distances = []
        avg_distances = []
        variances = []
        
        for batch_routes in routes:
            # Calculate distances for each route
            distances = torch.stack([calculate_path_distance(route) for route in batch_routes])
            
            # Calculate metrics
            max_distance = torch.max(distances)
            min_distance = torch.min(distances)
            avg_distance = torch.mean(distances)
            distance_variance = torch.var(distances)
            
            # New: Calculate path balance penalty
            balance_penalty = max_distance - min_distance
            
            # Calculate route smoothness (penalize sharp turns)
            smoothness_loss = torch.tensor(0.0, device=device)
            for route in batch_routes:
                for i in range(len(route) - 2):
                    vec1 = route[i + 1] - route[i]
                    vec2 = route[i + 2] - route[i + 1]
                    vec1 = vec1 / (torch.norm(vec1) + 1e-6)
                    vec2 = vec2 / (torch.norm(vec2) + 1e-6)
                    smoothness_loss = smoothness_loss - torch.dot(vec1, vec2)
            
            # Enhanced loss function focusing on max distance
            route_loss = (
                self.max_dist_weight * torch.pow(max_distance, 2) +  # Square max distance for stronger penalty
                self.var_weight * distance_variance +
                self.balance_weight * balance_penalty +
                self.smoothness_weight * (smoothness_loss / len(batch_routes))
            )
            
            # Add exponential penalty for very high max distances
            if max_distance > avg_distance * 1.2:  # If max is 20% above average
                route_loss = route_loss + self.max_dist_weight * torch.exp(max_distance - avg_distance)
            
            total_loss = total_loss + route_loss
            
            max_distances.append(max_distance.item())
            avg_distances.append(avg_distance.item())
            variances.append(distance_variance.item())
        
        avg_loss = total_loss / batch_size
        return (
            avg_loss,
            sum(max_distances) / len(max_distances),
            sum(avg_distances) / len(avg_distances),
            sum(variances) / len(variances)
        )
    
    def train(self, num_epochs, batch_size=32, grid_size=8):
        """Enhanced training loop with better progress tracking"""
        print("Starting training...")
        patience = 15  # Increased patience for max distance improvement
        patience_counter = 0
        min_max_distance = float('inf')
        
        for epoch in range(num_epochs):
            # Generate random batch of city coordinates
            batch_coords = []
            for _ in range(batch_size):
                coords = generate_city_coordinates(grid_size).unsqueeze(0)
                batch_coords.append(coords)
            batch_coords = torch.cat(batch_coords, dim=0).to(device)
            
            # Training step
            loss, max_dist, avg_dist, variance = self.train_step(batch_coords)
            
            # Update learning rate scheduler based on max_distance instead of loss
            self.scheduler.step(max_dist)
            
            # Early stopping check based on max distance
            if max_dist < min_max_distance:
                min_max_distance = max_dist
                patience_counter = 0
                # Save model if it achieves better max distance
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': loss,
                    'max_distance': max_dist,
                    'metrics': self.metrics_history
                }, 'best_mtsp_model.pt')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"Loss: {loss:.4f}, Max Distance: {max_dist:.2f}, Avg Distance: {avg_dist:.2f}")
                print(f"Variance: {variance:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                print(f"Best Max Distance: {min_max_distance:.2f}")
            
            # Early stopping if max distance hasn't improved
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best achieved max distance: {min_max_distance:.2f}")
                break
        
        print("Training completed!")
        return self.metrics_history

# Enhanced solve function with trained model
def solve_mtsp_enhanced(k, grid_size=8, model_path=None):
    """Solve MTSP using trained model if available"""
    model = EnhancedMTSPModel(grid_size=grid_size, k_salesmen=k).to(device)
    
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    
    model.eval()
    city_coords = generate_city_coordinates(grid_size).unsqueeze(0).to(device)
    
    with torch.no_grad():
        routes = model(city_coords)
        
        # Calculate and print route distances
        distances = []
        for route in routes[0]:  # First batch
            distance = calculate_path_distance(route)
            distances.append(distance.item())
        
        print("\nRoute Distances:")
        for i, dist in enumerate(distances):
            print(f"Salesman {i+1}: {dist:.2f}")
        print(f"Max Distance: {max(distances):.2f}")
        print(f"Average Distance: {sum(distances)/len(distances):.2f}")
    
    return routes[0]  # Return first batch's routes

# 7. Main Execution
if __name__ == "__main__":
    # Training example with optimized parameters
    k_salesmen = 4
    grid_size = 8
    model = EnhancedMTSPModel(grid_size=grid_size, k_salesmen=k_salesmen).to(device)
    trainer = MTSPTrainer(model, learning_rate=1e-3)  # Higher initial learning rate
    
    # Train model with more epochs and smaller batch size
    metrics = trainer.train(num_epochs=300, batch_size=8, grid_size=grid_size)
    
    # Solve MTSP with trained model
    routes = solve_mtsp_enhanced(k_salesmen, grid_size=grid_size, model_path='best_mtsp_model.pt')
    
    # Print routes with distances
    total_distance = 0
    for i, route in enumerate(routes):
        distance = calculate_path_distance(route).item()
        total_distance += distance
        print(f"\nSalesman {i+1} route (distance: {distance:.2f}):")
        print([tuple(map(float, city)) for city in route])
    
    print(f"\nTotal Distance: {total_distance:.2f}")
    print(f"Average Distance: {total_distance/k_salesmen:.2f}")
