import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_input_dim(env):
    """Calculate the total state dimension"""
    # Grid cells
    dim = env.G1 * env.G2
    # Agent positions (x,y for each agent)
    dim += env.n_agents * 2
    # Target states (position, detection, completion)
    dim += env.n_targets * 4
    return dim

class Generator(nn.Module):
    def __init__(self, num_agents, action_dim, hidden_dim, state_dim):
        super().__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # State encoder with proper dimensions
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Per-agent policy networks
        self.agent_policies = nn.ModuleList([
            nn.GRUCell(hidden_dim + action_dim, hidden_dim)
            for _ in range(num_agents)
        ])
        
        # Action heads with proper dimensions
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            for _ in range(num_agents)
        ])
        
    def forward(self, state, temperature=1.0):
        batch_size = state.size(0)
        
        # Encode global state
        global_features = self.state_encoder(state)
        
        # Initialize hidden states
        hidden_states = [torch.zeros(batch_size, self.hidden_dim, 
                                   device=state.device) 
                        for _ in range(self.num_agents)]
        
        # Generate action probabilities for each agent
        action_probs = []
        for i in range(self.num_agents):
            # Previous action embedding (zeros for first step)
            prev_action = torch.zeros(batch_size, self.action_dim, 
                                    device=state.device)
            
            # Update hidden state with global features and prev action
            agent_input = torch.cat([global_features, prev_action], dim=-1)
            hidden_states[i] = self.agent_policies[i](agent_input, hidden_states[i])
            
            # Get action logits and apply temperature
            logits = self.action_heads[i](hidden_states[i])
            scaled_logits = logits / temperature
            
            # Convert to probabilities
            probs = F.softmax(scaled_logits, dim=-1)
            action_probs.append(probs)
            
        return torch.stack(action_probs, dim=1)

class Discriminator(nn.Module):
    def __init__(self, num_agents, action_dim, hidden_dim, state_dim, max_seq_len):
        super().__init__()
        self.num_agents = num_agents
        
        # Input size includes state and actions for all agents
        combined_dim = state_dim + (num_agents * action_dim)
        
        # Trajectory encoder (bi-directional for full sequence context)
        self.trajectory_encoder = nn.GRU(
            input_size=combined_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        
        # Value prediction head
        self.value_head = nn.Sequential(
            nn.Linear(3200, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, trajectories):
        # Encode full trajectory
        encoded, _ = self.trajectory_encoder(trajectories)
        
        # Use final hidden state to predict value
        final_hidden = encoded[:, -1]

        value = self.value_head(final_hidden)
        
        return value

class MultiAgentSeqGAN:
    def __init__(self, env, config):
        self.env = env
        self.num_agents = env.n_agents
        self.action_dim = 5  # From your environment
        self.state_dim = calculate_input_dim(env)
        
        # Initialize networks
        self.generator = Generator(
            num_agents=self.num_agents,
            action_dim=self.action_dim,
            hidden_dim=config['hidden_dim'],
            state_dim=self.state_dim
        )
        
        self.discriminator = Discriminator(
            num_agents=self.num_agents,
            action_dim=self.action_dim,
            hidden_dim=config['hidden_dim'],
            state_dim=self.state_dim,
            max_seq_len=config['max_seq_len']
        )