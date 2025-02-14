"""
Install an additional SDK for JSON schema support Google AI Python SDK

$ pip install google.ai.generativelanguage
"""

import os
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

def get_path_model_chat_session():

    # Create the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_schema": content.Schema(
        type = content.Type.OBJECT,
        description = "Configuration schema for robust multi-UAV target load balancing environment",
        enum = [],
        required = ["n_agents", "grid_size", "max_steps", "n_targets"],
        properties = {
        "n_agents": content.Schema(
            type = content.Type.INTEGER,
        ),
        "grid_size": content.Schema(
            type = content.Type.OBJECT,
            description = "Size of the environment grid where each cell is 50 meters",
            enum = [],
            required = ["x", "y"],
            properties = {
            "x": content.Schema(
                type = content.Type.INTEGER,
            ),
            "y": content.Schema(
                type = content.Type.INTEGER,
            ),
            },
        ),
        "max_steps": content.Schema(
            type = content.Type.INTEGER,
        ),
        "comm_dist": content.Schema(
            type = content.Type.NUMBER,
        ),
        "gcs_pos": content.Schema(
            type = content.Type.OBJECT,
            description = "Position of ground control station",
            enum = [],
            required = ["x", "y"],
            properties = {
            "x": content.Schema(
                type = content.Type.INTEGER,
            ),
            "y": content.Schema(
                type = content.Type.INTEGER,
            ),
            },
        ),
        "enable_connectivity": content.Schema(
            type = content.Type.BOOLEAN,
        ),
        "n_targets": content.Schema(
            type = content.Type.INTEGER,
        ),
        "target_priorities": content.Schema(
            type = content.Type.OBJECT,
            properties = {
            "priorities": content.Schema(
                type = content.Type.OBJECT,
                properties = {
                "targets": content.Schema(
                    type = content.Type.ARRAY,
                    items = content.Schema(
                    type = content.Type.OBJECT,
                    enum = [],
                    required = ["id", "priority"],
                    properties = {
                        "id": content.Schema(
                        type = content.Type.INTEGER,
                        ),
                        "priority": content.Schema(
                        type = content.Type.NUMBER,
                        ),
                    },
                    ),
                ),
                },
            ),
            },
        ),
        "inform_times": content.Schema(
            type = content.Type.INTEGER,
        ),
        "sensor_config": content.Schema(
            type = content.Type.OBJECT,
            description = "Sensor configuration for target detection",
            properties = {
            "detection_range": content.Schema(
                type = content.Type.NUMBER,
            ),
            "base_reliability": content.Schema(
                type = content.Type.NUMBER,
            ),
            "noise_std": content.Schema(
                type = content.Type.NUMBER,
            ),
            "sensor_type": content.Schema(
                type = content.Type.STRING,
            ),
            },
        ),
        "drone_configs": content.Schema(
            type = content.Type.OBJECT,
            description = "Configuration for each drone",
            properties = {
            "drones": content.Schema(
                type = content.Type.ARRAY,
                items = content.Schema(
                type = content.Type.OBJECT,
                enum = [],
                required = ["drone_type"],
                properties = {
                    "drone_type": content.Schema(
                    type = content.Type.STRING,
                    ),
                    "initial_position": content.Schema(
                    type = content.Type.OBJECT,
                    enum = [],
                    required = ["x", "y"],
                    properties = {
                        "x": content.Schema(
                        type = content.Type.INTEGER,
                        ),
                        "y": content.Schema(
                        type = content.Type.INTEGER,
                        ),
                    },
                    ),
                },
                ),
            ),
            },
        ),
        "target_configs": content.Schema(
            type = content.Type.OBJECT,
            description = "Configuration for each target",
            properties = {
            "targets": content.Schema(
                type = content.Type.ARRAY,
                items = content.Schema(
                type = content.Type.OBJECT,
                enum = [],
                required = ["target_type", "priority"],
                properties = {
                    "target_type": content.Schema(
                    type = content.Type.STRING,
                    ),
                    "priority": content.Schema(
                    type = content.Type.NUMBER,
                    ),
                    "speed": content.Schema(
                    type = content.Type.NUMBER,
                    ),
                    "movement_pattern": content.Schema(
                    type = content.Type.OBJECT,
                    description = "List of waypoints for patrol patterns",
                    properties = {
                        "waypoints": content.Schema(
                        type = content.Type.OBJECT,
                        enum = [],
                        required = ["x", "y"],
                        properties = {
                            "x": content.Schema(
                            type = content.Type.INTEGER,
                            ),
                            "y": content.Schema(
                            type = content.Type.INTEGER,
                            ),
                        },
                        ),
                    },
                    ),
                },
                ),
            ),
            },
        ),
        "terrain_config": content.Schema(
            type = content.Type.OBJECT,
            properties = {
            "obstacles": content.Schema(
                type = content.Type.OBJECT,
                description = "Static obstacle positions",
                properties = {
                "positions": content.Schema(
                    type = content.Type.OBJECT,
                    enum = [],
                    required = ["x", "y"],
                    properties = {
                    "x": content.Schema(
                        type = content.Type.INTEGER,
                    ),
                    "y": content.Schema(
                        type = content.Type.INTEGER,
                    ),
                    },
                ),
                },
            ),
            "charging_stations": content.Schema(
                type = content.Type.OBJECT,
                description = "Charging station positions",
                properties = {
                "positions": content.Schema(
                    type = content.Type.OBJECT,
                    enum = [],
                    required = ["x", "y"],
                    properties = {
                    "x": content.Schema(
                        type = content.Type.INTEGER,
                    ),
                    "y": content.Schema(
                        type = content.Type.INTEGER,
                    ),
                    },
                ),
                },
            ),
            "comm_relays": content.Schema(
                type = content.Type.OBJECT,
                description = "Communication relay positions",
                properties = {
                "positions": content.Schema(
                    type = content.Type.OBJECT,
                    enum = [],
                    required = ["x", "y"],
                    properties = {
                    "x": content.Schema(
                        type = content.Type.INTEGER,
                    ),
                    "y": content.Schema(
                        type = content.Type.INTEGER,
                    ),
                    },
                ),
                },
            ),
            "interference": content.Schema(
                type = content.Type.OBJECT,
                properties = {
                "base_level": content.Schema(
                    type = content.Type.NUMBER,
                ),
                "high_interference_zones": content.Schema(
                    type = content.Type.OBJECT,
                    properties = {
                    "zones": content.Schema(
                        type = content.Type.OBJECT,
                        enum = [],
                        required = ["center", "radius", "strength"],
                        properties = {
                        "center": content.Schema(
                            type = content.Type.OBJECT,
                            enum = [],
                            required = ["x", "y"],
                            properties = {
                            "x": content.Schema(
                                type = content.Type.INTEGER,
                            ),
                            "y": content.Schema(
                                type = content.Type.INTEGER,
                            ),
                            },
                        ),
                        "radius": content.Schema(
                            type = content.Type.NUMBER,
                        ),
                        "strength": content.Schema(
                            type = content.Type.NUMBER,
                        ),
                        },
                    ),
                    },
                ),
                "low_interference_corridors": content.Schema(
                    type = content.Type.OBJECT,
                    properties = {
                    "corridors": content.Schema(
                        type = content.Type.OBJECT,
                        enum = [],
                        required = ["start", "end", "width"],
                        properties = {
                        "start": content.Schema(
                            type = content.Type.OBJECT,
                            enum = [],
                            required = ["x", "y"],
                            properties = {
                            "x": content.Schema(
                                type = content.Type.INTEGER,
                            ),
                            "y": content.Schema(
                                type = content.Type.INTEGER,
                            ),
                            },
                        ),
                        "end": content.Schema(
                            type = content.Type.OBJECT,
                            enum = [],
                            required = ["x", "y"],
                            properties = {
                            "x": content.Schema(
                                type = content.Type.INTEGER,
                            ),
                            "y": content.Schema(
                                type = content.Type.INTEGER,
                            ),
                            },
                        ),
                        "width": content.Schema(
                            type = content.Type.NUMBER,
                        ),
                        },
                    ),
                    },
                ),
                },
            ),
            "dynamic_obstacles": content.Schema(
                type = content.Type.OBJECT,
                properties = {
                "obstacles": content.Schema(
                    type = content.Type.OBJECT,
                    enum = [],
                    required = ["initial_position", "pattern", "speed"],
                    properties = {
                    "initial_position": content.Schema(
                        type = content.Type.OBJECT,
                        enum = [],
                        required = ["x", "y"],
                        properties = {
                        "x": content.Schema(
                            type = content.Type.INTEGER,
                        ),
                        "y": content.Schema(
                            type = content.Type.INTEGER,
                        ),
                        },
                    ),
                    "pattern": content.Schema(
                        type = content.Type.STRING,
                    ),
                    "waypoints": content.Schema(
                        type = content.Type.OBJECT,
                        enum = [],
                        required = ["x", "y"],
                        properties = {
                        "x": content.Schema(
                            type = content.Type.INTEGER,
                        ),
                        "y": content.Schema(
                            type = content.Type.INTEGER,
                        ),
                        },
                    ),
                    "speed": content.Schema(
                        type = content.Type.NUMBER,
                    ),
                    },
                ),
                },
            ),
            },
        ),
        "obs_config": content.Schema(
            type = content.Type.OBJECT,
            properties = {
            "known_target_positions": content.Schema(
                type = content.Type.BOOLEAN,
            ),
            "known_target_priorities": content.Schema(
                type = content.Type.BOOLEAN,
            ),
            "known_time_slots": content.Schema(
                type = content.Type.BOOLEAN,
            ),
            },
        ),
        "reward_weights": content.Schema(
            type = content.Type.OBJECT,
            properties = {
            "coverage": content.Schema(
                type = content.Type.NUMBER,
            ),
            "revisit": content.Schema(
                type = content.Type.NUMBER,
            ),
            "connectivity": content.Schema(
                type = content.Type.NUMBER,
            ),
            "target_detection": content.Schema(
                type = content.Type.NUMBER,
            ),
            "target_relay": content.Schema(
                type = content.Type.NUMBER,
            ),
            "target_completion": content.Schema(
                type = content.Type.NUMBER,
            ),
            "early_completion": content.Schema(
                type = content.Type.NUMBER,
            ),
            "load_balance": content.Schema(
                type = content.Type.NUMBER,
            ),
            "energy_efficiency": content.Schema(
                type = content.Type.NUMBER,
            ),
            "collision_penalty": content.Schema(
                type = content.Type.NUMBER,
            ),
            },
        ),
        },
    ),
    "response_mime_type": "application/json",
    }

    print("initialised model")

    model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    )

    print("started chat")

    chat_session = model.start_chat(
    history=[]
    )

    return chat_session



def convert_to_array_format(text_config):
    """Convert schema config format to array-based format used in environment."""
    import json
    import re
    
    # Convert text to dictionary
    config = {}
    try:
        # Handle both string and dict inputs
        if isinstance(text_config, str):
            config = json.loads(text_config)
        else:
            config = text_config.copy()
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format")
        
    # Convert coordinate objects to lists
    def convert_pos_to_list(pos):
        if isinstance(pos, dict) and 'x' in pos and 'y' in pos:
            return [pos['x'], pos['y']]
        return pos
        
    # Convert grid size
    if 'grid_size' in config:
        config['grid_size'] = [config['grid_size']['x'], config['grid_size']['y']]
        
    # Convert GCS position
    if 'gcs_pos' in config:
        config['gcs_pos'] = convert_pos_to_list(config['gcs_pos'])
        
    # Convert drone configs
    if 'drone_configs' in config:
        drone_list = []
        drones = config['drone_configs'].get('drones', {})
        if isinstance(drones, dict):  # Single drone
            drone_dict = {'drone_type': drones['drone_type']}
            if 'initial_position' in drones:
                drone_dict['initial_position'] = convert_pos_to_list(drones['initial_position'])
            drone_list.append(drone_dict)
        elif isinstance(drones, list):  # Multiple drones
            for drone in drones:
                drone_dict = {'drone_type': drone['drone_type']}
                if 'initial_position' in drone:
                    drone_dict['initial_position'] = convert_pos_to_list(drone['initial_position'])
                drone_list.append(drone_dict)
        config['drone_configs'] = drone_list
        
    # Convert target configs
    if 'target_configs' in config:
        target_list = []
        targets = config['target_configs'].get('targets', {})
        if isinstance(targets, dict):  # Single target
            target_dict = {
                'target_type': targets['target_type'],
                'priority': targets['priority']
            }
            if 'speed' in targets:
                target_dict['speed'] = targets['speed']
            if 'movement_pattern' in targets and 'waypoints' in targets['movement_pattern']:
                target_dict['movement_pattern'] = [
                    convert_pos_to_list(targets['movement_pattern']['waypoints'])
                ]
            target_list.append(target_dict)
        elif isinstance(targets, list):  # Multiple targets
            for target in targets:
                target_dict = {
                    'target_type': target['target_type'],
                    'priority': target['priority']
                }
                if 'speed' in target:
                    target_dict['speed'] = target['speed']
                if 'movement_pattern' in target and 'waypoints' in target['movement_pattern']:
                    target_dict['movement_pattern'] = [
                        convert_pos_to_list(target['movement_pattern']['waypoints'])
                    ]
                target_list.append(target_dict)
        config['target_configs'] = target_list
        
    # Convert terrain config positions
    if 'terrain_config' in config:
        terrain = config['terrain_config']
        
        # Convert obstacles
        if 'obstacles' in terrain:
            pos = terrain['obstacles'].get('positions', {})
            terrain['obstacles'] = [convert_pos_to_list(pos)]
            
        # Convert charging stations
        if 'charging_stations' in terrain:
            pos = terrain['charging_stations'].get('positions', {})
            terrain['charging_stations'] = [convert_pos_to_list(pos)]
            
        # Convert comm relays
        if 'comm_relays' in terrain:
            pos = terrain['comm_relays'].get('positions', {})
            terrain['comm_relays'] = [convert_pos_to_list(pos)]
            
        # Convert interference zones
        if 'interference' in terrain:
            interference = terrain['interference']
            
            if 'high_interference_zones' in interference:
                zones = interference['high_interference_zones'].get('zones', {})
                if isinstance(zones, dict):
                    zones = [zones]
                converted_zones = []
                for zone in zones:
                    if 'center' in zone:
                        zone['center'] = convert_pos_to_list(zone['center'])
                    converted_zones.append(zone)
                interference['high_interference_zones'] = converted_zones
                
            if 'low_interference_corridors' in interference:
                corridors = interference['low_interference_corridors'].get('corridors', {})
                if isinstance(corridors, dict):
                    corridors = [corridors]
                converted_corridors = []
                for corridor in corridors:
                    if 'start' in corridor:
                        corridor['start'] = convert_pos_to_list(corridor['start'])
                    if 'end' in corridor:
                        corridor['end'] = convert_pos_to_list(corridor['end'])
                    converted_corridors.append(corridor)
                interference['low_interference_corridors'] = converted_corridors
                    
    # Convert target priorities to array
    if 'target_priorities' in config:
        priorities = config['target_priorities'].get('priorities', {}).get('targets', {})
        if isinstance(priorities, dict):
            config['target_priorities'] = [priorities.get('priority', 1.0)]
            
    return config


if __name__ == "__main__":
    chat_session = get_path_model_chat_session()
    response = chat_session.send_message("on an 8x8 grid everything is normal 2 drones 1 target")
    print(convert_to_array_format(response.text))
