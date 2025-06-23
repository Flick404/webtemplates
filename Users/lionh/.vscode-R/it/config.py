"""
FSVM Configuration Settings

This file contains all configurable parameters for the Fracture-Sourced Virtual Machine.
"""

# FSVM - Fracture-Sourced Virtual Machine
# ------------------------------------------
# Phase 2 Optimized Configuration
# Rationale: Tuned for emergence, reactivity, and stability during recursive evolution.
# ------------------------------------------

### 🔧 CORE_SETTINGS
CORE_SETTINGS = {
    'memory_size': 1500,                 # Increased STM capacity for richer patterns
    'max_pattern_length': 12,            # Allows deeper pattern detection
    'min_pattern_repetitions': 3,        # More repetitions = more stable symbols
    'tension_threshold': 0.15,           # Lowered = more sensitive to resonance
    'high_tension_threshold': 0.65,      # Lowered = earlier chaos detection
    'recursive_iterations': 7,           # More cycles = deeper hallucination
    'cycle_delay': 0.001,                # Dramatically faster cognitive cycle
    'tension_multiplier': 1.0,           # UI-controllable tension sensitivity
}

### 🌀 ENTROPY_SETTINGS
ENTROPY_SETTINGS = {
    'max_files_to_sample': 7,            # Slightly more data exposure
    'web_request_timeout': 4,
    'max_file_read_size': 1500,          # Deeper text injection = more complexity
    'entropy_batch_size': 48,            # Larger batches = stronger entropy shock
}

### 🧠 MEMORY_SETTINGS
MEMORY_SETTINGS = {
    'memory_dir': 'fsvm_memory',
    'max_interactions': 2500,            # Longer history = conceptual linkage
    'save_interval': 5,                  # Faster saving = real-time adaptation
    'symbol_cleanup_threshold': 300,     # Symbols persist longer = stable ontology
}

### 📊 UI_SETTINGS
UI_SETTINGS = {
    'update_interval': 0.05,             # Faster refresh = real-time feel
    'max_tension_history': 1500,         # Smoother tension curve
    'chart_height': 350,
    'symbol_table_max_rows': 100,        # View deeper symbolic layers
}

### 🗣 COMMUNICATION_SETTINGS
COMMUNICATION_SETTINGS = {
    'response_timeout': 15,              # Longer time to "think"
    'max_response_length': 300,          # Room for abstract responses
    'silent_threshold': 0.6,             # More cautious responder
}

### ⚠️ SAFETY_SETTINGS
SAFETY_SETTINGS = {
    'max_cpu_usage': 90,                 # Push limits but respect thermal bounds
    'max_memory_usage': 3072,            # Allocate more RAM for long sequences
    'kill_switch_enabled': True,
    'error_recovery_enabled': True,
}

### 🧪 DEV_SETTINGS
DEV_SETTINGS = {
    'debug_mode': True,                  # Enable to trace Phase 2 logic
    'log_level': 'DEBUG',
    'test_mode': False,
    'performance_monitoring': True,
}

### 🌐 WEB_API_SETTINGS
WEB_API_SETTINGS = {
    'user_agent': 'FSVM-Entropy-Collector/2.0',
    'allowed_domains': [
        'api.quotable.io',
        'api.publicapis.org',
        'httpbin.org',
        'jsonplaceholder.typicode.com',
        'newsapi.org',
        'reddit.com',
        'wikipedia.org'
    ],
    'max_response_size': 20000,          # Richer entropy feed from larger payloads
}

### 🧬 NEURAL_SETTINGS
NEURAL_SETTINGS = {
    'use_neural_network': True,
    'hidden_size': 128,                  # More latent capacity
    'learning_rate': 0.0005,             # Reduced for smoother learning
    'batch_size': 4,                     # Micro-batch = fast, reactive updates
    'sequence_length': 16,               # More temporal context
}

### 🩸 DRIVE_SETTINGS
DRIVE_SETTINGS = {
    "curiosity": 0.55,
    "stability": 0.4,
    "boredom": 0.1,
    "novelty_hunger": 0.75,
    "boredom_threshold": 15,             # Faster boredom onset = drive to hallucinate
    "boredom_increment": 0.07,
    "curiosity_increment": 0.015,
    "stability_increment": 0.01,
    "drive_decay": 0.993                 # Slower decay = drives linger longer
} 