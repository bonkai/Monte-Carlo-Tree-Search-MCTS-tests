"""
config.py - Configuration and constants
"""
from dataclasses import dataclass

@dataclass
class Config:
    # LLM Settings
    api_url: str = "http://localhost:11434/api/chat"
    # model_name: str = "qwen2.5:14b"
    model_name: str = "llama3.2:3b"
    max_tokens: int = 50000
    
    # Database Settings
    db_host: str = "localhost"
    db_name: str = "llm_memory"
    table_name: str = "testing"
    
    # Memory Management
    current_context_tokens: int = 20000  # For 32k context
    memory_tokens: int = 12000
    similarity_threshold: float = 0.6
    
    # MCTS Settings
    mcts_iterations: int = 10
    simulation_depth: int = 3
    beam_width: int = 3
    exploration_constant: float = 1.414
    min_temperature: float = 0.1
    max_temperature: float = 1.0
    
    # Files and Logging
    log_file: str = "llm_queries.log"
    response_dir: str = "responses"