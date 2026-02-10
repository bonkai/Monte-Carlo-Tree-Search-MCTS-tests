"""
main.py - Main execution script
"""
import logging
import os
from datetime import datetime
from config import Config
from memory_manager import ContextManager, VectorStore
from mcts import MCTS
import requests
from typing import Optional, List, Dict, Any
import json

class LLMClient:
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=3,
            pool_maxsize=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def ask_question(
        self, 
        question: str, 
        temperature: float = 0.7,
        system_prompt: str = ""
    ) -> Optional[str]:
        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False
        }
        
        try:
            response = self.session.post(
                self.config.api_url,
                json=payload,
                timeout=(5, 300)
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'message' in result and 'content' in result['message']:
                    return result['message']['content'].strip()
        except Exception as e:
            logging.error(f"Error in LLM request: {str(e)}")
        
        return None

def setup_logging(config: Config) -> None:
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/{config.log_file}"),
            logging.StreamHandler()
        ]
    )

def main():
    config = Config()
    setup_logging(config)
    context_manager = ContextManager(config)
    llm_client = LLMClient(config)
    mcts = MCTS(llm_client, config)

    try:
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break

            # Build context-aware prompt
            system_prompt = context_manager.build_prompt(question)
            
            print("\nSearching for optimal response...")
            best_response = mcts.search(question, system_prompt)

            if best_response:
                print("\nBest Response:")
                print("=" * 80)
                print(best_response)
                print("=" * 80)

                # Update conversation history and memory
                context_manager.add_exchange(question, best_response)
                
            else:
                print("\nError: Could not generate a response.")

    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()