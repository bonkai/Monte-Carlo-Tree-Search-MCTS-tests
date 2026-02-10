import requests
import logging
import json
from typing import List, Dict, Optional, Tuple
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from datetime import datetime
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
import tiktoken
from database_manager import DatabaseManager
from collections import deque

# Initialize global models
logging.info("Initializing global models...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
current_project = 'coding'
logging.info("Global models initialized successfully")

@dataclass
class Config:
    API_URL: str = "http://localhost:11434/api/chat"
    MODEL_NAME: str = "deepseek-r1:32b"
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5
    MAX_TOKENS: int = 128000
    DEFAULT_TEMPERATURE: float = 0.9
    SIMILARITY_THRESHOLD: float = 0.6
    LOG_FILE: str = "llama_queries.log"
    RESPONSE_DIR: str = "responses"
    TARGET_HISTORY_TOKENS: int = 30000
    DEFAULT_SYSTEM_PROMPT: str = """You are a meticulous and perfectionist AI assistant with perfect recall. Messages in your context follow this format: '[YYYY-MM-DD HH:MM:SS] message'. When discussing conversation history, always include the exact timestamps.
    For example, if asked 'when did I ask about trees?', respond with the specific timestamp: '[2024-11-26 11:31:12] you asked "how many trees do i have in my yard?"'
    Always include full timestamps when recounting any past messages or questions."""

class ContextManager:
    def __init__(self, max_tokens: int = 16000):
        logging.info("Initializing ContextManager")
        self.max_tokens = max_tokens
        self.db_manager = None
        self.system_prompt = Config.DEFAULT_SYSTEM_PROMPT
        logging.info(f"Initial system prompt set: {self.system_prompt[:50]}...")

    def set_system_prompt(self, prompt: str):
        """Update the system prompt."""
        logging.info(f"Updating system prompt from: {self.system_prompt[:50]}...")
        logging.info(f"To new prompt: {prompt[:50]}...")
        self.system_prompt = prompt
        logging.info("System prompt updated successfully")

    def build_context(self, current_prompt: str, db_manager: DatabaseManager = None) -> List[Dict]:
        """Build context for the current prompt, trimming older messages to fit within the token limit."""
        logging.info(f"Building context for prompt: '{current_prompt[:50]}...'")
        start_time = time.time()

        # Start with the system prompt
        messages = [{"role": "system", "content": self.system_prompt}]
        total_tokens = count_tokens(self.system_prompt)  # Start with system prompt tokens

        if db_manager and db_manager.current_project:
            # Fetch all recent conversations
            recent_conversations = db_manager.get_recent_conversations(self.max_tokens)
            logging.info(f"Retrieved {len(recent_conversations)} recent conversations.")

            # Use a deque to manage context
            message_deque = deque()
            for conv in reversed(recent_conversations):  # Start with the newest messages
                user_message = {
                    "role": "user",
                    "content": f"[{conv['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] {conv['prompt']}"
                }
                assistant_message = {
                    "role": "assistant",
                    "content": conv['response']
                }

                user_tokens = count_tokens(user_message['content'])
                assistant_tokens = count_tokens(assistant_message['content'])

                # Add messages if they fit within the token limit
                if total_tokens + user_tokens + assistant_tokens <= self.max_tokens:
                    message_deque.appendleft(assistant_message)
                    message_deque.appendleft(user_message)
                    total_tokens += user_tokens + assistant_tokens
                else:
                    # Stop adding once the limit is exceeded
                    logging.info(f"Token limit reached: {total_tokens}. Stopping further additions.")
                    break

            # Append the deque contents to the context
            messages.extend(list(message_deque))

        # Add the user's current prompt
        current_prompt_tokens = count_tokens(current_prompt)
        if total_tokens + current_prompt_tokens <= self.max_tokens:
            messages.append({"role": "user", "content": current_prompt})
        else:
            logging.warning(f"Cannot include current prompt in context due to token limit.")

        logging.info(f"Context built with {len(messages)} messages and {total_tokens} tokens.")
        logging.info(f"Build time: {time.time() - start_time:.2f} seconds.")
        return messages

    def _format_messages(self, conversations: List[Dict], similar_examples: List[Tuple]) -> List[Dict]:
        logging.debug(f"Formatting messages from {len(conversations)} conversations and {len(similar_examples)} similar examples")
        messages = [
            {
                "role": "system",
                "content": "You are a meticulous and perfectionist AI assistant, dedicated to excellence in every task. With an unwavering attention to detail, you ensure that no aspect is overlooked, delivering responses that are not only precise and thoughtful but also tailored to exceed the user's needs. You approach each interaction with thoroughness and adaptability, ensuring optimal outcomes."
            }
        ]
        
        total_added = 0
        if similar_examples:
            for prompt, response, similarity in similar_examples:
                if similarity >= Config.SIMILARITY_THRESHOLD:
                    messages.extend([
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ])
                    total_added += 2
                    logging.debug(f"Added similar example with similarity score: {similarity:.3f}")
        
        for conv in conversations:
            messages.extend([
                {"role": "user", "content": conv["prompt"]},
                {"role": "assistant", "content": conv["response"]}
            ])
            total_added += 2
        
        messages = self._trim_messages(messages)
        logging.debug(f"Formatted {total_added} messages, {len(messages)} retained after trimming")
        return messages

    def _trim_messages(self, messages: List[Dict]) -> List[Dict]:
        """Trim messages to fit within token limit while preserving system message and recent context."""
        if not messages:
            return messages
            
        total_tokens = 0
        system_message = None
        trimmed_messages = []
        
        # Extract system message if present
        if messages[0]["role"] == "system":
            system_message = messages[0]
            messages = messages[1:]
            total_tokens = count_tokens(system_message["content"])
        
        # Add messages from most recent backwards until we hit token limit
        for msg in reversed(messages):
            msg_tokens = count_tokens(msg["content"])
            if total_tokens + msg_tokens <= self.max_tokens:
                trimmed_messages.insert(0, msg)
                total_tokens += msg_tokens
            else:
                break
        
        # Add system message back at the start if we had one
        if system_message:
            trimmed_messages.insert(0, system_message)
        
            logging.debug(f"Trimmed to {len(trimmed_messages)} messages with {total_tokens} total tokens")
            return trimmed_messages

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception as e:
        logging.error(f"Error counting tokens: {str(e)}")
        return len(text.split())  # Fallback to word count if tiktoken fails
    

class LLaMAAPI:
    def __init__(self, config: Config, db_manager: DatabaseManager):
        logging.info("Initializing LLaMAAPI...")
        self.config = config
        self.db_manager = db_manager
        self.context_manager = ContextManager(max_tokens=config.TARGET_HISTORY_TOKENS)
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=3,
            pool_maxsize=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        logging.info("LLaMAAPI initialized successfully")

    def update_model(self, model_name: str):
        """Update the model configuration."""
        self.config.MODEL_NAME = model_name
        logging.info(f"Updated model to: {model_name}")

    def ask_question(self, question: str, file_content: Optional[str] = None) -> Optional[str]:
        logging.info(f"Processing question: '{question[:50]}...'")
        logging.info(f"Current system prompt: {self.context_manager.system_prompt[:50]}...")
        start_time = time.time()
        
        # Build context with the database manager
        messages = self.context_manager.build_context(question, self.db_manager)
        logging.info(f"Built context with {len(messages)} messages")
        logging.info(f"System prompt in messages: {messages[0]['content'][:50]}...")
        
        # Add the current question
        if file_content:
            file_prompt = (
                "I'm providing you with the content of a file to reference. "
                "Please consider this content when answering the question that follows:\n\n"
                f"FILE CONTENT:\n{file_content}\n\n"
                "QUESTION:\n{question}"
            )
            messages.append({"role": "user", "content": file_prompt})
        else:
            messages.append({"role": "user", "content": question})
        
        payload = {
            "model": self.config.MODEL_NAME,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": 32000,
                "top_p": 0.9,
                "temperature": self.config.DEFAULT_TEMPERATURE,
                "num_ctx": self.config.MAX_TOKENS,
            }
        }
        
        answer = None
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.session.post(
                    self.config.API_URL,
                    json=payload,
                    timeout=(5, 3000)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'message' in result and 'content' in result['message']:
                        answer = result['message']['content'].strip()
                        duration = time.time() - start_time
                        logging.info(f"Got successful response in {duration:.2f} seconds")
                        
                        # Store the interaction in the database
                        if self.db_manager and self.db_manager.current_project:
                            try:
                                db_prompt = question if not file_content else f"[File included]\n{question}"
                                self.db_manager.insert_memory(db_prompt, answer)
                                logging.info("Successfully stored interaction in database")
                            except Exception as e:
                                logging.error(f"Failed to store memory: {str(e)}")
                        
                        return answer
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(self.config.RETRY_DELAY)
        
        return answer

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(text))
        logging.debug(f"Token count for text: {token_count}")
        return token_count
    except Exception as e:
        logging.error(f"Error counting tokens: {str(e)}")
        return len(text.split())  # Fallback to word count if tiktoken fails

def setup_logging():
    os.makedirs("logs", exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'
    )
    
    file_handler = logging.FileHandler(f"logs/{Config.LOG_FILE}")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info("Logging system initialized")

def main():
    setup_logging()
    logging.info("Starting application...")
    
    try:
        config = Config()
        llama_api = LLaMAAPI(config)
        db_manager = DatabaseManager()
        
        while True:
            # Get the question
            question = input("\nQuestion? (or 'quit' to exit): ").strip()
            if question.lower() == 'quit':
                break
                
            logging.info(f"Received question: '{question}'")
            
            # Ask if there's a file to include
            include_file = input("\nInclude a file? (Enter path or press Enter to skip): ").strip()
            file_content = None
            
            if include_file:
                try:
                    with open(include_file, 'r') as file:
                        file_content = file.read().strip()
                    logging.info(f"Successfully read file: {include_file}")
                    print(f"\nFile contents loaded successfully from: {include_file}")
                except Exception as e:
                    logging.error(f"Error reading file {include_file}: {str(e)}")
                    print(f"Error reading file: {str(e)}")
                    continue

            start_time = time.time()
            response = llama_api.ask_question(question, file_content)
            duration = time.time() - start_time
            
            if response:
                # Save response
                os.makedirs(config.RESPONSE_DIR, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save to database
                # If file content was included, add a note about it in the prompt
                if file_content:
                    db_prompt = f"[File included: {include_file}]\n{question}"
                else:
                    db_prompt = question
                    
                db_manager.insert_memory(db_prompt, response)
                
                # Save to file
                response_file = f"{config.RESPONSE_DIR}/response_{timestamp}.txt"
                with open(response_file, 'w') as f:
                    f.write(f"Question: {question}\n\n")
                    if file_content:
                        f.write(f"Included File: {include_file}\n\n")
                    f.write(f"Answer:\n{response}")
                
                print("\nAnswer:")
                print("=" * 80)
                print(response)
                print("=" * 80)
                
                logging.info(f"Completed processing in {duration:.2f} seconds")
                logging.info(f"Saved response to {response_file}")
            else:
                logging.error("No response received")
                
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        logging.error("Error details:", exc_info=True)
    finally:
        logging.info("Application shutting down")

if __name__ == "__main__":
    main()