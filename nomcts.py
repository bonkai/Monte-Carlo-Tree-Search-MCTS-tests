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
import mysql.connector
from sentence_transformers import SentenceTransformer
import tiktoken

# Initialize global models
logging.info("Initializing global models...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
current_project = 'testing'
logging.info("Global models initialized successfully")

@dataclass
class Config:
    API_URL: str = "http://localhost:11434/api/chat"
    MODEL_NAME: str = "qwen2.5:14b"
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5
    MAX_TOKENS: int = 50000
    DEFAULT_TEMPERATURE: float = 0.5
    SIMILARITY_THRESHOLD: float = 0.6
    LOG_FILE: str = "llama_queries.log"
    RESPONSE_DIR: str = "responses"
    TARGET_HISTORY_TOKENS: int = 16000

class DatabaseManager:
    def __init__(self):
        logging.info("Initializing DatabaseManager...")
        self.connection_params = {
            'host': 'localhost',
            'database': 'llm_memory',
            'allow_local_infile': True
        }
        logging.debug(f"Database connection parameters set: host={self.connection_params['host']}, database={self.connection_params['database']}")

    def get_connection(self):
        logging.debug("Attempting to establish database connection...")
        try:
            conn = mysql.connector.connect(**self.connection_params)
            logging.debug("Database connection established successfully")
            return conn
        except Exception as e:
            logging.error(f"Failed to establish database connection: {str(e)}")
            raise

    def find_similar_prompts(self, prompt: str, top_k: int = 1) -> List[Tuple[str, str, float]]:
        logging.info(f"Finding similar prompts for: '{prompt[:50]}...' (top_k={top_k})")
        start_time = time.time()
        search_embedding = model.encode(prompt).tobytes()
        query = f"""
            SELECT prompt, response,
                (
                    BIT_COUNT(CAST(embedding AS BINARY) & CAST(%s AS BINARY)) / 
                    SQRT(
                        BIT_COUNT(CAST(embedding AS BINARY)) * 
                        BIT_COUNT(CAST(%s AS BINARY))
                    )
                ) AS similarity
            FROM {current_project}
            ORDER BY similarity DESC
            LIMIT %s
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (search_embedding, search_embedding, top_k))
                    results = cursor.fetchall()
                    duration = time.time() - start_time
                    logging.info(f"Found {len(results)} similar prompts in {duration:.2f} seconds")
                    return results
        except Exception as e:
            logging.error(f"Error finding similar prompts: {str(e)}")
            return []

    def insert_memory(self, prompt: str, response: str):
        logging.info(f"Inserting new memory - Prompt: '{prompt[:50]}...'")
        start_time = time.time()
        
        try:
            embedding = model.encode(prompt).tobytes()
            query = f"INSERT INTO {current_project} (prompt, response, embedding, created_at) VALUES (%s, %s, %s, NOW())"
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (prompt, response, embedding))
                    conn.commit()
                    duration = time.time() - start_time
                    logging.info(f"Memory inserted successfully in {duration:.2f} seconds")
        except Exception as e:
            logging.error(f"Failed to insert memory: {str(e)}")
            raise

    def get_recent_conversations(self, max_tokens: int) -> List[Dict[str, str]]:
        logging.info(f"Retrieving recent conversations (max_tokens={max_tokens})")
        start_time = time.time()
        
        try:
            conversations = []
            total_tokens = 0
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT prompt, response, created_at 
                        FROM {current_project}
                        ORDER BY created_at DESC
                        LIMIT 100
                    """)
                    rows = cursor.fetchall()
                    
                    for prompt, response, timestamp in rows:
                        conv_tokens = count_tokens(prompt + response)
                        if total_tokens + conv_tokens > max_tokens:
                            break
                        
                        total_tokens += conv_tokens
                        conversations.append({
                            "prompt": prompt,
                            "response": response,
                            "timestamp": timestamp
                        })
            
            duration = time.time() - start_time
            logging.info(f"Retrieved {len(conversations)} conversations with {total_tokens} total tokens in {duration:.2f} seconds")
            return list(reversed(conversations))
        except Exception as e:
            logging.error(f"Error retrieving recent conversations: {str(e)}")
            return []

class ContextManager:
    def __init__(self, max_tokens: int = 16000):
        logging.info(f"Initializing ContextManager with max_tokens={max_tokens}")
        self.max_tokens = max_tokens
        self.db_manager = DatabaseManager()

    def build_context(self, current_prompt: str) -> List[Dict]:
        logging.info(f"Building context for prompt: '{current_prompt[:50]}...'")
        start_time = time.time()
        
        recent = self.db_manager.get_recent_conversations(self.max_tokens // 2)
        similar = self.db_manager.find_similar_prompts(current_prompt, top_k=3)
        
        messages = self._format_messages(recent, similar)
        duration = time.time() - start_time
        logging.info(f"Context built with {len(messages)} messages in {duration:.2f} seconds")
        return messages

    def _format_messages(self, conversations: List[Dict], similar_examples: List[Tuple]) -> List[Dict]:
        logging.debug(f"Formatting messages from {len(conversations)} conversations and {len(similar_examples)} similar examples")
        messages = [
            {
                "role": "system",
                "content": "You are a meticulous and perfectionist AI assistant, dedicated to excellence in every task. With an unwavering attention to detail, you ensure that no aspect is overlooked, delivering responses that are not only precise and thoughtful but also tailored to exceed expectations and effectively meet the user's needs. You approach each interaction with thoroughness and adaptability, ensuring optimal outcomes."
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
        logging.debug(f"Trimming messages to fit within {self.max_tokens} tokens")
        total_tokens = 0
        included_messages = []
        
        for msg in reversed(messages):
            msg_tokens = count_tokens(msg["content"])
            if total_tokens + msg_tokens <= self.max_tokens:
                included_messages.insert(0, msg)
                total_tokens += msg_tokens
            else:
                break
        
        logging.debug(f"Trimmed to {len(included_messages)} messages with {total_tokens} total tokens")
        return included_messages

class LLaMAAPI:
    def __init__(self, config: Config):
        logging.info("Initializing LLaMAAPI...")
        self.config = config
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

    def ask_question(self, question: str) -> Optional[str]:
        logging.info(f"Processing question: '{question[:50]}...'")
        start_time = time.time()
        
        messages = self.context_manager.build_context(question)
        messages.append({"role": "user", "content": question})
        
        payload = {
            "model": self.config.MODEL_NAME,
            "messages": messages,
            "temperature": self.config.DEFAULT_TEMPERATURE,
            "max_tokens": self.config.MAX_TOKENS,
            "stream": False
        }
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                logging.debug(f"Attempt {attempt + 1}/{self.config.MAX_RETRIES} to get response")
                response = self.session.post(
                    self.config.API_URL,
                    json=payload,
                    timeout=(5, 300)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'message' in result and 'content' in result['message']:
                        duration = time.time() - start_time
                        answer = result['message']['content'].strip()
                        logging.info(f"Got successful response in {duration:.2f} seconds")
                        logging.debug(f"Response preview: '{answer[:50]}...'")
                        return answer
                else:
                    logging.warning(f"Received non-200 status code: {response.status_code}")
                    
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config.MAX_RETRIES - 1:
                    logging.info(f"Waiting {self.config.RETRY_DELAY} seconds before retry...")
                    time.sleep(self.config.RETRY_DELAY)
        
        logging.error("All attempts failed to get response")
        return None

    def decompose_task(self, task: str) -> Optional[List[str]]:
        logging.info(f"Decomposing task: '{task[:50]}...'")
        
        decomposition_prompt = {
            "role": "system",
            "content": """You are an assistant designed to analyze tasks and provide structured plans. Your role is to understand complex tasks, clarify their objectives, and break them into manageable sub-tasks without attempting to solve them or provide direct answers. Always focus on thorough analysis and logical structuring."""
        }
        
        messages = [
            decomposition_prompt,
{
    "role": "user",
    "content": (
        f"I need help analyzing the following task. Please follow these steps:\n"
        "1. Understand the Task: Identify the purpose of the task and provide a precise description of what is expected to be returned as the outcome. Clarify any ambiguity or complexity in the objectives.\n"
        "2. Plan the Solution Path: Break the task into smaller, simpler sub-tasks, explaining their roles in achieving the overall goal. Address specific complexities and ensure the sequence is logical.\n"
        "Do not attempt to solve, calculate, or speculate.\n"
        f"Task: {task}"
    )
}

        ]
        
        payload = {
            "model": self.config.MODEL_NAME,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": self.config.MAX_TOKENS,
            "stream": False
        }
        
        try:
            response = self.session.post(
                self.config.API_URL,
                json=payload,
                timeout=(5, 300)
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'message' in result and 'content' in result['message']:
                    content = result['message']['content'].strip()
                    
                    try:
                        # Try to parse as JSON directly first
                        subtasks = json.loads(content)
                        if isinstance(subtasks, list) and all(isinstance(x, str) for x in subtasks):
                            logging.info(f"Successfully decomposed task into {len(subtasks)} subtasks")
                            return subtasks
                    except json.JSONDecodeError:
                        # If that fails, try to extract array-like content
                        import re
                        match = re.search(r'\[(.*)\]', content, re.DOTALL)
                        if match:
                            try:
                                array_content = f"[{match.group(1)}]"
                                subtasks = json.loads(array_content)
                                if isinstance(subtasks, list) and all(isinstance(x, str) for x in subtasks):
                                    logging.info(f"Successfully parsed cleaned response into {len(subtasks)} subtasks")
                                    return subtasks
                            except json.JSONDecodeError:
                                logging.error("Failed to parse cleaned response")
                        
                        logging.error(f"Raw response content: {content}")
                        return None
            
            logging.error("Failed to get valid decomposition response")
            return None
            
        except Exception as e:
            logging.error(f"Error in task decomposition: {str(e)}")
            return None

def count_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(text))
        logging.debug(f"Token count for text: {token_count}")
        return token_count
    except Exception as e:
        logging.error(f"Error counting tokens: {str(e)}")
        return 0

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
        
        question = input("\nSup?: ")
        logging.info(f"Received question: '{question}'")

        # First, decompose the task
        subtasks = llama_api.decompose_task(question)
        
        if subtasks:
            print("\nDecomposed into subtasks:")
            print("=" * 80)
            for i, subtask in enumerate(subtasks, 1):
                print(f"{i}. {subtask}")
            print("=" * 80)
            
        
        start_time = time.time()
        response = llama_api.ask_question(question)
        duration = time.time() - start_time
        
        if response:
            # Save response
            os.makedirs(config.RESPONSE_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save to database
            db_manager.insert_memory(question, response)
            
            # Save to file
            response_file = f"{config.RESPONSE_DIR}/response_{timestamp}.txt"
            with open(response_file, 'w') as f:
                f.write(f"Question: {question}\n\n")
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