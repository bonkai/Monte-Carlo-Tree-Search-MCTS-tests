"""
Enhanced LLM Context Manager with Vector Memory
Handles conversation context and long-term memory for local LLMs using vector database storage.
"""

import requests
import logging
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import mysql.connector
from sentence_transformers import SentenceTransformer
import tiktoken
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants
CURRENT_PROJECT = 'site_stuff'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

@dataclass
class Config:
    """Configuration settings for the LLM context manager."""
    api_url: str = "http://localhost:11434/api/chat"
    model_name: str = "qwen2.5:14b"
    max_retries: int = 3
    retry_delay: int = 5
    max_tokens: int = 50000
    default_temperature: float = 0.5
    similarity_threshold: float = 0.6
    log_file: str = "llm_queries.log"
    response_dir: str = "responses"
    current_context_tokens: int = 20000  # Increased for 32k context
    memory_tokens: int = 12000          # Increased for 32k context

class TokenCounter:
    """Handles token counting operations."""
    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if not text:
            return 0
        return len(self.encoder.encode(text))

class DatabaseManager:
    """Handles all database operations."""
    def __init__(self):
        self.connection_params = {
            'host': 'localhost',
            'database': 'llm_memory',
            'allow_local_infile': True
        }
    
    def get_connection(self) -> mysql.connector.connection.MySQLConnection:
        """Create and return a new database connection."""
        return mysql.connector.connect(**self.connection_params)
    
    def execute_query(self, query: str, params: Tuple = None, fetch: bool = True) -> Any:
        """Execute a database query and optionally fetch results."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params or ())
            if fetch:
                return cursor.fetchall()
            conn.commit()
            return None
        finally:
            cursor.close()
            conn.close()

class VectorStore:
    """Handles vector embeddings and similarity search."""
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.db = DatabaseManager()
    
    def encode_text(self, text: str) -> bytes:
        """Encode text into vector embedding."""
        return self.model.encode(text).tobytes()
    
    def find_similar(self, prompt: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Find similar prompts in the vector database."""
        embedding = self.encode_text(prompt)
        query = f"""
            SELECT prompt, response,
                (
                    BIT_COUNT(CAST(embedding AS BINARY) & CAST(%s AS BINARY)) / 
                    SQRT(
                        BIT_COUNT(CAST(embedding AS BINARY)) * 
                        BIT_COUNT(CAST(%s AS BINARY))
                    )
                ) AS similarity
            FROM {CURRENT_PROJECT}
            HAVING similarity > %s
            ORDER BY similarity DESC
            LIMIT %s
        """
        results = self.db.execute_query(query, (embedding, embedding, 0.6, top_k))
        return [(str(r[0]), str(r[1]), float(r[2])) for r in results]
    
    def store_memory(self, prompt: str, response: str) -> None:
        """Store new memory in vector database."""
        embedding = self.encode_text(prompt)
        query = f"""
            INSERT INTO {CURRENT_PROJECT} 
            (prompt, response, embedding, created_at) 
            VALUES (%s, %s, %s, NOW())
        """
        self.db.execute_query(query, (prompt, response, embedding), fetch=False)

class ContextManager:
    """Manages conversation context and memory retrieval."""
    def __init__(self, config: Config):
        self.config = config
        self.token_counter = TokenCounter()
        self.vector_store = VectorStore()
    
    def get_recent_conversations(self) -> List[Dict[str, str]]:
        """Retrieve recent conversations up to token limit."""
        db = DatabaseManager()
        query = f"""
            SELECT prompt, response, created_at 
            FROM {CURRENT_PROJECT}
            ORDER BY created_at DESC
            LIMIT 100
        """
        rows = db.execute_query(query)
        
        conversations = []
        total_tokens = 0
        
        for prompt, response, timestamp in rows:
            conv = {
                "prompt": str(prompt),
                "response": str(response),
                "timestamp": timestamp
            }
            
            conv_tokens = self.token_counter.count_tokens(prompt + response)
            if total_tokens + conv_tokens > self.config.current_context_tokens:
                break
                
            total_tokens += conv_tokens
            conversations.append(conv)
        
        return list(reversed(conversations))
    
    def build_context_prompt(self, question: str) -> str:
        """Build complete context-aware prompt."""
        parts = [
            "You are a meticulous and perfectionist AI assistant, dedicated to excellence in every task. "
            "With an unwavering attention to detail, you ensure that no aspect is overlooked, delivering "
            "responses that are not only precise and thoughtful but also tailored to exceed expectations "
            "and effectively meet the user's needs. You approach each interaction with thoroughness and "
            "adaptability, ensuring optimal outcomes."
        ]
        
        # Add recent conversations
        recent = self.get_recent_conversations()
        if recent:
            parts.append("\nRecent conversation history:")
            for conv in recent:
                parts.extend([
                    f"\nHuman: {conv['prompt']}",
                    f"Assistant: {conv['response']}"
                ])
        
        # Add similar examples
        similar = self.vector_store.find_similar(question)
        if similar:
            parts.append("\nRelevant past examples:")
            for prompt, response, similarity in similar:
                if similarity >= self.config.similarity_threshold:
                    parts.extend([
                        f"\nSimilar question: {prompt}",
                        f"Response: {response}"
                    ])
        
        return "\n".join(parts)

class LLMClient:
    """Handles communication with the local LLM server."""
    def __init__(self, config: Config):
        self.config = config
        self.session = self._setup_session()
    
    def _setup_session(self) -> requests.Session:
        """Set up HTTP session with retry handling."""
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=3,
            pool_maxsize=3
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def ask_question(self, question: str, temperature: Optional[float] = None,
                    system_prompt: str = "") -> Optional[str]:
        """Send question to LLM and get response."""
        temperature = temperature or self.config.default_temperature
        
        # Build messages array
        messages = [{"role": "system", "content": system_prompt or "You are a helpful assistant."}]
        messages.append({"role": "user", "content": question})
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False
        }
        
        for attempt in range(self.config.max_retries):
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
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
        
        return None

class MCTSExecutor:
    """Handles multiple LLM iterations using different temperatures."""
    def __init__(self, llm_client: 'LLMClient', iterations: int = 5):
        self.llm_client = llm_client
        self.iterations = iterations
    
    def run_iterations(self, question: str, system_prompt: str) -> List[Dict[str, Any]]:
        """Run multiple iterations with different temperatures."""
        responses = []
        # Generate temperatures between 0.3 and 0.7
        temperatures = np.linspace(0.3, 0.7, self.iterations)
        
        logging.info(f"Starting MCTS with {self.iterations} iterations")
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_temp = {
                executor.submit(
                    self.llm_client.ask_question, 
                    question, 
                    temp, 
                    system_prompt
                ): temp for temp in temperatures
            }
            
            for future in as_completed(future_to_temp):
                temp = future_to_temp[future]
                try:
                    response = future.result()
                    if response:
                        responses.append({
                            "temperature": temp,
                            "response": response
                        })
                        logging.info(f"Received response for temperature {temp}")
                except Exception as e:
                    logging.error(f"Error in MCTS iteration: {str(e)}")
        
        return responses

class ResponseSelector:
    """Handles selection of best response from multiple candidates."""
    @staticmethod
    def select_best(responses: List[Dict[str, Any]]) -> Optional[str]:
        """Select best response using similarity analysis."""
        if not responses:
            return None
        
        response_texts = [r['response'] for r in responses if 'response' in r]
        if not response_texts:
            return None
            
        if len(response_texts) == 1:
            return response_texts[0]
        
        # Calculate similarity matrix
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(response_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find response most similar to others (most central)
            avg_similarities = np.mean(similarity_matrix, axis=1)
            best_idx = np.argmax(avg_similarities)
            
            logging.info(f"Selected best response from {len(response_texts)} candidates")
            return response_texts[best_idx]
        except Exception as e:
            logging.error(f"Error selecting best response: {str(e)}")
            return response_texts[0]

def save_responses(responses: List[Dict], question: str, config: Config) -> None:
    """Save all responses for analysis."""
    os.makedirs(config.response_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    data = {
        "question": question,
        "timestamp": timestamp,
        "model": config.model_name,
        "responses": responses
    }
    
    filename = f"{config.response_dir}/responses_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    logging.info(f"Saved all responses to {filename}")

def setup_logging(config: Config) -> None:
    """Set up logging configuration."""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/{config.log_file}"),
            logging.StreamHandler()
        ]
    )

def main() -> None:
    """Main execution function."""
    # Initialize components
    config = Config()
    setup_logging(config)
    context_manager = ContextManager(config)
    llm_client = LLMClient(config)
    vector_store = VectorStore()
    mcts_executor = MCTSExecutor(llm_client, iterations=5)  # Default 5 iterations
    
    try:
        # Get user input
        question = input("Enter your question: ")
        
        # Build context-aware prompt
        system_prompt = context_manager.build_context_prompt(question)
        logging.info("Context prompt built successfully")
        
        # Run multiple iterations with MCTS
        responses = mcts_executor.run_iterations(question, system_prompt)
        
        if responses:
            # Save all responses
            save_responses(responses, question, config)
            
            # Select best response
            best_response = ResponseSelector.select_best(responses)
            
            if best_response:
                # Store in vector database
                vector_store.store_memory(question, best_response)
                
                # Print response
                print("\nBest Response:")
                print("=" * 80)
                print(best_response)
                print("=" * 80)
                
                # Save to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"{config.response_dir}/best_response_{timestamp}.txt", 'w') as f:
                    f.write(f"Question: {question}\n\n")
                    f.write(f"Best Response:\n{best_response}")
            else:
                logging.error("Could not select best response")
        else:
            logging.error("No responses received from MCTS iterations")
            
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()