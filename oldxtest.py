import requests
import logging
import json
from typing import List, Dict, Optional, Tuple
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from datetime import datetime
import os
import time
import sys
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import mysql.connector
from sentence_transformers import SentenceTransformer
import tiktoken

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
current_project = 'testing'

# Configuration
@dataclass
class Config:
    API_URL: str = "http://localhost:11434/api/chat"
    # MODEL_NAME: str = "llama3.2:3b"
    MODEL_NAME: str = "qwen2.5:14b"
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5
    MAX_TOKENS: int = 50000
    DEFAULT_TEMPERATURE: float = 0.5
    SIMILARITY_THRESHOLD: float = 0.6
    LOG_FILE: str = "llama_queries.log"
    RESPONSE_DIR: str = "responses"
    TARGET_HISTORY_TOKENS: int = 16000  # Target number of tokens for conversation history
    
def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    enc = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's encoder as an approximation
    return len(enc.encode(text))

def get_recent_conversations(cursor, max_tokens: int) -> List[Dict[str, str]]:
    """Retrieve recent conversations up to max_tokens."""
    query = f"""
    SELECT prompt, response, created_at 
    FROM {current_project}
    ORDER BY created_at DESC
    LIMIT 100  -- Add a reasonable limit for initial fetch
    """
    logging.info(f"Fetching recent conversations with query: {query}")
    
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        logging.info(f"Found {len(rows)} total conversations in database")
        
        conversations = []
        total_tokens = 0
        
        for prompt, response, timestamp in rows:
            conversation = {
                "prompt": prompt,
                "response": response,
                "timestamp": timestamp
            }
            
            # Count tokens in this conversation
            conv_tokens = count_tokens(prompt + response)
            logging.info(f"Conversation at {timestamp}: {conv_tokens} tokens")
            
            # Break if we would exceed our target
            if total_tokens + conv_tokens > max_tokens:
                logging.info(f"Reached token limit. Current: {total_tokens}, Would be: {total_tokens + conv_tokens}")
                break
                
            total_tokens += conv_tokens
            conversations.append(conversation)
        
        logging.info(f"Retrieved {len(conversations)} conversations totaling {total_tokens} tokens")
        return list(reversed(conversations))  # Return in chronological order
    except Exception as e:
        logging.error(f"Error retrieving conversations: {str(e)}")
        return []

def build_context_prompt(recent_conversations: List[Dict[str, str]], 
                        similar_results: List[Tuple[str, str, float]]) -> str:
    """Build a context-aware system prompt including recent history and similar examples."""
    context_parts = ["You are a meticulous and perfectionist AI assistant, dedicated to excellence in every task. With an unwavering attention to detail, you ensure that no aspect is overlooked, delivering responses that are not only precise and thoughtful but also tailored to exceed expectations and effectively meet the user's needs. You approach each interaction with thoroughness and adaptability, ensuring optimal outcomes."]
    
    # Add recent conversation history
    if recent_conversations:
        logging.info(f"Adding {len(recent_conversations)} recent conversations to context")
        context_parts.append("\nRecent conversation history:")
        for conv in recent_conversations:
            context_parts.append(f"\nHuman: {conv['prompt']}")
            context_parts.append(f"Assistant: {conv['response']}")
    else:
        logging.warning("No recent conversations found to add to context")
    
    # Add similar examples from vector search
    if similar_results:
        logging.info(f"Adding {len(similar_results)} similar examples to context")
        context_parts.append("\nRelevant past examples:")
        for prompt, response, similarity in similar_results:
            if similarity >= Config.SIMILARITY_THRESHOLD:
                context_parts.append(f"\nSimilar question: {prompt}")
                context_parts.append(f"Response: {response}")
    
    final_prompt = "\n".join(context_parts)
    logging.info(f"Final context prompt token count: {count_tokens(final_prompt)}")
    return final_prompt

# Set up logging
def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/{Config.LOG_FILE}"),
            logging.StreamHandler()
        ]
    )

class LLaMAAPI:
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

    def ask_question(self, question: str, temperature: Optional[float] = None, 
                    system_prompt: str = "You are a meticulous and perfectionist AI assistant, dedicated to excellence in every task. With an unwavering attention to detail, you ensure that no aspect is overlooked, delivering responses that are not only precise and thoughtful but also tailored to exceed expectations and effectively meet the user's needs. You approach each interaction with thoroughness and adaptability, ensuring optimal outcomes.") -> Optional[str]:
        temperature = temperature or self.config.DEFAULT_TEMPERATURE
        
        # Parse the system prompt to extract conversation history
        messages = [{"role": "system", "content": "You are a meticulous and perfectionist AI assistant, dedicated to excellence in every task. With an unwavering attention to detail, you ensure that no aspect is overlooked, delivering responses that are not only precise and thoughtful but also tailored to exceed expectations and effectively meet the user's needs. You approach each interaction with thoroughness and adaptability, ensuring optimal outcomes."}]
        
        # Extract conversations from system prompt
        if "Recent conversation history:" in system_prompt:
            conversations = system_prompt.split("Recent conversation history:")[1]
            if "Relevant past examples:" in conversations:
                conversations = conversations.split("Relevant past examples:")[0]
            
            # Split into individual messages
            current_role = None
            current_content = []
            
            for line in conversations.strip().split('\n'):
                if line.startswith('Human: '):
                    if current_role and current_content:
                        messages.append({
                            "role": "assistant" if current_role == "Assistant" else "user",
                            "content": '\n'.join(current_content).strip()
                        })
                    current_role = "Human"
                    current_content = [line.replace('Human: ', '')]
                elif line.startswith('Assistant: '):
                    if current_role and current_content:
                        messages.append({
                            "role": "assistant" if current_role == "Assistant" else "user",
                            "content": '\n'.join(current_content).strip()
                        })
                    current_role = "Assistant"
                    current_content = [line.replace('Assistant: ', '')]
                elif line.strip():
                    current_content.append(line.strip())
            
            # Add the last message if there is one
            if current_role and current_content:
                messages.append({
                    "role": "assistant" if current_role == "Assistant" else "user",
                    "content": '\n'.join(current_content).strip()
                })
        
        # Add the current question
        messages.append({"role": "user", "content": question})
        
        # Log the messages being sent to the API
        logging.info(f"Sending {len(messages)} messages to API")
        for msg in messages:
            logging.info(f"Message role: {msg['role']}, content preview: {msg['content'][:50]}...")
        
        payload = {
            "model": self.config.MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.config.MAX_TOKENS,
            "stream": False
        }
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.session.post(
                    self.config.API_URL,
                    json=payload,
                    timeout=(5, 300)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'message' in result and 'content' in result['message']:
                        return result['message']['content'].strip()
                
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(self.config.RETRY_DELAY)
        
        return None

def select_best_answer(responses: List[Dict], method='hybrid', similarity_threshold=0.6) -> Optional[str]:
    if not responses:
        logging.error("No responses provided to analyze")
        return None
    
    response_texts = [resp['response'] for resp in responses if 'response' in resp]
    
    if not response_texts:
        logging.error("No valid response texts found in responses")
        return None

    # Calculate similarity matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(response_texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
    except Exception as e:
        logging.error(f"Error calculating similarity: {str(e)}")
        return response_texts[0] if response_texts else None

    # Find most central response (most similar to others)
    avg_similarities = np.mean(similarity_matrix, axis=1)
    best_idx = np.argmax(avg_similarities)
    
    return response_texts[best_idx]

class MCTSExecutor:
    def __init__(self, llama_api: LLaMAAPI):
        self.llama_api = llama_api
    
    def run_iterations(self, question: str, iterations: int = 5, system_prompt: str = "You are a meticulous and perfectionist AI assistant, dedicated to excellence in every task. With an unwavering attention to detail, you ensure that no aspect is overlooked, delivering responses that are not only precise and thoughtful but also tailored to exceed expectations and effectively meet the user's needs. You approach each interaction with thoroughness and adaptability, ensuring optimal outcomes.") -> List[Dict]:
        responses = []
        temperatures = np.linspace(0.3, 0.7, iterations)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_temp = {
                executor.submit(self.llama_api.ask_question, question, temp, system_prompt): temp
                for temp in temperatures
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
                        logging.info(f"Successfully received response for temperature {temp}")
                except Exception as e:
                    logging.error(f"Error in MCTS iteration: {str(e)}")
        
        return responses

def save_responses(responses: List[Dict], question: str, config: Config):
    os.makedirs(config.RESPONSE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{config.RESPONSE_DIR}/responses_{timestamp}.json"
    
    data = {
        "question": question,
        "timestamp": timestamp,
        "model": config.MODEL_NAME,
        "responses": responses
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    logging.info(f"Responses saved to {filename}")
    return filename

def load_responses(filename: str) -> Optional[List[Dict]]:
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            return data.get('responses', [])
    except Exception as e:
        logging.error(f"Error loading responses: {str(e)}")
        return None
    
def find_similar_prompt(prompt, top_k=1):
    # Generate embedding for the search prompt
    search_embedding = model.encode(prompt).tobytes()

    # Modified query using dot product and vector magnitude for cosine similarity
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

    conn = mysql.connector.connect(
        host='localhost',
        database='llm_memory',
        allow_local_infile=True
    )
    cursor = conn.cursor()

    try:
        cursor.execute(query, (search_embedding, search_embedding, top_k))
        results = cursor.fetchall()
        return results
    finally:
        cursor.close()
        conn.close()
    
def insert_memory(prompt, response):
    embedding = model.encode(prompt)
    embedding_binary = embedding.tobytes()

    conn = mysql.connector.connect(
        host='localhost',
        database='llm_memory',
        allow_local_infile=True
    )
    cursor = conn.cursor()

    # Modified to include timestamp
    query = f"INSERT INTO {current_project} (prompt, response, embedding, created_at) VALUES (%s, %s, %s, NOW())"
    cursor.execute(query, (prompt, response, embedding_binary))

    conn.commit()
    cursor.close()
    conn.close()

def main():
    setup_logging()
    config = Config()
    llama_api = LLaMAAPI(config)
    mcts_executor = MCTSExecutor(llama_api)
    
    try:
#         question = """\
# can you make that email sound more personable? especially the subject line
# """
        question = input("\nEnter your question (or 'quit' to exit): ")
        iterations = 3
        
        # Connect to database
        try:
            conn = mysql.connector.connect(
                host='localhost',
                database='llm_memory',
                allow_local_infile=True
            )
            cursor = conn.cursor()
            logging.info("Successfully connected to database")
        except Exception as e:
            logging.error(f"Failed to connect to database: {str(e)}")
            return
        
        try:
            # Check if the table exists and has the required columns
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM information_schema.columns 
                WHERE table_schema = 'llm_memory'
                AND table_name = '{current_project}'
                AND column_name = 'created_at'
            """)
            if cursor.fetchone()[0] == 0:
                logging.error(f"Table {current_project} missing created_at column")
                return
                
            # Get recent conversations
            recent_conversations = get_recent_conversations(cursor, config.TARGET_HISTORY_TOKENS)
            
            # Get similar examples
            similar_results = find_similar_prompt(question, top_k=3)
            
            # Build context-aware system prompt
            system_prompt = build_context_prompt(recent_conversations, similar_results)
            logging.info("System prompt built successfully")
            
            # Log the first few lines of the system prompt for debugging
            preview = "\n".join(system_prompt.split("\n")[:5])
            logging.info(f"System prompt preview:\n{preview}...")
            
            logging.info(f"Starting analysis with {iterations} iterations")
            responses = mcts_executor.run_iterations(question, iterations, system_prompt)
            
            if responses:
                save_responses(responses, question, config)
                best_answer = select_best_answer(responses)
                
                if best_answer:
                    print("\nBest Answer:")
                    print("=" * 80)
                    print(best_answer)
                    print("=" * 80)
                    
                    insert_memory(question, best_answer)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    with open(f"{config.RESPONSE_DIR}/best_answer_{timestamp}.txt", 'w') as f:
                        f.write(f"Question: {question}\n\n")
                        f.write(f"Best Answer:\n{best_answer}")
                else:
                    logging.error("Could not determine best answer")
            else:
                logging.error("No responses received")
                
        except Exception as e:
            logging.error(f"Error in main execution: {str(e)}")
            logging.error("Error details:", exc_info=True)
        finally:
            cursor.close()
            conn.close()
            
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        logging.error("Error details:", exc_info=True)

if __name__ == "__main__":
    main()