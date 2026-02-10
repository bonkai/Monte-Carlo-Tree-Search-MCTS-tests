"""
memory_manager.py - Handles vector storage and retrieval
"""
from sentence_transformers import SentenceTransformer
import mysql.connector
import tiktoken
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime
from config import Config

class TokenCounter:
    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoder.encode(text))

class DatabaseManager:
    def __init__(self, config: Config):
        self.config = config
        self.connection_params = {
            'host': config.db_host,
            'database': config.db_name,
            'allow_local_infile': True
        }
        
    def execute_query(self, query: str, params: tuple = None, fetch: bool = True) -> Any:
        conn = None
        cursor = None
        try:
            conn = mysql.connector.connect(**self.connection_params)
            cursor = conn.cursor()
            
            cursor.execute(query, params or ())
            
            if fetch:
                result = cursor.fetchall()
                return result
            else:
                conn.commit()
                return None
                
        except Exception as e:
            logging.error(f"Database error: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

class VectorStore:
    def __init__(self, config: Config):
        self.config = config
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.db = DatabaseManager(config)
        self.token_counter = TokenCounter()
    
    def store_memory(self, prompt: str, response: str) -> None:
        try:
            embedding = self.model.encode(prompt).tobytes()
            
            query = f"""
                INSERT INTO {self.config.table_name} 
                (prompt, response, embedding, created_at) 
                VALUES (%s, %s, %s, NOW())
            """
            
            self.db.execute_query(
                query, 
                (prompt, response, embedding), 
                fetch=False
            )
            
            logging.info(f"Successfully stored memory: {prompt[:50]}...")
        except Exception as e:
            logging.error(f"Error storing memory: {str(e)}")
            raise
    
    def find_similar(self, prompt: str, token_limit: int) -> List[Tuple[str, str, float]]:
        """Find similar prompts in vector database."""
        embedding = self.model.encode(prompt).tobytes()
        
        # Modified query to improve similarity search
        query = f"""
            WITH similarity_scores AS (
                SELECT 
                    prompt, 
                    response,
                    created_at,
                    (
                        BIT_COUNT(CAST(embedding AS BINARY) & CAST(%s AS BINARY)) / 
                        SQRT(
                            BIT_COUNT(CAST(embedding AS BINARY)) * 
                            BIT_COUNT(CAST(%s AS BINARY))
                        )
                    ) AS similarity
                FROM {self.config.table_name}
                HAVING similarity > 0.5  # Increased threshold for better matches
                ORDER BY similarity DESC, created_at DESC
                LIMIT 5  # Get more candidates
            )
            SELECT prompt, response, similarity 
            FROM similarity_scores
        """
        
        try:
            results = self.db.execute_query(query, (embedding, embedding))
            
            processed_results = []
            current_tokens = 0
            
            for prompt, response, similarity in results:
                tokens = self.token_counter.count_tokens(prompt + response)
                if current_tokens + tokens <= token_limit:
                    processed_results.append((
                        str(prompt),
                        str(response),
                        float(similarity)
                    ))
                    current_tokens += tokens
                    logging.info(f"Found relevant memory (similarity: {similarity:.2f})")
                else:
                    break
            
            return processed_results
            
        except Exception as e:
            logging.error(f"Error in similarity search: {str(e)}")
            return []

class ContextManager:
    def __init__(self, config: Config):
        self.config = config
        self.vector_store = VectorStore(config)
        self.token_counter = TokenCounter()
        self.current_conversation: List[Dict[str, str]] = []

    def build_prompt(self, question: str) -> str:
        """Build complete prompt with context and memory."""
        logging.info("Building context-aware prompt...")
        
        parts = [
            "You are a helpful AI assistant with access to conversation history. "
            "You should reference relevant past conversations when they help answer the current question. "
            "If you see relevant past context, acknowledge it and build upon it."
        ]

        # Get similar past conversations
        similar = self.vector_store.find_similar(question, self.config.memory_tokens)
        
        if similar:
            parts.append("\nRelevant past conversations:")
            for prompt, response, similarity in similar:
                if similarity > 0.5:  # Only include reasonably similar conversations
                    parts.extend([
                        f"\nPrevious Human Question: {prompt}",
                        f"Previous Assistant Response: {response}",
                    ])
                    logging.info(f"Added relevant memory with similarity {similarity:.2f}")

        # Add current conversation context
        if self.current_conversation:
            parts.append("\nCurrent conversation context:")
            for msg in self.current_conversation:
                prefix = "Human" if msg["role"] == "user" else "Assistant"
                parts.append(f"{prefix}: {msg['content']}")

        # Add current question
        parts.append(f"\nCurrent Human Question: {question}")
        parts.append("Assistant: Let me help you with that.")

        final_prompt = "\n\n".join(parts)
        
        # Log what we're doing
        logging.info(f"Built prompt with {len(similar)} relevant memories and "
                    f"{len(self.current_conversation)} current conversation messages")
        
        return final_prompt

    def add_exchange(self, prompt: str, response: str) -> None:
        """Add a new exchange to the conversation history."""
        # Add to current conversation
        self.current_conversation.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ])
        
        # Store in vector memory
        try:
            self.vector_store.store_memory(prompt, response)
            logging.info("Successfully stored exchange in vector memory")
        except Exception as e:
            logging.error(f"Failed to store in vector memory: {str(e)}")

        self._trim_conversation()

    def _trim_conversation(self) -> None:
        """Trim conversation to fit within token limit."""
        total_tokens = 0
        trimmed = []
        
        # Process in reverse to keep most recent messages
        for msg in reversed(self.current_conversation):
            tokens = self.token_counter.count_tokens(msg["content"])
            if total_tokens + tokens <= self.config.current_context_tokens:
                trimmed.insert(0, msg)
                total_tokens += tokens
            else:
                break
        
        self.current_conversation = trimmed