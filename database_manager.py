# Create a new file called database_manager.py

import logging
import json
import mysql.connector
from typing import List, Dict, Tuple
from datetime import datetime
from sentence_transformers import SentenceTransformer
import tiktoken
import time
from typing import List, Dict, Optional, Tuple

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(text))
        logging.debug(f"Token count for text: {token_count}")
        return token_count
    except Exception as e:
        logging.error(f"Error counting tokens: {str(e)}")
        return len(text.split())

class ProjectStats:
    def __init__(self):
        self.message_count: int = 0
        self.last_used: Optional[datetime] = None
        self.description: str = ""
        self.tags: List[str] = []

class DatabaseManager:
    def __init__(self):
        logging.info("Initializing DatabaseManager...")
        self.connection_params = {
            'host': 'localhost',
            'database': 'llm_memory',
            'allow_local_infile': True
        }
        self.current_project = None
        logging.debug(f"Database connection parameters set: {self.connection_params}")

    def get_connection(self):
        """Get a database connection."""
        logging.debug("Attempting to establish database connection...")
        try:
            conn = mysql.connector.connect(**self.connection_params)
            logging.debug("Database connection established successfully")
            return conn
        except Exception as e:
            logging.error(f"Failed to establish database connection: {str(e)}")
            raise

    def get_projects(self) -> List[Tuple[str, ProjectStats]]:
        """Get all available project tables with their stats."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SHOW TABLES")
                    projects = []
                    for (table_name,) in cursor.fetchall():
                        if table_name != 'project_metadata':  # Skip metadata table
                            stats = self._get_project_stats(cursor, table_name)
                            projects.append((table_name, stats))
                    return projects
        except Exception as e:
            logging.error(f"Error getting projects: {str(e)}")
            return []

    def _get_project_stats(self, cursor, project_name: str) -> ProjectStats:
        """Get statistics for a specific project."""
        stats = ProjectStats()
        try:
            cursor.execute(f"""
                SELECT COUNT(*) as count, MAX(created_at) as last_used
                FROM {project_name}
            """)
            count, last_used = cursor.fetchone()
            stats.message_count = count
            stats.last_used = last_used

            cursor.execute("""
                SELECT metadata FROM project_metadata 
                WHERE project_name = %s
            """, (project_name,))
            result = cursor.fetchone()
            if result:
                metadata = json.loads(result[0])
                stats.description = metadata.get('description', '')
                stats.tags = metadata.get('tags', [])
        except Exception as e:
            logging.error(f"Error getting project stats: {str(e)}")
        return stats

    def set_current_project(self, project_name: str) -> bool:
        """Set the current project, creating it if it doesn't exist."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Ensure metadata table exists
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS project_metadata (
                            project_name varchar(255) PRIMARY KEY,
                            metadata JSON NOT NULL,
                            created_at timestamp DEFAULT CURRENT_TIMESTAMP,
                            updated_at timestamp DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                        )
                    """)

                    # Check if project exists
                    cursor.execute("SHOW TABLES LIKE %s", (project_name,))
                    exists = cursor.fetchone() is not None
                    
                    if not exists:
                        # Create new project table
                        create_table_sql = f"""
                        CREATE TABLE `{project_name}` (
                            `id` int NOT NULL AUTO_INCREMENT,
                            `prompt` text NOT NULL,
                            `response` text NOT NULL,
                            `embedding` blob NOT NULL,
                            `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (`id`)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
                        """
                        cursor.execute(create_table_sql)
                        
                        # Add initial metadata
                        cursor.execute("""
                            INSERT IGNORE INTO project_metadata (project_name, metadata)
                            VALUES (%s, %s)
                        """, (project_name, json.dumps({
                            'description': '',
                            'tags': []
                        })))
                        
                        conn.commit()
                        logging.info(f"Created new project table: {project_name}")
                    
                    self.current_project = project_name
                    logging.info(f"Set current project to: {project_name}")
                    return True
                    
        except Exception as e:
            logging.error(f"Error setting current project: {str(e)}")
            return False

    def get_recent_conversations(self, max_tokens: int) -> List[Dict[str, str]]:
        """Get recent conversations from current project, prioritizing latest."""
        if not self.current_project:
            logging.error("No project selected for getting conversations")
            return []

        logging.info(f"Retrieving recent conversations from {self.current_project}")
        start_time = time.time()

        try:
            conversations = []
            total_tokens = 0

            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Fetch latest conversations first
                    cursor.execute(f"""
                        SELECT prompt, response, created_at 
                        FROM {self.current_project}
                        ORDER BY created_at DESC
                        LIMIT 100
                    """)
                    rows = cursor.fetchall()

                    logging.info(f"Raw database results ({len(rows)} rows):")
                    for i, (prompt, response, timestamp) in enumerate(rows, 1):
                        logging.info(f"\nRow {i}:")
                        logging.info(f"Timestamp: {timestamp}")
                        logging.info(f"Prompt: {prompt}")
                        logging.info(f"Response: {response[:100]}...")

                        conv_tokens = count_tokens(prompt + response)
                        if total_tokens + conv_tokens <= max_tokens:
                            total_tokens += conv_tokens
                            conversations.append({
                                "prompt": prompt,
                                "response": response,
                                "timestamp": timestamp
                            })
                        else:
                            logging.info(f"Skipping remaining rows due to token limit ({total_tokens}/{max_tokens})")
                            break

            # Reverse to return conversations in chronological order
            conversations.reverse()

            duration = time.time() - start_time
            logging.info(f"Final conversations list ({len(conversations)} conversations):")
            for i, conv in enumerate(conversations, 1):
                logging.info(f"\nConversation {i}:")
                logging.info(f"Timestamp: {conv['timestamp']}")
                logging.info(f"Prompt: {conv['prompt']}")
                logging.info(f"Response: {conv['response'][:100]}...")

            logging.info(f"Retrieved {len(conversations)} conversations with {total_tokens} tokens in {duration:.2f}s")
            return conversations

        except Exception as e:
            logging.error(f"Error retrieving recent conversations: {str(e)}")
            return []

    def find_similar_prompts(self, prompt: str, top_k: int = 1) -> List[Tuple[str, str, float]]:
        """Find similar prompts in the current project."""
        if not self.current_project:
            logging.warning("No current project set")
            return []
            
        logging.info(f"Finding similar prompts for: '{prompt[:50]}...'")
        start_time = time.time()
        search_embedding = model.encode(prompt).tobytes()
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT prompt, response,
                            (
                                BIT_COUNT(CAST(embedding AS BINARY) & CAST(%s AS BINARY)) / 
                                SQRT(
                                    BIT_COUNT(CAST(embedding AS BINARY)) * 
                                    BIT_COUNT(CAST(%s AS BINARY))
                                )
                            ) AS similarity
                        FROM {self.current_project}
                        ORDER BY similarity DESC
                        LIMIT %s
                    """, (search_embedding, search_embedding, top_k))
                    
                    results = cursor.fetchall()
                    duration = time.time() - start_time
                    logging.info(f"Found {len(results)} similar prompts in {duration:.2f}s")
                    return results
        except Exception as e:
            logging.error(f"Error finding similar prompts: {str(e)}")
            return []

    def insert_memory(self, prompt: str, response: str) -> bool:
        """Insert a new memory into the current project."""
        if not self.current_project:
            logging.error("No project selected for memory insertion")
            return False
            
        logging.info(f"Inserting new memory into project {self.current_project}")
        start_time = time.time()
        
        try:
            embedding = model.encode(prompt).tobytes()
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        INSERT INTO {self.current_project} 
                        (prompt, response, embedding, created_at) 
                        VALUES (%s, %s, %s, NOW())
                    """, (prompt, response, embedding))
                    conn.commit()
                    
                    duration = time.time() - start_time
                    logging.info(f"Memory inserted successfully in {duration:.2f}s")
                    return True
        except Exception as e:
            logging.error(f"Failed to insert memory: {str(e)}")
            return False
        
    def create_project(self, project_name: str, description: str = None, tags: List[str] = None) -> bool:
        """Create a new project with metadata."""
        try:
            # First check if project already exists
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SHOW TABLES LIKE %s", (project_name,))
                    if cursor.fetchone():
                        logging.error(f"Project {project_name} already exists")
                        return False
                    
                    # Create new project table
                    create_table_sql = f"""
                    CREATE TABLE `{project_name}` (
                        `id` int NOT NULL AUTO_INCREMENT,
                        `prompt` text NOT NULL,
                        `response` text NOT NULL,
                        `embedding` blob NOT NULL,
                        `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (`id`)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
                    """
                    cursor.execute(create_table_sql)
                    
                    # Ensure metadata table exists
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS project_metadata (
                            project_name varchar(255) PRIMARY KEY,
                            metadata JSON NOT NULL,
                            created_at timestamp DEFAULT CURRENT_TIMESTAMP,
                            updated_at timestamp DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Add metadata
                    metadata = {
                        'description': description or '',
                        'tags': tags or []
                    }
                    cursor.execute("""
                        INSERT INTO project_metadata (project_name, metadata)
                        VALUES (%s, %s)
                    """, (project_name, json.dumps(metadata)))
                    
                    conn.commit()
                    logging.info(f"Successfully created new project: {project_name}")
                    return True
                    
        except Exception as e:
            logging.error(f"Error creating project: {str(e)}")
            return False    

    def update_project_metadata(self, project_name: str, description: str = None, tags: List[str] = None) -> bool:
        """Update project metadata."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT metadata FROM project_metadata
                        WHERE project_name = %s
                    """, (project_name,))
                    result = cursor.fetchone()
                    metadata = json.loads(result[0]) if result else {'description': '', 'tags': []}
                    
                    if description is not None:
                        metadata['description'] = description
                    if tags is not None:
                        metadata['tags'] = tags
                    
                    cursor.execute("""
                        INSERT INTO project_metadata (project_name, metadata)
                        VALUES (%s, %s)
                        ON DUPLICATE KEY UPDATE metadata = VALUES(metadata)
                    """, (project_name, json.dumps(metadata)))
                    
                    conn.commit()
                    return True
        except Exception as e:
            logging.error(f"Error updating project metadata: {str(e)}")
            return False