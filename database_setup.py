# database_setup.py
import mysql.connector
from config import Config

def setup_database():
    config = Config()
    
    conn = mysql.connector.connect(
        host=config.db_host,
        database=config.db_name,
        allow_local_infile=True
    )
    cursor = conn.cursor()
    
    try:
        # Create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.db_name}")
        cursor.execute(f"USE {config.db_name}")
        
        # Create table with proper schema
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {config.table_name} (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            embedding LONGBLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_created_at (created_at)
        )
        """)
        
        conn.commit()
        print("Database and table setup completed successfully")
        
    except Exception as e:
        print(f"Error setting up database: {str(e)}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    setup_database()