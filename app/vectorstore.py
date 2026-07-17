import os
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Load database URL from environment variables
load_dotenv()
DB_URL = os.getenv("SUPABASE_DB_URL")

def _parse_db_url(url: str) -> dict:
    """
    Manually parses connection parameters from a postgresql:// URL.
    This handles special characters in passwords (like brackets or percent signs)
    that standard DSN parsers might choke on.
    """
    if not url.startswith("postgresql://"):
        raise ValueError("Invalid URL scheme. Must start with postgresql://")
        
    remainder = url[len("postgresql://"):]
    
    if "/" in remainder:
        authority, dbname = remainder.rsplit("/", 1)
    else:
        authority = remainder
        dbname = ""
        
    if "@" in authority:
        creds, host_port = authority.rsplit("@", 1)
    else:
        creds = ""
        host_port = authority
        
    if ":" in creds:
        username, password = creds.split(":", 1)
    else:
        username = creds
        password = ""
        
    if ":" in host_port:
        host, port = host_port.rsplit(":", 1)
        try:
            port = int(port)
        except ValueError:
            port = 5432
    else:
        host = host_port
        port = 5432
        
    # Unquote username and password to decode percent-encoded characters safely
    from urllib.parse import unquote
    username = unquote(username)
    password = unquote(password)
    
    return {
        "user": username,
        "password": password,
        "host": host,
        "port": port,
        "database": dbname
    }

def _get_connection():
    """
    Attempts to connect using the raw DB_URL first. If DSN parsing fails 
    (e.g., due to unescaped special characters in the password), it parses 
    manually and establishes the connection.
    """
    if not DB_URL:
        raise ValueError("SUPABASE_DB_URL environment variable is not set. Please set it in your .env file.")
        
    try:
        return psycopg2.connect(DB_URL)
    except psycopg2.OperationalError as e:
        err_str = str(e).lower()
        if "invalid dsn" in err_str or "percent" in err_str or "parse" in err_str:
            creds = _parse_db_url(DB_URL)
            return psycopg2.connect(
                dbname=creds['database'],
                user=creds['user'],
                password=creds['password'],
                host=creds['host'],
                port=creds['port']
            )
        raise e

def save_chunks(doc_id: str, chunks: list[str], embeddings: list[list[float]]):
    """
    Saves document chunks and their corresponding embeddings into the Supabase database.
    Inserts all chunks in a single bulk operation using psycopg2's execute_values.
    """
    conn = None
    try:
        conn = _get_connection()
        with conn.cursor() as cur:
            # Query template for bulk insert
            query = """
                INSERT INTO chunks (doc_id, chunk_id, text, embedding)
                VALUES %s
            """
            
            # Prepare rows to insert
            records = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Convert the Python float list to a pgvector string representation: '[0.1,0.2,...]'
                emb_str = f"[{','.join(map(str, embedding))}]"
                records.append((doc_id, idx, chunk, emb_str))
            
            # Insert all records in bulk
            execute_values(cur, query, records, template="(%s, %s, %s, %s::vector)")
            conn.commit()
            
    except Exception as e:
        print(f"Database error in save_chunks: {e}")
        if conn:
            conn.rollback()
        raise e
    finally:
        # Ensure connection is closed and not left open
        if conn:
            conn.close()

def search(doc_id: str, query_embedding: list[float], top_k: int = 5) -> list[str]:
    """
    Searches stored chunks for a specific doc_id using pgvector cosine distance operator (<=>).
    Returns the top_k most similar chunk texts.
    """
    conn = None
    try:
        # Format embedding as vector string
        emb_str = f"[{','.join(map(str, query_embedding))}]"
        
        conn = _get_connection()
        with conn.cursor() as cur:
            # SQL Query selecting the text, ordered by cosine distance
            query = """
                SELECT text FROM chunks
                WHERE doc_id = %s
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s
            """
            cur.execute(query, (doc_id, emb_str, top_k))
            results = cur.fetchall()
            
            # Return list of text values
            return [row[0] for row in results]
            
    except Exception as e:
        print(f"Database error in search: {e}")
        return []
    finally:
        if conn:
            conn.close()

def doc_exists(doc_id: str) -> bool:
    """
    Checks if a document has any chunks stored in the Supabase database.
    """
    conn = None
    try:
        conn = _get_connection()
        with conn.cursor() as cur:
            query = "SELECT EXISTS(SELECT 1 FROM chunks WHERE doc_id = %s)"
            cur.execute(query, (doc_id,))
            result = cur.fetchone()
            return result[0] if result else False
            
    except Exception as e:
        print(f"Database error in doc_exists: {e}")
        return False
    finally:
        if conn:
            conn.close()
