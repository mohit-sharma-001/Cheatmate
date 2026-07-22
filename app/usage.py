import os
import psycopg2
from fastapi import HTTPException
from dotenv import load_dotenv

# Load database URL from environment variables
load_dotenv()
DB_URL = os.getenv("SUPABASE_DB_URL")

def _parse_db_url(url: str) -> dict:
    """
    Manually parses connection parameters from a postgresql:// URL.
    This handles special characters in passwords (like brackets or percent signs)
    that standard DSN parsers might choke on. Matches app/vectorstore.py.
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
    Attempts to connect using the raw DB_URL first. If DSN parsing fails,
    it parses manually and establishes the connection. Matches app/vectorstore.py.
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

def check_and_increment_upload(identifier: str, is_guest: bool) -> None:
    """
    Checks if the user (or guest) has hit their daily upload limit.
    If limit is reached, raises HTTPException(429).
    Otherwise, increments or inserts the count for today.
    """
    limit = 2 if is_guest else 5
    conn = None
    try:
        conn = _get_connection()
        with conn.cursor() as cur:
            # Check current upload count for today
            query_check = """
                SELECT upload_count FROM upload_usage
                WHERE identifier = %s AND usage_date = CURRENT_DATE
            """
            cur.execute(query_check, (identifier,))
            row = cur.fetchone()
            
            if row and row[0] >= limit:
                if is_guest:
                    raise HTTPException(
                        status_code=429,
                        detail="You've hit today's upload limit. Try again tomorrow, or log in for a higher limit."
                    )
                else:
                    raise HTTPException(
                        status_code=429,
                        detail="You've hit today's upload limit. Try again tomorrow."
                    )
            
            # Insert a new row, or increment upload_count if one already exists for today
            query_upsert = """
                INSERT INTO upload_usage (identifier, usage_date, upload_count)
                VALUES (%s, CURRENT_DATE, 1)
                ON CONFLICT (identifier, usage_date)
                DO UPDATE SET upload_count = upload_usage.upload_count + 1
            """
            cur.execute(query_upsert, (identifier,))
            conn.commit()
            
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error checking/incrementing upload usage: {e}")
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Database error checking upload usage: {str(e)}"
        )
    finally:
        if conn:
            conn.close()

def check_and_add_document_to_conversation(conversation_id: str | None, doc_id: str) -> None:
    """
    Checks if the conversation already contains 5 documents.
    If so, raises HTTPException(400).
    Otherwise, links the document to the conversation.
    """
    if conversation_id is None:
        return
        
    conn = None
    try:
        conn = _get_connection()
        with conn.cursor() as cur:
            # Count existing documents for this conversation
            query_count = """
                SELECT COUNT(*) FROM conversation_documents
                WHERE conversation_id = %s
            """
            cur.execute(query_count, (conversation_id,))
            count = cur.fetchone()[0]
            
            if count >= 5:
                raise HTTPException(
                    status_code=400,
                    detail="This chat already has the maximum of 5 documents. Start a new chat to upload more."
                )
            
            # Insert linking row
            query_insert = """
                INSERT INTO conversation_documents (conversation_id, doc_id, added_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (conversation_id, doc_id) DO NOTHING
            """
            cur.execute(query_insert, (conversation_id, doc_id))
            conn.commit()
            
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error checking/adding document to conversation: {e}")
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Database error linking document to conversation: {str(e)}"
        )
    finally:
        if conn:
            conn.close()

def check_and_increment_download(identifier: str, is_guest: bool) -> None:
    """
    Checks if the user (or guest) has hit their daily download limit.
    If limit is reached, raises HTTPException(429).
    Otherwise, increments or inserts the count for today in the download_usage table.
    """
    limit = 2 if is_guest else 5
    conn = None
    try:
        conn = _get_connection()
        with conn.cursor() as cur:
            # Check current download count for today
            query_check = """
                SELECT download_count FROM download_usage
                WHERE identifier = %s AND usage_date = CURRENT_DATE
            """
            cur.execute(query_check, (identifier,))
            row = cur.fetchone()
            
            if row and row[0] >= limit:
                raise HTTPException(
                    status_code=429,
                    detail="You've hit today's download limit. Try again tomorrow." + (" Log in for a higher limit." if is_guest else "")
                )
            
            # Insert a new row, or increment download_count if one already exists for today
            query_upsert = """
                INSERT INTO download_usage (identifier, usage_date, download_count)
                VALUES (%s, CURRENT_DATE, 1)
                ON CONFLICT (identifier, usage_date)
                DO UPDATE SET download_count = download_usage.download_count + 1
            """
            cur.execute(query_upsert, (identifier,))
            conn.commit()
            
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error checking/incrementing download usage: {e}")
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Database error checking download usage: {str(e)}"
        )
    finally:
        if conn:
            conn.close()

