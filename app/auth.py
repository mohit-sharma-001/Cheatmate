import os
import jwt
from jwt import PyJWKClient
from fastapi import HTTPException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")

if not SUPABASE_URL:
    # Print a warning but do not crash immediately so local testing without a DB
    # or URL configured can still load the module if needed, but raise on use.
    print("Warning: SUPABASE_URL not set in environment.")

# Cache the JWKS client so we do not fetch the JWKS metadata on every single request.
jwks_client = None
if SUPABASE_URL:
    jwks_url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json"
    jwks_client = PyJWKClient(jwks_url)

def get_user_id(authorization: str | None) -> str | None:
    """
    Extracts and decodes the JWT token from the Authorization header.
    Returns the user's UUID (the 'sub' claim) if valid, or None if the header
    is missing/invalid (meaning the user is a guest).
    Raises HTTPException(401) if the token is invalid or expired.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
        
    token = authorization.split(" ")[1]
    
    if not jwks_client:
        raise HTTPException(
            status_code=500,
            detail="Supabase URL is not configured. Cannot verify authentication."
        )
        
    try:
        # Fetch the signing key corresponding to the key ID in the JWT header
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        
        # Decode and verify token
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256", "RS256"],
            audience="authenticated"
        )
        
        # Return the sub claim which corresponds to the user's UUID
        return payload.get("sub")
        
    except (jwt.PyJWTError, Exception) as e:
        print(f"JWT Verification failed: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session, please log in again"
        )

def get_identifier(authorization: str | None, guest_id: str | None) -> str:
    """
    Retrieves the unique identifier for usage limits.
    Returns the user's UUID if logged in, or the guest_id if a guest.
    Raises HTTPException(400) if neither is available.
    """
    user_id = get_user_id(authorization)
    if user_id:
        return user_id
        
    if guest_id:
        return guest_id
        
    raise HTTPException(
        status_code=400,
        detail="Missing guest identifier"
    )
