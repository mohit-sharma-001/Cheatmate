import os
from dotenv import load_dotenv
load_dotenv()
url = os.getenv("SUPABASE_URL")
print(repr(url))