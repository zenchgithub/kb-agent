from jose import jwt
import os
from env_loader import load_app_env

load_app_env()

JWT_SECRET = os.environ["SUPABASE_JWT_SECRET"]
token = os.environ["SUPABASE_ACCESS_TOKEN"]

payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
print(payload)
