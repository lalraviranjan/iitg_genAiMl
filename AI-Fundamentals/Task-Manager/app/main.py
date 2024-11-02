from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import bcrypt
from models.database_setup import create_connection

app = FastAPI()

class LoginModel(BaseModel):
    username: str
    password: str

class TaskModel(BaseModel):
    user_id: int
    description: str

@app.post('/api/login')

def connect_db():
    conn = create_connection()
    cursor = conn.cursor()
    return conn, cursor

def is_user_registered(type, username):
    conn, cursor = connect_db()
    if type == 'login':
       cursor.execute("SELECT id, password FROM users WHERE username = ?", (username,))
    else:
        cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        
    result = cursor.fetchone()
    conn.close()
    if type == 'login':
        return (result[0], result[1]) if result else None
    else:
        return result if result else None
    

def login_user(login: LoginModel):
    user = is_user_registered('login', login.username)
    if user and bcrypt.checkpw(login.password.encode('utf-8'), user[1]):
        return {"user_id": user[0]}
    raise HTTPException(status_code=401, detail="Invalid credentials")