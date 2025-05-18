import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException, status, Depends, Request, Response
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, validator
from passlib.context import CryptContext
import jwt
import secrets
import re
from email_validator import validate_email, EmailNotValidError

# Import CrewAI router - commented out due to TensorFlow/Keras compatibility issues
# from agents.api import router as crewai_router

# ——— Config & Constants ———
DB_PATH = os.path.join(os.path.dirname(__file__), "market_eye.db")
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# JWT settings
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Rate limiting settings
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 30

class UserIn(BaseModel):
    username: str
    email: EmailStr
    password: str

    @validator('username')
    def username_valid(cls, v):
        if not re.match("^[a-zA-Z0-9_-]{3,20}$", v):
            raise ValueError('Username must be 3-20 characters and contain only letters, numbers, underscore, hyphen')
        return v

    @validator('password')
    def password_valid(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search("[A-Z]", v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search("[a-z]", v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search("[0-9]", v):
            raise ValueError('Password must contain at least one number')
        return v

class Token(BaseModel):
    access_token: str
    token_type: str

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: sqlite3.Connection = Depends(get_db)):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    if user["account_status"] != "active":
        raise HTTPException(status_code=403, detail=f"Account is {user['account_status']}")
    return user

def log_activity(db: sqlite3.Connection, user_id: int, action: str, details: str = None, request: Request = None):
    ip_address = request.client.host if request else None
    user_agent = request.headers.get("user-agent") if request else None
    
    db.execute(
        """INSERT INTO activity_logs (user_id, action, details, ip_address, user_agent) 
           VALUES (?, ?, ?, ?, ?)""",
        (user_id, action, details, ip_address, user_agent)
    )
    db.commit()

app = FastAPI(title="Market Eye AI")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include CrewAI router - commented out due to TensorFlow/Keras compatibility issues
# app.include_router(crewai_router)

@app.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(user: UserIn, request: Request, db: sqlite3.Connection = Depends(get_db)):
    hashed = pwd_ctx.hash(user.password)
    try:
        db.execute(
            "INSERT INTO users (username, email, password_hashed) VALUES (?, ?, ?)",
            (user.username, user.email, hashed),
        )
        db.commit()
        user_id = db.execute(
            "SELECT user_id FROM users WHERE username = ?", (user.username,)
        ).fetchone()["user_id"]
        
        log_activity(db, user_id, "signup", "New user registration", request)
        return {"message": "User created successfully"}
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            raise HTTPException(status_code=409, detail="Username already exists")
        elif "email" in str(e):
            raise HTTPException(status_code=409, detail="Email already registered")
        raise HTTPException(status_code=409, detail="Registration failed")

@app.post("/login", response_model=Token)
async def login(user: UserIn, request: Request, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute(
        "SELECT * FROM users WHERE username = ?", (user.username,)
    ).fetchone()
    
    if not row:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if row["account_status"] != "active":
        raise HTTPException(status_code=403, detail=f"Account is {row['account_status']}")
    
    if row["failed_login_attempts"] >= MAX_FAILED_ATTEMPTS:
        lockout_time = datetime.strptime(row["last_login"], "%Y-%m-%d %H:%M:%S")
        if datetime.utcnow() < lockout_time + timedelta(minutes=LOCKOUT_DURATION_MINUTES):
            raise HTTPException(
                status_code=403, 
                detail=f"Account locked. Try again after {LOCKOUT_DURATION_MINUTES} minutes"
            )
        # Reset failed attempts after lockout period
        db.execute(
            "UPDATE users SET failed_login_attempts = 0 WHERE user_id = ?",
            (row["user_id"],)
        )
        db.commit()
    
    if not pwd_ctx.verify(user.password, row["password_hashed"]):
        db.execute(
            "UPDATE users SET failed_login_attempts = failed_login_attempts + 1 WHERE user_id = ?",
            (row["user_id"],)
        )
        db.commit()
        log_activity(db, row["user_id"], "failed_login", "Invalid password attempt", request)
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Successful login - reset failed attempts and update last login
    db.execute(
        """UPDATE users 
           SET failed_login_attempts = 0, 
               last_login = CURRENT_TIMESTAMP 
           WHERE user_id = ?""",
        (row["user_id"],)
    )
    db.commit()
    
    # Create access token
    access_token = create_access_token({"sub": user.username})
    
    # Create session
    db.execute(
        """INSERT INTO user_sessions (user_id, token, expires_at) 
           VALUES (?, ?, datetime('now', '+30 minutes'))""",
        (row["user_id"], access_token)
    )
    db.commit()
    
    log_activity(db, row["user_id"], "login", "Successful login", request)
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/logout")
async def logout(
    current_user: dict = Depends(get_current_user),
    token: str = Depends(oauth2_scheme),
    request: Request = None,
    db: sqlite3.Connection = Depends(get_db)
):
    # Invalidate the session
    db.execute(
        "UPDATE user_sessions SET is_active = FALSE WHERE token = ?",
        (token,)
    )
    db.commit()
    
    log_activity(db, current_user["user_id"], "logout", "User logged out", request)
    return {"message": "Successfully logged out"}

@app.post("/change-password")
async def change_password(
    old_password: str,
    new_password: str,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
    db: sqlite3.Connection = Depends(get_db)
):
    # Verify old password
    if not pwd_ctx.verify(old_password, current_user["password_hashed"]):
        log_activity(
            db, 
            current_user["user_id"], 
            "change_password_failed", 
            "Invalid old password",
            request
        )
        raise HTTPException(status_code=401, detail="Invalid old password")
    
    # Validate new password
    try:
        UserIn(
            username="temp",
            email="temp@temp.com",
            password=new_password
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Update password
    hashed_new = pwd_ctx.hash(new_password)
    db.execute(
        "UPDATE users SET password_hashed = ? WHERE user_id = ?",
        (hashed_new, current_user["user_id"])
    )
    db.commit()
    
    log_activity(
        db,
        current_user["user_id"],
        "change_password",
        "Password successfully changed",
        request
    )
    return {"message": "Password successfully changed"}

@app.get("/user/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    return {
        "username": current_user["username"],
        "email": current_user["email"],
        "registration_date": current_user["registration_date"],
        "last_login": current_user["last_login"],
        "account_status": current_user["account_status"]
    }

@app.get("/user/activity")
async def get_activity(
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
    limit: int = 10
):
    logs = db.execute(
        """SELECT action, details, ip_address, timestamp 
           FROM activity_logs 
           WHERE user_id = ? 
           ORDER BY timestamp DESC 
           LIMIT ?""",
        (current_user["user_id"], limit)
    ).fetchall()
    
    return [dict(log) for log in logs]

@app.get("/")
async def root():
    return {
        "app": "Market Eye AI",
        "version": "1.0.0",
        "description": "AI-Powered Stock Analysis System",
        "docs": "/docs",
        "crewai": "/crewai"
    }