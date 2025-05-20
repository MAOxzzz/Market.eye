import os
import sys
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

# Import CrewAI router
from agents.api import router as crewai_router

# ——— Config & Constants ———
# Fix path resolution for PyInstaller bundle
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Running as compiled executable
    bundle_dir = sys._MEIPASS
    DB_PATH = os.path.join(bundle_dir, "market_eye.db")
    print(f"Using database at: {DB_PATH}")
    
    # Create database tables if they don't exist
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Check if users table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            print("Creating users table...")
            cursor.execute("""
            CREATE TABLE users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hashed TEXT NOT NULL,
                account_status TEXT DEFAULT 'active',
                failed_login_attempts INTEGER DEFAULT 0,
                last_login TEXT
            )
            """)
            
            # Add a default admin user
            hashed_password = "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"  # "Password123"
            cursor.execute(
                "INSERT INTO users (username, email, password_hashed) VALUES (?, ?, ?)",
                ("admin", "admin@example.com", hashed_password)
            )
            
        # Check if activity_logs table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='activity_logs'")
        if not cursor.fetchone():
            print("Creating activity_logs table...")
            cursor.execute("""
            CREATE TABLE activity_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
            """)
            
        # Check if user_sessions table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_sessions'")
        if not cursor.fetchone():
            print("Creating user_sessions table...")
            cursor.execute("""
            CREATE TABLE user_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                token TEXT,
                expires_at DATETIME,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
            """)
            
        conn.commit()
        conn.close()
        print("Database setup complete")
    except Exception as e:
        print(f"Database initialization error: {str(e)}")
else:
    # Running as script
    DB_PATH = os.path.join(os.path.dirname(__file__), "market_eye.db")
    print(f"Using database at: {DB_PATH}")

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
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        raise

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
    try:
        ip_address = request.client.host if request else None
        user_agent = request.headers.get("user-agent") if request else None
        
        db.execute(
            """INSERT INTO activity_logs (user_id, action, details, ip_address, user_agent) 
            VALUES (?, ?, ?, ?, ?)""",
            (user_id, action, details, ip_address, user_agent)
        )
        db.commit()
    except Exception as e:
        print(f"Error logging activity: {str(e)}")

app = FastAPI(title="Market Eye AI")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include CrewAI router
app.include_router(crewai_router)

@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    return {"message": "Market Eye AI API is running"}

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
    try:
        row = db.execute(
            "SELECT * FROM users WHERE username = ?", (user.username,)
        ).fetchone()
        
        if not row:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        if row["account_status"] != "active":
            raise HTTPException(status_code=403, detail=f"Account is {row['account_status']}")
        
        if row["failed_login_attempts"] >= MAX_FAILED_ATTEMPTS:
            lockout_time = datetime.strptime(row["last_login"], "%Y-%m-%d %H:%M:%S") if row["last_login"] else datetime.utcnow()
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
    except Exception as e:
        print(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/logout")
async def logout(
    current_user: dict = Depends(get_current_user),
    request: Request = None,
    db: sqlite3.Connection = Depends(get_db)
):
    # Invalidate the session
    try:
        db.execute(
            "DELETE FROM user_sessions WHERE user_id = ?",
            (current_user["user_id"],)
        )
        db.commit()
        
        log_activity(db, current_user["user_id"], "logout", "User logged out", request)
        return {"message": "Logged out successfully"}
    except Exception as e:
        print(f"Logout error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

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
        log_activity(db, current_user["user_id"], "failed_password_change", "Invalid old password", request)
        raise HTTPException(status_code=400, detail="Invalid old password")
    
    # Validate new password
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if not re.search("[A-Z]", new_password):
        raise HTTPException(status_code=400, detail="Password must contain at least one uppercase letter")
    if not re.search("[a-z]", new_password):
        raise HTTPException(status_code=400, detail="Password must contain at least one lowercase letter")
    if not re.search("[0-9]", new_password):
        raise HTTPException(status_code=400, detail="Password must contain at least one number")
    
    # Update password
    hashed = pwd_ctx.hash(new_password)
    db.execute(
        "UPDATE users SET password_hashed = ? WHERE user_id = ?",
        (hashed, current_user["user_id"])
    )
    db.commit()
    
    # Invalidate all sessions for this user
    db.execute(
        "DELETE FROM user_sessions WHERE user_id = ?",
        (current_user["user_id"],)
    )
    db.commit()
    
    log_activity(db, current_user["user_id"], "password_change", "Password changed successfully", request)
    return {"message": "Password changed successfully. Please log in again."}

@app.get("/user/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    return {
        "username": current_user["username"],
        "email": current_user["email"],
        "last_login": current_user["last_login"]
    }

@app.get("/user/activity")
async def get_activity(
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
    limit: int = 10
):
    activities = db.execute(
        """SELECT action, details, timestamp FROM activity_logs 
           WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?""",
        (current_user["user_id"], limit)
    ).fetchall()
    
    return [dict(a) for a in activities]