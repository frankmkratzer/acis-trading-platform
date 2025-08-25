# backend/auth.py
# Authentication and authorization for ACIS Trading Platform

import os
import jwt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
import logging

from .schemas import TokenData

# Configure logging
logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable is required")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer token security
security = HTTPBearer()

# Fake user database (replace with real database in production)
fake_users_db = {
    "admin": {
        "username": "admin",
        "email": "admin@acis.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "admin123"
        "disabled": False,
        "role": "admin"
    },
    "trader": {
        "username": "trader",
        "email": "trader@acis.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "trader123"
        "disabled": False,
        "role": "trader"
    }
}


class User:
    def __init__(self, username: str, email: str, disabled: bool = False, role: str = "user"):
        self.username = username
        self.email = email
        self.disabled = disabled
        self.role = role


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[User]:
    """Get user from database (currently fake db)"""
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return User(
            username=user_dict["username"],
            email=user_dict["email"],
            disabled=user_dict["disabled"],
            role=user_dict["role"]
        )
    return None


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate user credentials"""
    user = get_user(username)
    if not user:
        logger.warning(f"Authentication failed: User '{username}' not found")
        return None

    user_dict = fake_users_db[username]
    if not verify_password(password, user_dict["hashed_password"]):
        logger.warning(f"Authentication failed: Invalid password for user '{username}'")
        return None

    if user.disabled:
        logger.warning(f"Authentication failed: User '{username}' is disabled")
        return None

    logger.info(f"User '{username}' authenticated successfully")
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})

    try:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        logger.info(f"Access token created for user: {data.get('sub')}")
        return encoded_jwt
    except Exception as e:
        logger.error(f"Failed to create access token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create access token"
        )


def decode_access_token(token: str) -> Optional[TokenData]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")

        if username is None:
            logger.warning("Token payload missing 'sub' field")
            return None

        token_data = TokenData(username=username)
        return token_data

    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        return None
    except jwt.JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        return None


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        token_data = decode_access_token(token)

        if token_data is None or token_data.username is None:
            raise credentials_exception

        username = token_data.username

    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise credentials_exception

    user = get_user(username=username)
    if user is None:
        logger.warning(f"User '{username}' not found after token validation")
        raise credentials_exception

    if user.disabled:
        logger.warning(f"User '{username}' is disabled")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled"
        )

    return user