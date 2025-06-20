from datetime import datetime, timedelta
from typing import Optional, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import logging
from .config import settings
from .database import get_db

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError as e:
        logger.warning(f"Token verification failed: {e}")
        return None

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get current authenticated user from token"""
    from ..models.user import User  # Import here to avoid circular imports

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        payload = verify_token(token)
        if payload is None:
            raise credentials_exception

        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception

    return user

def require_roles(*allowed_roles):
    """Decorator to require specific user roles"""
    def decorator(current_user = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return decorator

def check_user_permissions(user, required_permission: str) -> bool:
    """Check if user has required permission"""
    # Define role-based permissions
    role_permissions = {
        "super_admin": ["*"],  # All permissions
        "admin": [
            "user_management", "system_config", "data_export", 
            "compliance_full", "monitoring_full"
        ],
        "engineer_of_record": [
            "compliance_full", "monitoring_full", "data_export",
            "tsf_management", "risk_assessment"
        ],
        "tsf_operator": [
            "monitoring_read", "data_entry", "alerts_manage"
        ],
        "regulator": [
            "compliance_read", "monitoring_read", "reports_access"
        ],
        "management": [
            "reports_access", "monitoring_read", "compliance_read"
        ],
        "consultant": [
            "monitoring_read", "data_analysis", "reports_access"
        ],
        "viewer": [
            "monitoring_read", "reports_read"
        ]
    }

    user_permissions = role_permissions.get(user.role, [])

    # Super admin has all permissions
    if "*" in user_permissions:
        return True

    # Check specific permission
    return required_permission in user_permissions
