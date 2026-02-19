"""
Authentication API for Roboto SAI
Integrates with Supabase Auth for user management
"""

import os
import logging
import requests
from fastapi import APIRouter, HTTPException, Request, Response, Depends
from fastapi.security import HTTPBearer
from pydantic import BaseModel, EmailStr
from typing import Optional
import logging
import os
from utils.supabase_client import get_async_supabase_client
from supabase._async.client import AsyncClient

logger = logging.getLogger(__name__)

router = APIRouter()

# Request models
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

class MagicLinkRequest(BaseModel):
    email: EmailStr

class UpdateUserRequest(BaseModel):
    display_name: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    email: str
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    provider: Optional[str] = None

class AuthResponse(BaseModel):
    success: bool
    user: Optional[UserResponse] = None
    message: Optional[str] = None

# Dependency to get Supabase client
async def get_supabase_client() -> AsyncClient:
    client = await get_async_supabase_client()
    if not client:
        raise HTTPException(status_code=500, detail="Supabase client not available")
    return client

@router.get("/me", response_model=AuthResponse)
async def get_me(request: Request, supabase: AsyncClient = Depends(get_supabase_client)):
    """Get current user information from session"""
    try:
        # Accept token from Authorization header or access_token cookie
        auth_header = request.headers.get("authorization")
        token = None
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
        else:
            token = request.cookies.get("access_token")

        if not token:
            return AuthResponse(success=False, message="No auth token provided")

        # Verify the token with Supabase
        user_response = await supabase.auth.get_user(token)

        if not user_response.user:
            return AuthResponse(success=False, message="Invalid token")

        user = user_response.user
        user_data = UserResponse(
            id=user.id,
            email=user.email,
            display_name=getattr(user, 'user_metadata', {}).get('display_name'),
            avatar_url=getattr(user, 'user_metadata', {}).get('avatar_url'),
            provider=getattr(user, 'app_metadata', {}).get('provider', 'supabase')
        )

        return AuthResponse(success=True, user=user_data)

    except Exception as e:
        logger.error(f"Error getting user: {e}")
        return AuthResponse(success=False, message="Failed to get user information")

@router.post("/login", response_model=AuthResponse)
async def login(
    login_data: LoginRequest,
    response: Response,
    request: Request,
    supabase: AsyncClient = Depends(get_supabase_client)
):
    """Login with email and password"""
    try:
        auth_response = await supabase.auth.sign_in_with_password({
            "email": login_data.email,
            "password": login_data.password
        })

        if not auth_response.user or not auth_response.session:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        user = auth_response.user
        user_data = UserResponse(
            id=user.id,
            email=user.email,
            display_name=getattr(user, 'user_metadata', {}).get('display_name'),
            avatar_url=getattr(user, 'user_metadata', {}).get('avatar_url'),
            provider=getattr(user, 'app_metadata', {}).get('provider', 'supabase')
        )

        # Determine secure and samesite based on environment and request scheme
        is_prod = (os.getenv("PYTHON_ENV") or "").strip().lower() == "production"
        forwarded_proto = request.headers.get("x-forwarded-proto")
        scheme = forwarded_proto or request.url.scheme
        secure = scheme == "https" or is_prod
        samesite = "none" if secure else "lax"

        # Set httpOnly cookie with the access token
        response.set_cookie(
            key="access_token",
            value=auth_response.session.access_token,
            httponly=True,
            secure=secure,
            samesite=samesite,
            max_age=3600  # 1 hour
        )

        return AuthResponse(success=True, user=user_data)

    except Exception as e:
        logger.error(f"Login failed: {e}")
        # Check for specific Supabase error messages
        error_str = str(e).lower()
        if "invalid login credentials" in error_str or "email not confirmed" in error_str:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        elif "account not found" in error_str or "user not found" in error_str:
            raise HTTPException(status_code=401, detail="Account not found")
        else:
            raise HTTPException(status_code=401, detail="Login failed")

@router.post("/register", response_model=AuthResponse)
async def register(
    register_data: RegisterRequest,
    supabase: AsyncClient = Depends(get_supabase_client)
):
    """Register a new user"""
    try:
        auth_response = await supabase.auth.sign_up({
            "email": register_data.email,
            "password": register_data.password
        })

        if not auth_response.user:
            raise HTTPException(status_code=400, detail="Registration failed")

        # Note: Supabase may require email confirmation
        return AuthResponse(
            success=True,
            message="Registration successful. Please check your email for confirmation." if not auth_response.session else "Registration successful"
        )

    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=400, detail="Registration failed")

@router.post("/magic/request", response_model=AuthResponse)
async def request_magic_link(
    magic_data: MagicLinkRequest,
    supabase: AsyncClient = Depends(get_supabase_client)
):
    """Request a magic link for passwordless login"""
    try:
        redirect_url = os.getenv("FRONTEND_ORIGIN", "http://localhost:8080").rstrip("/") + "/auth/callback"
        await supabase.auth.sign_in_with_otp({
            "email": magic_data.email,
            "options": {
                "emailRedirectTo": redirect_url
            }
        })

        return AuthResponse(
            success=True,
            message="Magic link sent to your email"
        )

    except Exception as e:
        logger.error(f"Magic link request failed: {e}")
        raise HTTPException(status_code=400, detail="Failed to send magic link")

@router.post("/logout", response_model=AuthResponse)
async def logout(
    response: Response,
    supabase: AsyncClient = Depends(get_supabase_client)
):
    """Logout the current user"""
    try:
        # Clear the cookie
        response.delete_cookie("access_token")

        # Sign out from Supabase
        await supabase.auth.sign_out()

        return AuthResponse(success=True, message="Logged out successfully")

    except Exception as e:
        logger.error(f"Logout failed: {e}")
        return AuthResponse(success=False, message="Logout failed")

@router.patch("/update", response_model=AuthResponse)
async def update_user(
    update_data: UpdateUserRequest,
    request: Request,
    supabase: AsyncClient = Depends(get_supabase_client)
):
    """Update current user profile information"""
    try:
        # Accept token from Authorization header or access_token cookie
        auth_header = request.headers.get("authorization")
        token = None
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
        else:
            token = request.cookies.get("access_token")

        if not token:
            raise HTTPException(status_code=401, detail="No auth token provided")

        # Get Supabase URL and anon key for direct API call
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_anon_key:
            raise HTTPException(status_code=500, detail="Supabase configuration missing")

        # Prepare update data with validation
        body = {}
        if update_data.display_name is not None:
            display_name = update_data.display_name.strip()
            if not display_name:
                raise HTTPException(status_code=400, detail="display_name must be a non-empty string")

            # Additional backend validation
            if len(display_name) < 2:
                raise HTTPException(status_code=400, detail="Username must be at least 2 characters long")
            if len(display_name) > 20:
                raise HTTPException(status_code=400, detail="Username cannot exceed 20 characters")
            if not display_name.replace('_', '').replace('-', '').isalnum():
                raise HTTPException(status_code=400, detail="Username can only contain letters, numbers, underscores, and hyphens")

            body["data"] = {"display_name": display_name}

        if not body:
            raise HTTPException(status_code=400, detail="No valid update fields provided")

        # Use Supabase REST API to update user metadata
        update_url = f"{supabase_url}/auth/v1/user"
        headers = {
            "apikey": supabase_anon_key,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        response = requests.put(update_url, json=body, headers=headers)

        if response.status_code != 200:
            logger.error(f"Supabase update response: {response.status_code} - {response.text}")
            raise HTTPException(status_code=400, detail=f"Supabase update failed: {response.text}")

        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Supabase update failed: {response.text}")

        # Get updated user info
        user_response = await supabase.auth.get_user(token)

        if not user_response.user:
            raise HTTPException(status_code=400, detail="Failed to fetch updated user")

        user = user_response.user
        user_data = UserResponse(
            id=user.id,
            email=user.email,
            display_name=user.user_metadata.get('display_name') if user.user_metadata else None,
            avatar_url=user.user_metadata.get('avatar_url') if user.user_metadata else None,
            provider=user.app_metadata.get('provider', 'supabase') if user.app_metadata else 'supabase'
        )

        return AuthResponse(success=True, user=user_data, message="Profile updated successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")