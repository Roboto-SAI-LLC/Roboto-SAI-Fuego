"""
Stripe Payment Integration Module
Handles checkout sessions and webhooks for subscriptions.
"""

import os
import logging
import stripe
from fastapi import APIRouter, HTTPException, Header, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from .utils.supabase_client import get_supabase_client
from supabase import create_client, Client

def get_service_client() -> Optional[Client]:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)

from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
logger.debug("Stripe initialized: secret key configured=%s", bool(stripe.api_key))


class CheckoutRequest(BaseModel):
    price_id: Optional[str] = None
    success_url: str
    cancel_url: str


@router.post("/api/create-checkout-session", tags=["Payments"])
async def create_checkout_session(req: CheckoutRequest, request: Request):
    """Create a Stripe Checkout Session for subscription."""
    if not stripe.api_key:
        raise HTTPException(status_code=500, detail="Stripe not configured")

    # Get user from session (simple check, assuming auth middleware or similar logic elsewhere)
    # Ideally, we decode the session cookie here or pass the user ID from a dependency
    # For now, we'll try to get the user ID from the request state or cookie if available
    # But since this is a new module, let's just assume we need valid auth.
    
    # Simple dependency usage to get user (reusing logic from main.py if possible, or independent)
    # To keep it loosely coupled, we'll extract the user_id from the session cookie manually here
    # or rely on the frontend to pass a user_id (not secure).
    # Better approach: Use the same get_current_user dependency if we can import it, 
    # but circular imports might be an issue.
    # We will rely on the session cookie 'session_id' and query supabase.
    
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=503, detail="Database unavailable")

    session_id_cookie = request.cookies.get("session_id")
    if not session_id_cookie:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Verify session
    try:
        # Check auth_sessions (sync wrapper needed for async fastapi)
        # We'll just do a direct call assuming threaded execution or async client if available
        # The main.py uses 'run_supabase_async', let's stick to simple sync for now or replicate
        # For simplicity in this module:
        sess_res = supabase.table("auth_sessions").select("user_id").eq("id", session_id_cookie).execute()
        if not sess_res.data:
            raise HTTPException(status_code=401, detail="Invalid session")
        
        user_id = sess_res.data[0]["user_id"]
        
        # Get user email
        user_res = supabase.table("users").select("email").eq("id", user_id).execute()
        user_email = user_res.data[0]["email"] if user_res.data else None

    except Exception as e:
        logger.error(f"Auth check failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

    price_id = req.price_id or os.getenv("STRIPE_PRICE_ID")
    if not price_id:
        raise HTTPException(status_code=400, detail="Price ID not configured")

    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price": price_id,
                    "quantity": 1,
                },
            ],
            mode="subscription",
            success_url=req.success_url,
            cancel_url=req.cancel_url,
            customer_email=user_email,
            client_reference_id=user_id,
            metadata={
                "user_id": user_id
            }
        )
        return {"sessionId": checkout_session.id, "url": checkout_session.url}
    except Exception as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/stripe-webhook", tags=["Payments"])
async def stripe_webhook(request: Request, stripe_signature: str = Header(None)):
    """Handle Stripe webhooks."""
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    payload = await request.body()

    if not webhook_secret:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")

    try:
        event = stripe.Webhook.construct_event(
            payload, stripe_signature, webhook_secret
        )
    except ValueError as e:
        # Invalid payload
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Handle the event
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        handle_checkout_completed(session)
    elif event["type"] == "customer.subscription.updated":
         # Logic to update subscription status (e.g. renewals, cancellations)
         # For now, checkout.session.completed is enough to activate
         pass
    elif event["type"] == "customer.subscription.deleted":
        # Handle cancellation
        subscription = event["data"]["object"]
        handle_subscription_deleted(subscription)

    return JSONResponse(content={"status": "success"})


def handle_checkout_completed(session):
    """Activate subscription for user."""
    user_id = session.get("client_reference_id")
    subscription_id = session.get("subscription")
    customer_id = session.get("customer")
    
    if not user_id:
        # Try metadata
        user_id = session.get("metadata", {}).get("user_id")
        
    if not user_id:
        logger.error("No user_id found in session")
        return

    logger.info(f"Activating subscription for user {user_id}")
    
    # Use service client for admin updates
    supabase = get_service_client() or get_supabase_client()
    if not supabase:
        logger.error("Supabase not available for webhook")
        return

    try:
        # Update user table
        data = {
            "subscription_status": "active",
            "subscription_id": subscription_id,
            "stripe_customer_id": customer_id,
            "updated_at": datetime.now().isoformat()
        }
        
        supabase.table("users").update(data).eq("id", user_id).execute()
        logger.info(f"User {user_id} upgraded to premium")
    except Exception as e:
        logger.error(f"Failed to update user subscription: {e}")

def handle_subscription_deleted(subscription):
    """Deactivate subscription."""
    customer_id = subscription.get("customer")
    
    supabase = get_service_client() or get_supabase_client()
    if not supabase:
        return

    try:
        # Find user by strip_customer_id
        # We need to query the users table for this customer_id
        res = supabase.table("users").select("id").eq("stripe_customer_id", customer_id).execute()
        
        if res.data:
            user_id = res.data[0]["id"]
            data = {
                "subscription_status": "inactive",
                "updated_at": datetime.now().isoformat()
            }
            supabase.table("users").update(data).eq("id", user_id).execute()
            logger.info(f"Subscription canceled for user {user_id}")
        else:
            logger.warning(f"No user found for canceled subscription customer {customer_id}")
            
    except Exception as e:
        logger.error(f"Failed to cancel subscription: {e}")
