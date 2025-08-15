# codec/backend/auth.py

import os
import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from google.oauth2 import id_token
from google.auth.transport import requests

# --- Configuration ---

# Load the Google Client ID from environment variables.
# This is the "Client ID for Web application" you get from the Google Cloud Console
# when you set up OAuth 2.0 credentials.
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")

# A critical check to ensure the application is configured correctly on startup.
if not GOOGLE_CLIENT_ID:
    error_msg = "GOOGLE_CLIENT_ID environment variable not set. The authentication system will not work."
    logging.critical(error_msg)
    # In a real production app, you might want to exit here. For the prototype, a log is sufficient.
    # raise ValueError(error_msg)

# --- FastAPI Security Scheme ---

# This creates a security scheme that looks for a "Bearer" token in the
# "Authorization" header of incoming requests.
http_bearer = HTTPBearer(
    bearerFormat="JWT",
    description="A Google ID Token (JWT) is required for this endpoint."
)


# --- Reusable Authentication Dependency ---

async def get_current_user_id(token: str = Depends(http_bearer)) -> str:
    """
    A FastAPI dependency that verifies a Google ID token and returns the user's unique ID.

    This function is the gatekeeper for all protected API endpoints. It performs
    the following steps:
    1. Extracts the bearer token from the Authorization header.
    2. Uses the `google-auth` library to verify the token's signature, expiration,
       and audience against Google's public keys and our client ID.
    3. If valid, it extracts and returns the 'sub' (subject) claim, which is
       Google's unique and permanent identifier for the user.
    4. If invalid, it raises a 401 Unauthorized HTTPException, denying access.

    Args:
        token: The bearer token automatically extracted by FastAPI's security system.

    Returns:
        The unique user ID (Google 'sub' claim) as a string.

    Raises:
        HTTPException: With status 401 if the token is invalid, expired, or missing.
    """
    if not GOOGLE_CLIENT_ID:
        # This prevents the app from running into an error if the env var is missing.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is not configured on the server."
        )

    try:
        # The core of the verification process.
        # `id_token.verify_oauth2_token` checks the token against Google's servers.
        id_info = id_token.verify_oauth2_token(
            token.credentials,  # The actual token string
            requests.Request(), # A transport object for making the verification request
            GOOGLE_CLIENT_ID    # The audience our token should be for
        )

        # The 'sub' (subject) field is the recommended unique identifier for the user.
        # It is permanent and never reused, even if the user's email changes.
        user_id = id_info.get('sub')
        if not user_id:
            raise ValueError("Token is valid but does not contain a 'sub' claim.")

        return user_id

    except ValueError as e:
        # This exception is raised by `verify_oauth2_token` for various reasons:
        # - The token is expired.
        # - The signature is invalid.
        # - The audience (aud) claim doesn't match our GOOGLE_CLIENT_ID.
        # - The token is malformed.
        logging.warning(f"Authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials. The token may be expired or invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        # Catch any other unexpected errors during verification.
        logging.error(f"An unexpected error occurred during token verification: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during authentication."
        )