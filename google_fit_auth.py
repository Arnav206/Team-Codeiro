# -*- coding: utf-8 -*-
"""
Google Fit Multi-User Authentication Engine
Manages OAuth 2.0 authorization and token persistence per user.
"""

import os
import sys
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# Scopes required to fetch comprehensive health, fitness, and vital telemetry
SCOPES = [
    "https://www.googleapis.com/auth/fitness.activity.read",
    "https://www.googleapis.com/auth/fitness.body.read",
    "https://www.googleapis.com/auth/fitness.sleep.read",
    "https://www.googleapis.com/auth/fitness.blood_pressure.read",
    "https://www.googleapis.com/auth/fitness.blood_glucose.read",
    "https://www.googleapis.com/auth/fitness.oxygen_saturation.read",
    "https://www.googleapis.com/auth/fitness.heart_rate.read"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_SECRET_FILE = os.path.join(BASE_DIR, "client_secret.json")
TOKENS_DIR = os.path.join(BASE_DIR, "tokens")

os.makedirs(TOKENS_DIR, exist_ok=True)

def get_token_path(user_id: str) -> str:
    """Return token filepath for a given user ID (e.g. user1, user2)."""
    clean_id = "".join(c for c in user_id if c.isalnum() or c in ("-", "_"))
    return os.path.join(TOKENS_DIR, f"{clean_id}_token.json")

def authenticate_user(user_id: str = "user1", port: int = 8080) -> Credentials:
    """
    Authenticate a specific user ID against Google Fitness API.
    Reuses existing valid/refreshable tokens, or launches a local OAuth server.
    """
    token_path = get_token_path(user_id)
    creds = None

    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
            print(f"[AUTH] Existing token loaded for '{user_id}'.")
        except Exception as e:
            print(f"[AUTH] Could not load token for '{user_id}': {e}. Re-authenticating...")
            creds = None

    # Refresh expired credentials if refresh_token is present
    if creds and creds.expired and creds.refresh_token:
        try:
            print(f"[AUTH] Refreshing token for '{user_id}'...")
            creds.refresh(Request())
            with open(token_path, "w", encoding="utf-8") as token_file:
                token_file.write(creds.to_json())
            print(f"[AUTH] Token refreshed successfully for '{user_id}'.")
            return creds
        except Exception as e:
            print(f"[AUTH] Failed to refresh token: {e}. Launching new OAuth flow...")
            creds = None

    # Perform full OAuth login if no valid credentials exist
    if not creds or not creds.valid:
        if not os.path.exists(CLIENT_SECRET_FILE):
            raise FileNotFoundError(
                f"Client secret file not found at: {CLIENT_SECRET_FILE}. "
                "Ensure client_secret.json exists in the project root."
            )

        print(f"\n=======================================================")
        print(f" Initiating Google OAuth Login for User: '{user_id}'")
        print(f" Redirect URI: http://localhost:{port}/")
        print(f"=======================================================")

        flow = InstalledAppFlow.from_client_secrets_file(
            CLIENT_SECRET_FILE,
            scopes=SCOPES
        )
        flow.redirect_uri = f"http://localhost:{port}/"

        # Ensure port 8080 is free
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass

        flow.redirect_uri = f"http://localhost:{port}/"

        try:
            creds = flow.run_local_server(
                port=port,
                prompt="consent",
                authorization_prompt_message="Opening browser for Google Fit authentication...",
                success_message="Authentication successful! You can close this tab and return to the terminal."
            )
        except Exception as e:
            print(f"\n[AUTH ERROR] OAuth flow failed: {e}")
            print(f"Make sure 'http://localhost:{port}/' is added to Authorized redirect URIs in Google Cloud Console.")
            raise

        # Save credentials for future seamless runs
        with open(token_path, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())
        print(f"[AUTH] Credentials saved successfully to: {token_path}\n")

    return creds

def list_authenticated_users():
    """List all users who have an authenticated token file."""
    if not os.path.exists(TOKENS_DIR):
        return []
    files = [f for f in os.listdir(TOKENS_DIR) if f.endswith("_token.json")]
    return [f.replace("_token.json", "") for f in files]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Google Fit User Authentication")
    parser.add_argument("--user", type=str, default="user1", help="User ID to authenticate (e.g. user1, user2)")
    parser.add_argument("--port", type=int, default=8080, help="Local redirect port (default: 8080)")
    parser.add_argument("--list", action="store_true", help="List all currently authenticated users")
    args = parser.parse_args()

    if args.list:
        users = list_authenticated_users()
        print(f"Authenticated Users ({len(users)}): {', '.join(users) if users else 'None'}")
    else:
        authenticate_user(user_id=args.user, port=args.port)
