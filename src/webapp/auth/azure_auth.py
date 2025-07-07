import os
import streamlit as st
import requests
from urllib.parse import urlencode
from typing import Optional, Dict, Any
import secrets
import time
from dotenv import load_dotenv
from datetime import datetime, timedelta
import threading
import re

from src.config.secrets import (
    get_azure_ad_client_id,
    get_azure_ad_client_secret,
    get_azure_ad_tenant_id,
    get_azure_ad_redirect_uri,
)

# Load environment variables at module level
load_dotenv()

# Global thread-safe auth state storage
_session_lock = threading.RLock()
_auth_states = {}

# Cleanup thread
_cleanup_thread = None
_cleanup_stop = threading.Event()


def _cleanup_expired_data():
    """Background thread to clean expired auth states"""
    while not _cleanup_stop.wait(60):  # Check every minute
        current_time = datetime.now()

        with _session_lock:
            # Clean expired auth states (older than 10 minutes)
            cutoff = current_time - timedelta(minutes=10)
            expired_states = []
            for state, timestamp in _auth_states.items():
                if timestamp <= cutoff:
                    expired_states.append(state)

            for state in expired_states:
                del _auth_states[state]


# Start cleanup thread
if _cleanup_thread is None:
    _cleanup_thread = threading.Thread(target=_cleanup_expired_data, daemon=True)
    _cleanup_thread.start()


class AzureADAuth:
    """Azure AD authentication handler for Streamlit with secure session management"""

    def __init__(self):
        self.client_id = get_azure_ad_client_id() or "your-client-id"
        self.client_secret = get_azure_ad_client_secret() or "your-client-secret"
        self.tenant_id = get_azure_ad_tenant_id() or "your-tenant-id"
        self.redirect_uri = get_azure_ad_redirect_uri() or "http://localhost:8501"

        if not all([self.client_id, self.client_secret, self.tenant_id]) or self.client_id == "your-client-id":
            st.error("Configuration Azure AD manquante. Veuillez configurer les variables d'environnement.")
            st.info(
                "Variables requises: STREAMLIT_AZURE_CLIENT_ID, "
                "STREAMLIT_AZURE_CLIENT_SECRET, STREAMLIT_AZURE_TENANT_ID"
            )
            st.stop()

    @staticmethod
    def _get_auth_states() -> Dict[str, datetime]:
        """Get thread-safe auth states"""
        with _session_lock:
            return _auth_states.copy()

    def _validate_email(self, email: str) -> bool:
        """Validate email format and domain"""
        if not email:
            return False

        # Basic email validation
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, email):
            return False

        # Check allowed domains
        allowed_domains = os.getenv("AZURE_ALLOWED_DOMAINS", "isskar.fr").split(",")
        domain = email.split("@")[1].lower()

        return domain in [d.strip().lower() for d in allowed_domains]

    def _store_state(self, state: str):
        """Store auth state in thread-safe memory"""
        with _session_lock:
            _auth_states[state] = datetime.now()

    def _verify_state(self, state: str) -> bool:
        """Verify auth state from thread-safe memory"""
        with _session_lock:
            if state in _auth_states:
                # Check if not expired
                cutoff = datetime.now() - timedelta(minutes=10)
                if _auth_states[state] > cutoff:
                    # Remove used state
                    del _auth_states[state]
                    return True
        return False

    def get_auth_url(self) -> str:
        """Generate Azure AD authorization URL"""
        # Generate fresh state for each auth request
        state = secrets.token_urlsafe(32)
        self._store_state(state)

        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": "openid profile email User.Read",
            "state": state,
            "response_mode": "query",
        }

        auth_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/authorize"
        return f"{auth_url}?{urlencode(params)}"

    def handle_callback(self, query_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle the OAuth callback and exchange code for tokens"""
        if "code" not in query_params:
            st.error("No authorization code received. Please try logging in again.")
            return None

        code = query_params["code"]
        if isinstance(code, list):
            code = code[0]

        state = query_params.get("state")
        if isinstance(state, list):
            state = state[0]

        # Check state parameter exists and matches
        if not state:
            st.error("No state parameter received. Please try logging in again.")
            return None

        if not self._verify_state(state):
            st.error("Invalid or expired authentication state. Please try logging in again.")
            return None

        # Exchange code for tokens
        token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"

        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        }

        try:
            response = requests.post(token_url, data=data, timeout=30)

            if response.status_code != 200:
                # Handle token exchange errors
                error_data = {}
                try:
                    error_data = response.json()
                except Exception as e:
                    error_msg = f"Failed to parse error response: {str(e)}"

                if "error_description" in error_data:
                    error_msg = f"Authentication failed: {error_data['error_description']}"
                elif "error" in error_data:
                    error_msg = f"Authentication failed: {error_data['error']}"
                else:
                    error_msg = f"Authentication failed: HTTP {response.status_code}"

                st.error(error_msg)
                st.info("Please try logging in again.")
                return None

            token_data = response.json()

            # Validate token response
            if "access_token" not in token_data:
                st.error("Invalid token response from Microsoft. Please try logging in again.")
                return None

            # Get user info
            user_info = self.get_user_info(token_data["access_token"])

            if user_info:
                user_email = user_info.get("mail") or user_info.get("userPrincipalName")

                # Validate email domain
                if not self._validate_email(user_email):
                    st.error("Access denied. Your email domain is not authorized. Please contact an administrator.")
                    return None

                # Store tokens and user info in session
                user_data = {
                    "email": user_email,
                    "name": user_info.get("displayName"),
                    "id": user_info.get("id"),
                    "is_admin": self._is_admin_user(user_email),
                }

                st.session_state["azure_tokens"] = token_data
                st.session_state["user"] = user_data
                st.session_state["user_id"] = user_info.get("id")

                return st.session_state["user"]
            else:
                st.error("Failed to retrieve user information. Please try logging in again.")
                return None

        except requests.RequestException as e:
            st.error(f"Network error during authentication: {str(e)}")
            st.info("Please check your internet connection and try again.")
            return None
        except Exception as e:
            st.error(f"Unexpected error during authentication: {str(e)}")
            st.info("Please try logging in again.")
            return None

        return None

    def get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Get user information from Microsoft Graph API"""
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

        try:
            response = requests.get("https://graph.microsoft.com/v1.0/me", headers=headers, timeout=30)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                st.error("Access token expired. Please log in again.")
                return None
            elif response.status_code == 403:
                st.error("Insufficient permissions to access user information.")
                return None
            else:
                st.error(f"Failed to get user info: HTTP {response.status_code}")
                return None

        except requests.RequestException as e:
            st.error(f"Network error retrieving user info: {str(e)}")
            return None

    def _is_admin_user(self, email: str) -> bool:
        """Check if user is admin based on email"""
        admin_emails = os.getenv("AZURE_ADMIN_EMAILS", "emin.calyaka@isskar.fr,nicolas.lambropoulos@isskar.fr").split(
            ","
        )
        return email.lower() in [admin.strip().lower() for admin in admin_emails]

    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return "user" in st.session_state and st.session_state["user"] is not None

    def _clear_session_state(self):
        """Clear session state without redirecting"""
        keys_to_remove = ["user", "user_id", "azure_tokens", "messages", "current_conversation_id"]
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]

    def logout(self):
        """Logout user and clear session"""
        self._clear_session_state()

        logout_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/logout?post_logout_redirect_uri={self.redirect_uri}"
        st.success("✅ Déconnexion réussie!")
        st.markdown(f'<meta http-equiv="refresh" content="2;url={logout_url}">', unsafe_allow_html=True)

    def require_auth(self, func):
        """Decorator to require authentication for a function"""

        def wrapper(*args, **kwargs):
            if not self.is_authenticated():
                self.show_login_page()
                return None
            return func(*args, **kwargs)

        return wrapper

    def show_login_page(self):
        """Display login page with username/password form"""
        st.title("🔐 Authentication Required")
        st.write("Please log in with your Microsoft account to access ISSCHAT.")

        query_params = st.query_params

        # Handle OAuth callback if present
        if "code" in query_params:
            with st.spinner("Authenticating..."):
                user = self.handle_callback(dict(query_params))
                if user:
                    st.success(f"Welcome {user['name']}!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Authentication failed. Please try again.")
                    # Show debug info if available
                    if st.checkbox("Show debug information"):
                        st.json(dict(query_params))
                    if st.button("Try Again"):
                        st.rerun()
        elif "error" in query_params:
            error_desc = query_params.get("error_description", "Authentication error")
            st.error(f"Authentication error: {error_desc}")
            # Show debug info for errors
            with st.expander("Debug information"):
                st.json(dict(query_params))
            if st.button("Try Again"):
                st.rerun()
        else:
            # Show OAuth login only
            _, col2, _ = st.columns([1, 2, 1])
            with col2:
                st.subheader("Sign in with Microsoft")

                # OAuth flow option
                auth_url = self.get_auth_url()
                st.markdown(
                    f"""
                <div style="text-align: center; margin: 2rem 0;">
                    <a href="{auth_url}" target="_self" style="
                        background-color: #0078d4;
                        color: white;
                        padding: 12px 24px;
                        text-decoration: none;
                        border-radius: 6px;
                        font-weight: bold;
                        display: inline-block;
                        font-size: 16px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        🔐 Sign in with Microsoft
                    </a>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.info("Click the button above to sign in with your @isskar.fr account.")

                # Information about multi-tab limitation
                st.warning(
                    "📝 **Note importante**: Pour des raisons de sécurité, vous devez vous reconnecter dans chaque nouvel onglet du navigateur."  # noqa : E501
                )

                # Show debug information if needed
                if st.checkbox("Show configuration debug info"):
                    self.show_debug_info()

    def show_debug_info(self):
        """Show debug information for troubleshooting"""
        st.subheader("Configuration Debug Info")

        # Check configuration
        config_status = {
            "Client ID": "✅ Set" if self.client_id and self.client_id != "your-client-id" else "❌ Missing",
            "Client Secret": "✅ Set"
            if self.client_secret and self.client_secret != "your-client-secret"
            else "❌ Missing",
            "Tenant ID": "✅ Set" if self.tenant_id and self.tenant_id != "your-tenant-id" else "❌ Missing",
            "Redirect URI": "✅ Set" if self.redirect_uri else "❌ Missing",
        }

        for key, status in config_status.items():
            st.write(f"**{key}**: {status}")

        st.subheader("Current Session State")
        session_info = {
            "Authenticated": self.is_authenticated(),
            "User in session": "user" in st.session_state,
            "Tokens in session": "azure_tokens" in st.session_state,
            "Auth states count": len(self._get_auth_states()),
        }

        for key, value in session_info.items():
            st.write(f"**{key}**: {value}")

        # Test endpoints
        st.subheader("Endpoint Accessibility")

        try:
            import requests

            # Test authorization endpoint
            auth_endpoint = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/authorize"
            token_endpoint = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"

            st.write(f"**Authorization endpoint**: {auth_endpoint}")
            st.write(f"**Token endpoint**: {token_endpoint}")

            # Test metadata endpoint
            metadata_url = f"https://login.microsoftonline.com/{self.tenant_id}/v2.0/.well-known/openid_configuration"
            try:
                response = requests.get(metadata_url, timeout=5)
                if response.status_code == 200:
                    st.write("**OpenID metadata**: ✅ Accessible")
                else:
                    st.write(f"**OpenID metadata**: ❌ HTTP {response.status_code}")
            except Exception as e:
                st.write(f"**OpenID metadata**: ❌ Error: {e}")

        except Exception as e:
            st.write(f"**Endpoint test error**: {e}")

    def get_authenticated_url(self) -> Optional[str]:
        """Get base URL for authenticated users"""
        if self.is_authenticated():
            return self.redirect_uri
        return None
