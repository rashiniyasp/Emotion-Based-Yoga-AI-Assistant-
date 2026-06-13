import os
import streamlit as st
from twilio.rest import Client
from streamlit_webrtc import RTCConfiguration

@st.cache_resource
def get_ice_servers():
    """
    Fetches ICE servers from Twilio for reliable WebRTC connections bypassing strict NAT/firewalls.
    Falls back to Google STUN if Twilio credentials are not provided.
    """
    account_sid = None
    auth_token = None

    # 1. Try environment variables FIRST (Hugging Face Spaces injects secrets this way)
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")

    # 2. If env vars are empty, try Streamlit secrets (for local development)
    if not account_sid or not auth_token:
        try:
            account_sid = st.secrets.get("TWILIO_ACCOUNT_SID", None)
            auth_token = st.secrets.get("TWILIO_AUTH_TOKEN", None)
        except Exception:
            pass  # No secrets.toml file exists, that's fine

    # 3. If we have credentials, fetch TURN servers from Twilio
    if account_sid and auth_token:
        try:
            client = Client(account_sid, auth_token)
            token = client.tokens.create()
            print(f"✅ Twilio TURN servers loaded successfully! ({len(token.ice_servers)} servers)")
            return token.ice_servers
        except Exception as e:
            print(f"❌ Warning: Failed to fetch Twilio ICE servers: {e}")
    else:
        print("⚠️ No Twilio credentials found. Using Google STUN fallback (WebRTC may fail behind firewalls).")

    # Fallback to standard Google STUN if no Twilio secrets or if Twilio fails
    return [{"urls": ["stun:stun.l.google.com:19302"]}]

def get_rtc_config():
    """Returns the RTCConfiguration object initialized with the correct STUN/TURN servers."""
    ice_servers = get_ice_servers()
    return RTCConfiguration({"iceServers": ice_servers})

