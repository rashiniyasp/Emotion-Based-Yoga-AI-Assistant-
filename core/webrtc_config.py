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
    try:
        # Check environment variables first (Hugging Face Spaces), then st.secrets
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID", st.secrets.get("TWILIO_ACCOUNT_SID", None))
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN", st.secrets.get("TWILIO_AUTH_TOKEN", None))

        if account_sid and auth_token:
            client = Client(account_sid, auth_token)
            token = client.tokens.create()
            return token.ice_servers
    except Exception as e:
        print(f"Warning: Failed to fetch Twilio ICE servers: {e}")

    # Fallback to standard Google STUN if no Twilio secrets or if Twilio fails
    return [{"urls": ["stun:stun.l.google.com:19302"]}]

def get_rtc_config():
    """Returns the RTCConfiguration object initialized with the correct STUN/TURN servers."""
    ice_servers = get_ice_servers()
    return RTCConfiguration({"iceServers": ice_servers})
