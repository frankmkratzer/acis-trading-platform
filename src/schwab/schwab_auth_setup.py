#!/usr/bin/env python3
# File: schwab_auth_setup.py

from schwab_broker import SchwabAuthManager
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Parse the authorization code
        query = urlparse(self.path).query
        params = parse_qs(query)

        if 'code' in params:
            self.server.auth_code = params['code'][0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Authorization successful! You can close this window.")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Authorization failed!")


def setup_schwab_auth():
    auth_manager = SchwabAuthManager()

    # Get authorization URL
    auth_url = auth_manager.get_authorization_url()
    print(f"Opening browser for authorization...")
    webbrowser.open(auth_url)

    # Start local server to receive callback
    server = HTTPServer(('localhost', 8080), CallbackHandler)
    server.auth_code = None

    print("Waiting for authorization...")
    while server.auth_code is None:
        server.handle_request()

    # Exchange code for tokens
    if auth_manager.exchange_code_for_tokens(server.auth_code):
        print("✅ Authentication successful! Tokens saved.")
    else:
        print("❌ Authentication failed!")


if __name__ == "__main__":
    setup_schwab_auth()