import requests
import json
import webbrowser
import http.server
import socketserver
import urllib.parse
import threading
import time
import os

# Your Schwab API credentials
CLIENT_ID = "Vtbsc861GI48iT3JgAr8bp5Hvy5cVe7O"
CLIENT_SECRET = "SvMJwXrepRDQBiXr"

# OAuth endpoints
AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"

# Your redirect URI (must match what's registered with Schwab)
# For local testing, we'll use a local server
REDIRECT_URI = "http://localhost:8000/callback"

# File to store tokens
TOKEN_FILE = "schwab_tokens.json"

class OAuthCallbackHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Extract authorization code from URL
        if self.path.startswith('/callback'):
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            
            if 'code' in params:
                # Store the authorization code
                self.server.authorization_code = params['code'][0]
                
                # Send success response to browser
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Authorization Successful!</h1><p>You can close this window now.</p></body></html>")
            else:
                # Handle error
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Authorization Failed!</h1><p>No authorization code received.</p></body></html>")
        
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

def start_auth_server():
    # Start local server to receive OAuth callback
    server = socketserver.TCPServer(("", 8000), OAuthCallbackHandler)
    server.authorization_code = None
    
    # Run server in a separate thread
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    
    return server

def get_tokens():
    # Check if we already have valid tokens
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            tokens = json.load(f)
            
        # Check if access token is still valid
        if tokens.get('expires_at', 0) > time.time():
            return tokens.get('access_token')
        
        # If we have a refresh token, try to get a new access token
        if 'refresh_token' in tokens:
            print("Refreshing access token...")
            refresh_data = {
                'grant_type': 'refresh_token',
                'refresh_token': tokens['refresh_token'],
                'client_id': CLIENT_ID,
                'client_secret': CLIENT_SECRET
            }
            
            response = requests.post(TOKEN_URL, data=refresh_data)
            
            if response.status_code == 200:
                new_tokens = response.json()
                # Add expiration time
                new_tokens['expires_at'] = time.time() + new_tokens.get('expires_in', 3600)
                
                # Save tokens
                with open(TOKEN_FILE, 'w') as f:
                    json.dump(new_tokens, f)
                
                print("Token refreshed successfully!")
                return new_tokens.get('access_token')
    
    # If we don't have valid tokens, start the authorization flow
    print("Starting new authorization flow...")
    
    # Start local server
    server = start_auth_server()
    
    # Create authorization URL
    auth_params = {
        'response_type': 'code',
        'client_id': CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'scope': 'readonly'  # Adjust based on Schwab's requirements
    }
    
    auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(auth_params)}"
    
    # Open browser for user to authenticate
    print(f"Opening browser for authorization at: {auth_url}")
    webbrowser.open(auth_url)
    
    # Wait for callback
    print("Waiting for authorization...")
    timeout = 300  # 5 minutes
    start_time = time.time()
    
    while server.authorization_code is None:
        if time.time() - start_time > timeout:
            server.shutdown()
            raise Exception("Authorization timed out!")
        time.sleep(1)
    
    # We have the authorization code, now get tokens
    code = server.authorization_code
    server.shutdown()
    
    # Exchange code for tokens
    token_data = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': REDIRECT_URI,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    }
    
    response = requests.post(TOKEN_URL, data=token_data)
    
    if response.status_code != 200:
        print(f"Error getting tokens: {response.text}")
        return None
    
    tokens = response.json()
    # Add expiration time
    tokens['expires_at'] = time.time() + tokens.get('expires_in', 3600)
    
    # Save tokens
    with open(TOKEN_FILE, 'w') as f:
        json.dump(tokens, f)
    
    print("Authorization successful!")
    return tokens.get('access_token')

if __name__ == "__main__":
    access_token = get_tokens()
    if access_token:
        print(f"Access token: {access_token[:10]}...")
    else:
        print("Failed to get access token")
