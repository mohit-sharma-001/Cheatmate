import urllib.request
url = "https://rbledvzqmdufmqecjskl.supabase.co/auth/v1/.well-known/jwks.json"
response = urllib.request.urlopen(url)
print(response.read())