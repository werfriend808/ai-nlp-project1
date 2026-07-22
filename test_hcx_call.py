from dotenv import load_dotenv
import os
import requests

load_dotenv()
API_KEY = os.getenv("HCI__API_KEY")

MODEL = "HCX-DASH-002"
url = f"https://clovastudio.stream.ntruss.com/v3/chat-completions/{MODEL}"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
payload = {
    "messages": [{"role": "user", "content": "안녕"}]
}

response = requests.post(url, headers=headers, json=payload)
print("상태코드:", response.status_code)
print("응답:", response.json())
