import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict


client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com",
)

