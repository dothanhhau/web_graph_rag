import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_chat_response(user_input):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Bạn là một trợ lý thân thiện."},
                {"role": "user", "content": user_input}
            ]
        )
        reply = response['choices'][0]['message']['content']
        return reply.strip()
    except Exception as e:
        return f"Lỗi: {str(e)}"