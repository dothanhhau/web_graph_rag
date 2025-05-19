from flask import Blueprint, render_template, request, jsonify
from app.models.chat_model import get_chat_response

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/')
def index():
    return render_template('chat_view.html')

@chat_bp.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message')
    bot_reply = get_chat_response(user_msg)
    return jsonify({'response': bot_reply})