from flask import Blueprint, render_template, request, jsonify, Response
from app.models.chat_model import get_chat_response, generate_cypher, query_neo4j, generate_answer
import time

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/')
def index():
    return render_template('chat_view.html')

@chat_bp.route('/chatupdate')
def chatupdate():
    return render_template('chat_view_update.html')

@chat_bp.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message')
    cypher, answer = get_chat_response(user_msg)
    return jsonify({'cypher': cypher, 'answer': answer})

@chat_bp.route('/api/stream', methods=['POST'])
def chat_stream():
    user_msg = request.json.get('question')

    def generate():
        yield f"Đang chuyển câu hỏi tự nhiên thành truy vấn ngôn ngữ cypher...\n"
        cypher = generate_cypher(user_msg)
        yield f"{cypher}\n"

        yield f"\nĐang truy vấn cơ sở dữ liệu đồ thị...\n"
        res_cypher = query_neo4j(cypher)
        yield f"{res_cypher}\n"

        yield f"\nĐang sinh câu trả lời cuối cùng...\n"
        final_res = generate_answer(user_msg, res_cypher)
        yield f"$@%123{final_res}\n"

    return Response(generate(), content_type='text/plain')

@chat_bp.route('/huongdan', methods=['GET'])
def huongdan():
    return render_template('huong_dan.html')