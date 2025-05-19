from flask import Flask

def create_app():
    app = Flask(__name__, template_folder='views')

    from app.controllers.chat_controller import chat_bp
    app.register_blueprint(chat_bp)

    return app