# api/app.py
# ─────────────────────────────────────────────────────────────
# Flask application entry point
# Registers all blueprints and initializes extensions
# ─────────────────────────────────────────────────────────────

import os
from flask import Flask
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv

from api.routes_auth import auth_bp

# Load environment variables from .env
load_dotenv()


def create_app():
    app = Flask(__name__)

    # JWT secret key — used to sign and verify tokens
    app.config["JWT_SECRET_KEY"] = os.getenv(
        "JWT_SECRET_KEY", "kairosai_dev_secret"
    )

    # Initialize JWT manager with the app
    JWTManager(app)

    # Register the auth blueprint — all routes prefixed with /auth
    app.register_blueprint(auth_bp)

    # Health check endpoint — useful to verify API is running
    @app.route("/health")
    def health():
        return {"status": "KairosAI API is running"}, 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)