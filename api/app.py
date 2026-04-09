# api/app.py
# ─────────────────────────────────────────────────────────────
# KairosAI Flask Application Entry Point
# Registers all blueprints and initializes extensions
# ─────────────────────────────────────────────────────────────

import os
from flask import Flask
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv

load_dotenv()


def create_app():
    app = Flask(__name__)

    # JWT configuration
    app.config["JWT_SECRET_KEY"] = os.getenv(
        "JWT_SECRET_KEY", "kairosai_dev_secret"
    )

    # Initialize JWT
    JWTManager(app)

    # Register blueprints
    from api.routes_auth        import auth_bp
    from api.routes_predictions import predictions_bp
    from api.routes_portfolio   import portfolio_bp
    from api.routes_agent       import agent_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(predictions_bp)
    app.register_blueprint(portfolio_bp)
    app.register_blueprint(agent_bp)

    # Health check
    @app.route("/health")
    def health():
        return {"status": "KairosAI API running", "version": "1.0"}, 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)