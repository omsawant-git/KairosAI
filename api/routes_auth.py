# api/routes_auth.py
# ─────────────────────────────────────────────────────────────
# Auth routes — register, login, logout, get current user
# bcrypt handles password hashing
# JWT handles session tokens
# ─────────────────────────────────────────────────────────────

import bcrypt
from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    get_jwt_identity
)
from sqlalchemy import text
from database.db import SessionLocal

# All routes in this file are prefixed with /auth
auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


def get_db():
    # Returns a fresh database session for this request
    return SessionLocal()


# ── REGISTER ─────────────────────────────────────────────────
@auth_bp.route("/register", methods=["POST"])
def register():
    # Expects: { "email": "...", "password": "...", "full_name": "..." }
    data      = request.get_json()
    email     = data.get("email", "").strip().lower()
    password  = data.get("password", "")
    full_name = data.get("full_name", "")

    # Validate inputs
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    db = get_db()
    try:
        # Check if this email is already registered
        existing = db.execute(
            text("SELECT id FROM users WHERE email = :email"),
            {"email": email}
        ).fetchone()

        if existing:
            return jsonify({"error": "Email already registered"}), 409

        # Hash the password with bcrypt before storing
        # Never store plain text passwords
        hashed = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt()
        )

        # Insert the new user into the database
        db.execute(
            text("""
                INSERT INTO users (email, hashed_password, full_name)
                VALUES (:email, :hashed_password, :full_name)
            """),
            {
                "email": email,
                "hashed_password": hashed.decode("utf-8"),
                "full_name": full_name
            }
        )
        db.commit()

        return jsonify({"message": "Account created successfully"}), 201

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


# ── LOGIN ─────────────────────────────────────────────────────
@auth_bp.route("/login", methods=["POST"])
def login():
    # Expects: { "email": "...", "password": "..." }
    data     = request.get_json()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    db = get_db()
    try:
        # Look up user by email
        user = db.execute(
            text("""
                SELECT id, hashed_password, full_name
                FROM users
                WHERE email = :email
            """),
            {"email": email}
        ).fetchone()

        # Return same error for wrong email or wrong password
        # so attackers can't tell which one failed
        if not user:
            return jsonify({"error": "Invalid email or password"}), 401

        # Compare submitted password against stored bcrypt hash
        password_matches = bcrypt.checkpw(
            password.encode("utf-8"),
            user.hashed_password.encode("utf-8")
        )

        if not password_matches:
            return jsonify({"error": "Invalid email or password"}), 401

        # Create JWT token — identity is user UUID as string
        # This token is sent back and stored by the client
        access_token = create_access_token(identity=str(user.id))

        return jsonify({
            "access_token": access_token,
            "user": {
                "id":        str(user.id),
                "email":     email,
                "full_name": user.full_name
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


# ── GET CURRENT USER ──────────────────────────────────────────
@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    # Returns profile of the currently logged in user
    # Requires Authorization: Bearer <token> header
    user_id = get_jwt_identity()

    db = get_db()
    try:
        user = db.execute(
            text("""
                SELECT id, email, full_name, created_at
                FROM users
                WHERE id = :id
            """),
            {"id": user_id}
        ).fetchone()

        if not user:
            return jsonify({"error": "User not found"}), 404

        return jsonify({
            "id":         str(user.id),
            "email":      user.email,
            "full_name":  user.full_name,
            "created_at": str(user.created_at)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


# ── LOGOUT ────────────────────────────────────────────────────
@auth_bp.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    # JWT is stateless — actual logout happens client-side
    # by deleting the token from storage (st.session_state in Streamlit)
    # This endpoint is a clean hook for future token blacklisting
    return jsonify({"message": "Logged out successfully"}), 200