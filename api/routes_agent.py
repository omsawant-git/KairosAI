# api/routes_agent.py
# ─────────────────────────────────────────────────────────────
# KairosAI Flask API — AI Agent Routes
# Endpoints:
#   POST /agent/chat    — send message to agent
#   POST /agent/reset   — clear conversation memory
#   GET  /agent/status  — agent health check
# ─────────────────────────────────────────────────────────────

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity

agent_bp = Blueprint("agent", __name__, url_prefix="/agent")

# Store agent executors per user session
# In production this would use Redis — for now in-memory is fine
_agents = {}


def get_or_create_agent(user_id: str):
    # Returns existing agent for user or creates a new one
    if user_id not in _agents:
        from agent.agent import create_agent
        _agents[user_id] = create_agent()
    return _agents[user_id]


# ── CHAT ──────────────────────────────────────────────────────
@agent_bp.route("/chat", methods=["POST"])
@jwt_required()
def chat():
    # Sends a message to the AI agent
    # Expects JSON: { "message": "What is NVDA sentiment?" }
    # Returns agent response with tool trace
    user_id = get_jwt_identity()
    data    = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "message is required"}), 400

    try:
        executor = get_or_create_agent(user_id)

        # Run agent
        result   = executor.invoke({"input": message})
        response = result.get("output", "No response generated.")

        # Extract tool trace from intermediate steps
        steps = result.get("intermediate_steps", [])
        trace = []
        for action, observation in steps:
            trace.append({
                "tool":        action.tool,
                "input":       str(action.tool_input)[:200],
                "observation": str(observation)[:300]
            })

        return jsonify({
            "response": response,
            "trace":    trace,
            "steps":    len(trace)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── RESET MEMORY ──────────────────────────────────────────────
@agent_bp.route("/reset", methods=["POST"])
@jwt_required()
def reset():
    # Clears conversation memory for this user
    user_id = get_jwt_identity()

    if user_id in _agents:
        _agents[user_id].memory.clear()

    return jsonify({"message": "Conversation memory cleared"}), 200


# ── STATUS ────────────────────────────────────────────────────
@agent_bp.route("/status", methods=["GET"])
@jwt_required()
def status():
    # Returns agent health status
    return jsonify({
        "status":       "online",
        "model":        "llama-3.3-70b-versatile",
        "provider":     "groq",
        "tools":        ["sql_query", "vector_search", "live_price"],
        "active_users": len(_agents)
    }), 200