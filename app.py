from flask import Flask, request, jsonify, render_template
from agent import IPCCLLMAgent  # Import the real class from agent.py
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize Flask app and agent
app = Flask(__name__)
agent = IPCCLLMAgent()


@app.route("/process_message", methods=["POST"])
def process_message():
    try:
        data = request.get_json()
        message = data.get("message", "")
        history = data.get("history", [])
        model = data.get("model", "mock")
        report_focus = data.get("report_focus", "all")

        # Call the agent's method
        history, _ = agent.process_message(message, history, model, report_focus)
        return jsonify({"history": history})

    except Exception as e:
        return jsonify({"error": f"⚠️ Error processing message: {str(e)}"}), 500


@app.route("/new_session", methods=["POST"])
def new_session():
    history = agent.new_session()
    return jsonify({"history": history, "session_id": agent.session_id})


@app.route("/switch_session", methods=["POST"])
def switch_session():
    data = request.get_json()
    session_id = data.get("session_id", "")
    history = agent.switch_session(session_id)
    return jsonify({"history": history, "session_id": session_id})


@app.route("/get_sessions", methods=["GET"])
def get_sessions():
    return jsonify({"sessions": agent.get_session_list()})


@app.route("/chat_ui.html")
def serve_ui():
    return render_template("chat_ui.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
