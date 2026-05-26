# app.py

from flask import Flask, jsonify
from asgiref.wsgi import WsgiToAsgi

# -----------------------------
# Step 1: Create Flask App
# -----------------------------

flask_app = Flask(__name__)

# -----------------------------
# Step 2: Routes
# -----------------------------

@flask_app.route("/")
def home():

    return jsonify({
        "message": "Flask app running with Uvicorn"
    })

# -----------------------------
# Step 3: Convert WSGI -> ASGI
# -----------------------------

app = WsgiToAsgi(flask_app)