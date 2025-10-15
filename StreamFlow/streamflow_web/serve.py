#!/usr/bin/env python3
"""
StreamFlow Web Interface Server (Flask version)
-----------------------------------------------
Serve your StreamFlow documentation and static web files with style.
"""

import webbrowser
import threading
import time
from pathlib import Path
from flask import Flask, send_from_directory, render_template_string, abort
import markdown

# === Configuration ===
PORT = 8080
WEB_DIR = Path(__file__).parent
INDEX_FILE = WEB_DIR / "index.html"
README_FILE = WEB_DIR / "README.md"

# === Flask App ===
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")

# === Routes ===

@app.route("/")
def index():
    """Serve the main index.html file."""
    if INDEX_FILE.exists():
        return send_from_directory(WEB_DIR, "index.html")
    return (
        "<h1 style='font-family:sans-serif;color:red;'>❌ index.html not found!</h1>",
        404,
    )

@app.route("/<path:path>")
def static_files(path):
    """Serve static files (CSS, JS, images, etc.)."""
    file_path = WEB_DIR / path
    if file_path.exists():
        return send_from_directory(WEB_DIR, path)
    abort(404)


@app.route("/readme")
def readme():
    """Render README.md"""
    if not README_FILE.exists():
        return (
            "<h1 style='font-family:sans-serif;color:red;'>❌ README.md not found!</h1>",
            404,
        )

    # Convert Markdown -> HTML
    with open(README_FILE, "r", encoding="utf-8") as f:
        md_content = f.read()
    html_content = markdown.markdown(md_content, extensions=["fenced_code", "tables"])

    return render_template_string(html_content)

# === Utility Functions ===

def open_browser():
    """Open the browser automatically after a short delay."""
    time.sleep(1)
    webbrowser.open(f"http://localhost:{PORT}")


# === Main Entry Point ===
if __name__ == "__main__":
    print("🌟 StreamFlow Web Documentation Server")
    print("=" * 50)
    print(f"📁 Serving from: {WEB_DIR}")
    print(f"📱 Open your browser at: http://localhost:{PORT}")
    print("⏹️  Press Ctrl+C to stop the server")

    if not INDEX_FILE.exists():
        print(
            f"⚠️ Warning: {INDEX_FILE.name} not found — homepage may not load correctly."
        )

    # Open browser in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Run the Flask server
    app.run(port=PORT, debug=False)
