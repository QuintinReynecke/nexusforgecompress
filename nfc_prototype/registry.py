import os
import sys
from flask import Flask, request, send_file, abort
from nfc_prototype.core import NFCStore

app = Flask(__name__)
# The server's local store
store = NFCStore(store_path="./registry_store")

@app.route('/<shard>/<hash_hex>', methods=['GET'])
def get_blob(shard, hash_hex):
    """Retrieve a block."""
    data = store.get(hash_hex)
    if data:
        # We need a temp file to send or we can use a BytesIO
        from io import BytesIO
        return send_file(BytesIO(data), mimetype='application/octet-stream')
    abort(404)

@app.route('/upload/<hash_hex>', methods=['POST'])
def upload_blob(hash_hex):
    """Store a new block."""
    if store.put(request.data, bytes.fromhex(hash_hex)):
        return "Stored", 200
    return "Already exists", 200

def run_server(port=5000):
    print(f"Starting NFC Neural Registry on port {port}...")
    app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_RUN_PORT", os.environ.get("PORT", 5000)))
    run_server(port=port)
