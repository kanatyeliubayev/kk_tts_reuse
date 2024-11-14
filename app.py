from flask import Flask, request, jsonify, send_file
import subprocess
import os

app = Flask(__name__)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    text = request.form.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    output_file = "output.wav"
    try:
        # Run synthesize.py with the provided text
        subprocess.run(["python3", "synthesize.py", "--text", text, "--output", output_file], check=True)
        
        # Serve the generated audio file
        return send_file(output_file, mimetype='audio/wav')
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the output file after sending
        if os.path.exists(output_file):
            os.remove(output_file)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
