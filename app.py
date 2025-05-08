from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)

# RAM sorunu yaşamamak için hafif model kullanıyoruz
generator = pipeline('text-generation', model='distilgpt2')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        result = generator(prompt, max_length=128, do_sample=True, temperature=0.7)
        return jsonify({"output": result[0]['generated_text']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Railway otomatik PORT atadığı için ona uyum sağlıyoruz
port = int(os.environ.get("PORT", 5000))
app.run(host='0.0.0.0', port=port)
