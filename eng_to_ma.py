from flask import Flask, request, jsonify
from flask_cors import CORS
from googletrans import Translator

app = Flask(__name__)
CORS(app)  # This allows your HTML file to make requests to the server

def translate_to_multiple_languages(text):
    """
    Translates a given string from English to Marathi and Hindi.
    """
    try:
        translator = Translator()
        # Translate to Marathi (dest='mr')
        marathi_translation = translator.translate(text, dest='mr').text
        # Translate to Hindi (dest='hi')
        hindi_translation = translator.translate(text, dest='hi').text
        return {
            "marathi": marathi_translation,
            "hindi": hindi_translation
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

@app.route('/translate', methods=['POST'])
def translate_text():
    """
    Handles translation requests and returns the translated text in Marathi and Hindi.
    """
    data = request.json
    english_text = data.get('text')
    
    if not english_text:
        return jsonify({"error": "No text provided"}), 400
        
    translations = translate_to_multiple_languages(english_text)
    
    if translations:
        return jsonify({
            "english": english_text,
            "translations": translations
        })
    else:
        return jsonify({"error": "Translation failed"}), 500

if __name__ == '__main__':
    # Make sure to install Flask and googletrans before running
    # pip install Flask flask_cors googletrans==4.0.0rc1
    app.run(debug=True)
