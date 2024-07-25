from flask import Flask, request, jsonify
from main import colorize_and_add_audio
from util import load_video_from_json, video_to_base64
import concurrent.futures
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/colorizevideo', methods=['POST'])
def colorize():
    data = request.get_json()
    logging.info("Received data: %s", data)
    
    if not data:
        return jsonify({"error": "Invalid input, no JSON data provided"}), 400

    if 'video' not in data:
        return jsonify({"error": "No video provided"}), 400

    try:
        video_path, video_format = load_video_from_json(data['video'])
    except Exception as e:
        logging.error("Error loading video from JSON: %s", str(e))
        return jsonify({"error": str(e)}), 500

    output_path_siggraph17 = 'output_siggraph17.mp4'
    output_path_eccv16 = 'output_eccv16.mp4'

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(colorize_and_add_audio, video_path, output_path_siggraph17, 'siggraph17'),
                executor.submit(colorize_and_add_audio, video_path, output_path_eccv16, 'eccv16')
            ]
            concurrent.futures.wait(futures)
    except Exception as e:
        logging.error("Error processing video: %s", str(e))
        return jsonify({"error": str(e)}), 500

    try:
        original_video_base64 = video_to_base64(video_path, video_format)
        colorized_video_siggraph17_base64 = video_to_base64(output_path_siggraph17, video_format)
        colorized_video_eccv16_base64 = video_to_base64(output_path_eccv16, video_format)
    except Exception as e:
        logging.error("Error converting video to base64: %s", str(e))
        return jsonify({"error": str(e)}), 500

    return jsonify({
        'original_video': original_video_base64,
        'colorized_video_siggraph17': colorized_video_siggraph17_base64,
        'colorized_video_eccv16': colorized_video_eccv16_base64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
