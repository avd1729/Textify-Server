from flask import Flask, request, jsonify, send_file
import os
import datetime
import pickle
from filelock import FileLock
from io import BytesIO

app = Flask(__name__)

UPLOAD_FOLDER = 'models'
LOCK_PATH = 'aggregate.lock'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    try:
        if 'model_file' not in request.files:
            return jsonify({'error': 'No model file part'}), 400

        file = request.files['model_file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        device_id = request.form.get('device_id', 'unknown_device')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{device_id}_{timestamp}.pkl"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        # Optional: Validate model structure
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                if not all(k in model_data for k in ['n', 'vocabulary', 'total_words']):
                    os.remove(filepath)
                    return jsonify({'error': 'Invalid model structure'}), 400
        except Exception as e:
            os.remove(filepath)
            return jsonify({'error': f'Model validation failed: {str(e)}'}), 400

        return jsonify({
            'success': True,
            'message': 'Model uploaded successfully',
            'filename': filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download_aggregated_model', methods=['GET'])
def download_aggregated_model():
    lock = FileLock(LOCK_PATH)
    try:
        with lock:
            model_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.pkl')]
            if not model_files:
                return jsonify({'error': 'No uploaded models to aggregate'}), 400

            aggregated = None
            count = 0

            for fname in model_files:
                with open(os.path.join(UPLOAD_FOLDER, fname), 'rb') as f:
                    model = pickle.load(f)

                if aggregated is None:
                    aggregated = model
                else:
                    aggregated['n'] += model['n']
                    aggregated['total_words'] += model['total_words']
                    for word, val in model['vocabulary'].items():
                        aggregated['vocabulary'][word] = aggregated['vocabulary'].get(word, 0) + val

                count += 1

            # Send aggregated model as a file-like object
            buffer = BytesIO()
            pickle.dump(aggregated, buffer)
            buffer.seek(0)

            return send_file(
                buffer,
                as_attachment=True,
                download_name='aggregated_model.pkl',
                mimetype='application/octet-stream'
            )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
