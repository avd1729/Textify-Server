from flask import Flask, request, jsonify
import os
import datetime
import pickle

app = Flask(__name__)

# Directory to store uploaded models
UPLOAD_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    try:
        # Check if the post request has the file part
        if 'model_file' not in request.files:
            return jsonify({'error': 'No model file part'}), 400
        
        file = request.files['model_file']
        
        # If user does not select file, browser also submits empty part
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Get the device ID
        device_id = request.form.get('device_id', 'unknown_device')
        
        # Create a unique filename with timestamp and device ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{device_id}_{timestamp}.pkl"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save the file
        file.save(filepath)
        
        # Optionally validate the model file
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                # Check basic structure to validate it's our expected model format
                if not ('n' in model_data and 'vocabulary' in model_data):
                    return jsonify({'error': 'Invalid model format'}), 400
        except Exception as e:
            # If validation fails, delete the file
            os.remove(filepath)
            return jsonify({'error': f'Model validation failed: {str(e)}'}), 400
        
        # Process the model (e.g., aggregate it with others, analyze it, etc.)
        # This would be implemented based on your specific needs
        
        return jsonify({
            'success': True,
            'message': 'Model uploaded successfully',
            'filename': filename,
            'model_stats': {
                'vocab_size': len(model_data['vocabulary']),
                'total_words': model_data['total_words']
            }
        })
    
    except Exception as e:
        app.logger.error(f"Error in upload_model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)