from flask import Flask, request, jsonify, send_file
import os
import datetime
import pickle
from filelock import FileLock
from io import BytesIO
import logging
import json
from dotenv import load_dotenv
from pymongo import MongoClient
import uuid

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Standard console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

app = Flask(__name__)

# MongoDB connection setup
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    logger.warning("MongoDB URI not found in environment variables. Logging to MongoDB is disabled.")
    mongo_client = None
    db = None
    log_collection = None
else:
    try:
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client.model_aggregation
        log_collection = db.logs
        logger.info("Successfully connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        mongo_client = None
        db = None
        log_collection = None

UPLOAD_FOLDER = 'models'
LOCK_PATH = 'aggregate.lock'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def log_to_mongo(event_type, details, status="info"):
    """Log events to MongoDB"""
    if log_collection is not None:
        try:
            log_entry = {
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.utcnow(),
                "event_type": event_type,
                "status": status,
                "details": details
            }
            log_collection.insert_one(log_entry)
        except Exception as e:
            logger.error(f"Failed to log to MongoDB: {str(e)}")

@app.route('/upload_model', methods=['POST'])
def upload_model():
    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id}: Received model upload request")
    
    try:
        if 'model_file' not in request.files:
            error_msg = "No model file part"
            logger.warning(f"Request {request_id}: {error_msg}")
            log_to_mongo("model_upload", {"request_id": request_id, "error": error_msg}, "warning")
            return jsonify({'error': error_msg}), 400
        
        file = request.files['model_file']
        if file.filename == '':
            error_msg = "No selected file"
            logger.warning(f"Request {request_id}: {error_msg}")
            log_to_mongo("model_upload", {"request_id": request_id, "error": error_msg}, "warning")
            return jsonify({'error': error_msg}), 400
        
        device_id = request.form.get('device_id', 'unknown_device')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{device_id}_{timestamp}.pkl"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        file.save(filepath)
        logger.info(f"Request {request_id}: Saved model file from device {device_id} as {filename}")
        
        # Optional: Validate model structure
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                if not all(k in model_data for k in ['n', 'vocabulary', 'total_words']):
                    os.remove(filepath)
                    error_msg = "Invalid model structure"
                    logger.error(f"Request {request_id}: {error_msg}")
                    log_to_mongo("model_validation", {
                        "request_id": request_id,
                        "device_id": device_id,
                        "filename": filename,
                        "error": error_msg
                    }, "error")
                    return jsonify({'error': error_msg}), 400
                
                # Log model statistics
                stats = {
                    "n": model_data['n'],
                    "vocab_size": len(model_data['vocabulary']),
                    "total_words": model_data['total_words']
                }
                logger.info(f"Request {request_id}: Model statistics - {json.dumps(stats)}")
                
        except Exception as e:
            os.remove(filepath)
            error_msg = f"Model validation failed: {str(e)}"
            logger.error(f"Request {request_id}: {error_msg}")
            log_to_mongo("model_validation", {
                "request_id": request_id,
                "device_id": device_id,
                "filename": filename,
                "error": error_msg
            }, "error")
            return jsonify({'error': error_msg}), 400
        
        log_to_mongo("model_upload", {
            "request_id": request_id,
            "device_id": device_id,
            "filename": filename,
            "file_size": os.path.getsize(filepath),
            "stats": stats
        }, "info")
        
        return jsonify({
            'success': True,
            'message': 'Model uploaded successfully',
            'filename': filename
        })
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Request {request_id}: Upload failed - {error_msg}")
        log_to_mongo("model_upload", {"request_id": request_id, "error": error_msg}, "error")
        return jsonify({'error': error_msg}), 500

@app.route('/download_aggregated_model', methods=['GET'])
def download_aggregated_model():
    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id}: Received aggregated model download request")
    
    lock = FileLock(LOCK_PATH)
    try:
        with lock:
            model_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.pkl')]
            if not model_files:
                error_msg = "No uploaded models to aggregate"
                logger.warning(f"Request {request_id}: {error_msg}")
                log_to_mongo("model_download", {"request_id": request_id, "error": error_msg}, "warning")
                return jsonify({'error': error_msg}), 400
            
            logger.info(f"Request {request_id}: Aggregating {len(model_files)} models")
            
            aggregated = None
            count = 0
            processed_files = []
            
            for fname in model_files:
                try:
                    with open(os.path.join(UPLOAD_FOLDER, fname), 'rb') as f:
                        model = pickle.load(f)
                    
                    if aggregated is None:
                        aggregated = model
                    else:
                        aggregated['n'] += model['n']
                        aggregated['total_words'] += model['total_words']
                        for word in model['vocabulary']:
                            aggregated['vocabulary'][word] = aggregated['vocabulary'].get(word, 0) + 1
                    
                    count += 1
                    processed_files.append(fname)
                    logger.debug(f"Request {request_id}: Processed model file {fname}")
                    
                except Exception as e:
                    logger.error(f"Request {request_id}: Failed to process model file {fname}: {str(e)}")
            
            # Log aggregation results
            agg_stats = {
                "processed_files": count,
                "total_processed": count,
                "n": aggregated['n'] if aggregated else 0,
                "vocab_size": len(aggregated['vocabulary']) if aggregated else 0,
                "total_words": aggregated['total_words'] if aggregated else 0
            }
            logger.info(f"Request {request_id}: Aggregation stats - {json.dumps(agg_stats)}")
            
            # Send aggregated model as a file-like object
            buffer = BytesIO()
            pickle.dump(aggregated, buffer)
            buffer.seek(0)
            
            log_to_mongo("model_aggregation", {
                "request_id": request_id,
                "num_files": len(model_files),
                "processed_files": count,
                "stats": agg_stats
            }, "info")
            
            return send_file(
                buffer,
                as_attachment=True,
                download_name='aggregated_model.pkl',
                mimetype='application/octet-stream'
            )
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Request {request_id}: Download failed - {error_msg}")
        log_to_mongo("model_download", {"request_id": request_id, "error": error_msg}, "error")
        return jsonify({'error': error_msg}), 500

@app.route('/health', methods=['GET'])
def health_check():
    health_status = {
        'status': 'ok',
    }
    
    log_to_mongo("health_check", health_status, "info")
    logger.info(f"Health check: {json.dumps(health_status)}")
    
    return jsonify(health_status)

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(host='0.0.0.0', port=5000, debug=True)