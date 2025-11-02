from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import os

# Allow all hosts to connect to Mlflow
os.environ["MLFLOW_ALLOWED_HOSTS"] = "*"

# Configure MLflow tracking URI and authentication
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))

# Set MLflow authentication if credentials are provided
if os.getenv("MLFLOW_TRACKING_USERNAME"):
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
if os.getenv("MLFLOW_TRACKING_PASSWORD"):
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

app = Flask(__name__)

MODEL_NAME = "iris"
MODEL_STAGE = "Production"

# Try to load model once on startup
try:
    app.config["MODEL"] = mlflow.pyfunc.load_model(
        model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    )
    print("Model loaded successfully at startup.")
except Exception as e:
    app.config["MODEL"] = None
    print(f"Could not load model at startup: {e}")
    print("App will start without a model. You can load it later using /reload.")


@app.route("/health", methods=["GET"])
def health():
    """Simple health check."""
    return jsonify(status="healthy", model_loaded=app.config["MODEL"] is not None)


@app.route("/predict", methods=["POST"])
def predict():
    """Make predictions using the model stored in app config."""
    model = app.config["MODEL"]
    if model is None:
        return jsonify(error="Model not loaded. Please call /reload first."), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify(error="No JSON data provided"), 400
        
        if "data" not in data or "columns" not in data:
            return jsonify(error="Missing required fields: 'data' and 'columns'"), 400
        
        df = pd.DataFrame(data["data"], columns=data["columns"])
        preds = model.predict(df)
        return jsonify(predictions=preds.tolist())
    except KeyError as e:
        return jsonify(error=f"Missing required field: {str(e)}"), 400
    except ValueError as e:
        return jsonify(error=f"Invalid data format: {str(e)}"), 400
    except Exception as e:
        return jsonify(error=f"Prediction failed: {str(e)}"), 500


@app.route("/reload", methods=["POST"])
def reload_model():
    """Reload model from MLflow and store in Flask app config."""
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        app.config["MODEL"] = model
        return jsonify(message="Model reloaded successfully.")
    except Exception as e:
        return jsonify(error=f"Failed to load model: {e}"), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)