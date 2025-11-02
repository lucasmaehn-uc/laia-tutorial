import os

import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "user1")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "UserPass123!")

# Configure MLflow to use remote server with artifact uploads
mlflow.set_tracking_uri("http://dsn2026hotcrp.dei.uc.pt:8080")

# The MLflow server is configured with --serve-artifacts, so artifacts should be
# uploaded via HTTP. We use mlflow-artifacts:/ URI for this.
experiment_name = "user-jrc-iris_classification"
artifact_uri = "mlflow-artifacts:/"

# Check if experiment exists and has correct artifact location
existing_experiment = mlflow.get_experiment_by_name(experiment_name)

# if existing_experiment:
#     # Check if the existing experiment has a problematic artifact location
#     if existing_experiment.artifact_location.startswith("/mlflow") or \
#        existing_experiment.artifact_location.startswith("file:///mlflow"):
#         print(f"Existing experiment has Docker path artifact location: {existing_experiment.artifact_location}")
#         print("Creating new experiment with remote artifact storage...")
#         # Use a new experiment name for remote artifact storage
#         experiment_name = "iris_classification_remote"
#         try:
#             experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
#             print(f"Created new experiment '{experiment_name}'")
#         except Exception:
#             pass
#     else:
#         print(f"Using existing experiment '{experiment_name}'")
# else:
#     # Create new experiment with proper artifact location
#     try:
#         experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
#         print(f"Created new experiment '{experiment_name}' with remote artifact storage")
#     except Exception as e:
#         print(f"Note: {e}")

mlflow.set_experiment(experiment_name)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

best_acc = 0
best_run_id = None
best_C = None

for C in [0.1, 1.0, 10.0]:
    with mlflow.start_run() as run:
        model = LogisticRegression(max_iter=200, C=C)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", acc)

        fig, ax = plt.subplots()
        ax.scatter(y_test, preds)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"C={C}")
        mlflow.log_figure(fig, f"results_C{C}.png")
        plt.close(fig)

        # Infer model signature
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        
        mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=X_train[:5])
        
        # Track the best model
        if acc > best_acc:
            best_acc = acc
            best_run_id = run.info.run_id
            best_C = C

# Register the best model
print(f"\nBest model: C={best_C}, accuracy={best_acc:.4f}")
model_uri = f"runs:/{best_run_id}/model"
registered_model = mlflow.register_model(model_uri, "iris-jrc")
print(f"Registered model version: {registered_model.version}")