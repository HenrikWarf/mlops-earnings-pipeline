from datetime import datetime
import os
import sys
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip

from flask import Flask

app = Flask(__name__)

@app.route("/", methods=["POST"])
def pipeline_run(): 

    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
    MODEL_VERSION = "v12"
    LOCATION = "us-west1"
    MODEL_FILENAME = 'earnings_model_{}.pkl'.format(MODEL_VERSION)
    JOB_NAME = MODEL_FILENAME + TIMESTAMP
    MACHINE_TYPE = "n1-standard-4"
    REPLICA_COUNT = 1
    ACCELERATOR_COUNT = 0
    PROJECT_ID = "crazy-hippo-01"
    BUCKET_NAME = "gs://custom-earnings-model/" 
    SERVICE_ACCOUNT = 'pipelines-vertex-ai@crazy-hippo-01.iam.gserviceaccount.com'
    BASE_IMAGE_URI = "europe-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest"
    MODEL_DISPLAY_NAME = MODEL_FILENAME + TIMESTAMP
    SCRIPT_PATH = "task.py"


    aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_NAME)

    job = aiplatform.CustomTrainingJob(
        display_name=JOB_NAME,
        script_path=SCRIPT_PATH,
        container_uri=BASE_IMAGE_URI,
        requirements=["pyarrow", 
                      "pandas", 
                      "numpy", 
                      "sklearn", 
                      "google.cloud", 
                      "google-cloud-bigquery", 
                      "google-cloud-storage"],
        model_serving_container_image_uri=BASE_IMAGE_URI,
    )

    # Start the training

    model = job.run(
        model_display_name=MODEL_DISPLAY_NAME,
    #    args=CMDARGS,
        replica_count=REPLICA_COUNT,
        machine_type=MACHINE_TYPE,
        service_account=SERVICE_ACCOUNT,
        accelerator_count=ACCELERATOR_COUNT,
        )


    Print("EarningsPred. Pipeline Job Done!")

if __name__ == "__main__":
    PORT = int(os.getenv("PORT")) if os.getenv("PORT") else 8080

    # This is used when running locally. Gunicorn is used to run the
    # application on Cloud Run. See entrypoint in Dockerfile.
    app.run(host="127.0.0.1", port=PORT, debug=True)
