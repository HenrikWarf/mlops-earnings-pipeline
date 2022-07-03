from datetime import datetime
import os
import sys
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip

from flask import Flask

app = Flask(__name__)

@app.route("/", methods=["POST"])
def pipeline_run(): 

    REGION = "us-central1"
    PROJECT_ID = "mlops-dev-999-c6b8"
    DISPLAY_NAME = "earnings-classifier-pipeline"
    COMPILED_PIPELINE_PATH = "earnings_pipeline.json"
    JOB_ID = ""
    PIPELINE_PARAMETERS = {}
    ENABLE_CACHING = False
    SERVICE_ACCOUNT = 'ml1-dev-sa@mlops-dev-999-c6b8.iam.gserviceaccount.com'
    PIPELINE_ROOT_PATH = 'gs://crazy-pipelines/earnings_classifier/crazy-hippo'
    


    from google.cloud import aiplatform

    job = aiplatform.PipelineJob(display_name = DISPLAY_NAME,
                                 template_path = COMPILED_PIPELINE_PATH,
                                 job_id = JOB_ID,
                                 pipeline_root = PIPELINE_ROOT_PATH,
                                 parameter_values = PIPELINE_PARAMETERS,
                                 enable_caching = ENABLE_CACHING,
                                 project = PROJECT_ID,
                                 location = REGION)

    job.submit(
        service_account=SERVICE_ACCOUNT
    )


    print("Pipeline is Executing!")
    
    return("Done!")

if __name__ == "__main__":
    PORT = int(os.getenv("PORT")) if os.getenv("PORT") else 8080

    # This is used when running locally. Gunicorn is used to run the
    # application on Cloud Run. See entrypoint in Dockerfile.
    app.run(host="127.0.0.1", port=PORT, debug=True)
