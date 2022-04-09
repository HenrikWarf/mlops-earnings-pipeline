
# Specifies base image and tag
FROM gcr.io/google.com/cloudsdktool/cloud-sdk:latest
WORKDIR /root

# Copies the trainer code to the docker image.
COPY cloud-run-execution/main.py /root/main.py
COPY cloud-run-execution/requirements.txt /root/requirements.txt
COPY pipeline-code/task.py /root/task.py

# Installs additional packages
RUN pip3 install -r requirements.txt 

#Execute the Application
#ENTRYPOINT ["python3", "pipeline-run.py"]
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
