# Vertex AI - Creating Custom Pipeline Executions

# Libraries and Framework Imports
import pandas as pd
import numpy as np
import pickle
import os

# scikit-learn modules for training, data split, preprocessing (standardization + lab and model evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from google.cloud import bigquery
from google.cloud import storage

import pyarrow

# Set some initial project and ML variables
MODEL_VERSION = "v9"
PROJECT_ID = "crazy-hippo-01"
BUCKET = "custom-earnings-model"

model_filename = 'earnings_model_{}.pkl'.format(MODEL_VERSION)

# Upload model artifact to Cloud Storage
model_directory = str(os.getenv("AIP_MODEL_DIR"))
print(model_directory)

if model_directory == "None" :
    local_run = True
    model_directory = "gs://{}/{}".format(BUCKET, MODEL_VERSION)
    storage_path = os.path.join(model_directory, model_filename)
    print(storage_path)
else: 
    local_run = False
    storage_path = os.path.join(model_directory, model_filename)
    print(storage_path)

query = """
        SELECT *
        FROM `clv.earnings_per_year`
    """

query_two = """
        SELECT *
        FROM `clv.new_data`
    """  


# Load data into container from BigQuery using the Client Library
def ingestion_from_bq(PROJECT_ID, query):
    
    client = bigquery.Client(location="US", 
                             project=PROJECT_ID
                            )
    
    print("Client creating using project: {}".format(client.project))

    # Run ingestion query against the raw dataset table and store data in pandas dataframe
    query = query

    query_job = client.query(
        query,
        location="US",
    ) 

    df = query_job.to_dataframe()
    
    return(df)


#Organize Features into numeric and categorical
def exctract_data_types(dataframe) :
    cat = []
    num = []
    undef = []
    
    for category in dataframe.columns:
        if df[category].dtype == object:
            cat.append(category)
        elif df[category].dtype == int:
            num.append(category)
        elif df[category].dtype == float:
            num.append(category)
        else : 
            undef.append(category)
    
    return cat, num, undef

#Data Preprocessing
def preprocessing_function(cat_columns, num_columns):
    # Scaling of Numeric Features
    numeric_features = num_columns
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), 
               ("scaler", MinMaxScaler()),
              # ("scaler", StandardScaler()())
              ]
    )

    # Scaling of Categorical Features
    categorical_features = cat_columns
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Use ColumnTransformer to execute Scaling strategy
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    
    return preprocessor

#Save to Cloud Storage
def save_file(storage_path, PROJECT_ID, model, local_run, model_filename):
        
    if local_run == False:
        """Saves a file to the bucket."""
        # Save model artifact to local filesystem (doesn't persist)
        local_path = model_filename
        with open(local_path, 'wb') as model_file:
              pickle.dump(model, model_file)

        # Upload model artifact to Cloud Storage
        blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
        blob.upload_from_filename(local_path) 
        print("Model saved in {}".format(storage_path))
        
    else: 
         # Save model artifact to local filesystem (doesn't persist)
        local_path = model_filename
        with open(local_path, 'wb') as model_file:
              pickle.dump(model, model_file)

        print("Model saved on local file system with the name: {}".format(local_path))


def load_model(storage_path, PROJECT_ID, model_filename):
    """Downloads a blob from the bucket."""
    
    blob = storage.blob.Blob.from_string(storage_path, 
                                    client=storage.Client(project=PROJECT_ID)
                                   )
    blob.download_to_filename(model_filename)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            model_filename, storage_path, model_filename
        )
    )



# Create Predictions
def run_inference(dataframe, loaded_model, PROJECT_ID):

    loaded_model.predict(new_data)
    predictions = loaded_model.predict(new_data)
    predictions = pd.DataFrame(predictions, columns=['prediction'])


    # Store predictions together with imported new data
    combined_predictions = pd.concat([predictions, new_data], axis=1)

    # ### Load predictions into table in Bigquery
    client = bigquery.Client(location="US", 
                             project=PROJECT_ID, 
                            )
    print("Client creating using project: {}".format(client.project))

    table_id = 'clv.custom_predictions'

    # Since string columns use the "object" dtype, pass in a (partial) schema
    # to ensure the correct BigQuery data type.
    job_config = bigquery.LoadJobConfig(schema=[
        bigquery.SchemaField("prediction", "STRING"),
        bigquery.SchemaField("age", "INT64"),
        bigquery.SchemaField("workclass", "STRING"),
        bigquery.SchemaField("fnlwgt", "INT64"),
        bigquery.SchemaField("education", "STRING"),
        bigquery.SchemaField("education_num", "INT64"),
        bigquery.SchemaField("marital_status", "STRING"),
        bigquery.SchemaField("occupation", "STRING"),
        bigquery.SchemaField("relationship", "STRING"),
        bigquery.SchemaField("race", "STRING"),
        bigquery.SchemaField("sex", "STRING"),
        bigquery.SchemaField("capital_gain", "FLOAT64"),
        bigquery.SchemaField("capital_loss", "FLOAT64"),
        bigquery.SchemaField("hours_per_week", "INT64"),
        bigquery.SchemaField("native_country", "STRING"),
    ])

    job = client.load_table_from_dataframe(
        combined_predictions, table_id, 
        job_config=job_config
    )
    
    print("Predictions load into table {} done!".format(table_id))


# Pipeline Definition
# ------------------------------------------------
# Pipeline Definition

# 1. Inget data from Bigquery
df = ingestion_from_bq(PROJECT_ID, query)

# 2. Copy dataset and seperate Label from Examples
label_ready = df["income"]
df_ready = df.drop(columns=["income"])

# 3. Extract data types from dataframe (categorical and numeric)
cat_columns, num_columns, undefined = exctract_data_types(df_ready)

# 4. Preprocess data 
preprocessor = preprocessing_function(cat_columns, num_columns)

# 5. Build Sklern Pipeline with preprocessor included
clf = Pipeline(
    steps=[("preprocessor", preprocessor), 
           ("classifier", RandomForestClassifier(max_depth=7, random_state=0))]
)

# 6. Split dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(df_ready, label_ready, test_size=0.2, random_state=0)

# 7. Train and fit the model to the data
clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))


# 8. Artifcat/Model Storage
# --------------------------------
# Save Model to Local filesystem
# with open(model_filename, 'wb') as model_file:
#  pickle.dump(clf, model_file)

#(Optional)Save Model to Cloud Storage
save_file(storage_path, PROJECT_ID, clf, local_run, model_filename)
#save_file_two(storage_path, PROJECT_ID, model_filename)

# (Optional)Load Model from Cloud Storage
# load_model(storage_path, PROJECT_ID, model_filename)

# (Optional)Load Model in Container
# loaded_model = pickle.load(open(model_filename, 'rb'))
# ---------------------------------


#9. Ingest new data to run batch prediction on
new_data = ingestion_from_bq(PROJECT_ID, query_two)

#10. Execute inference job and save results to BigQuery Table
run_inference(new_data, clf, PROJECT_ID)

print("Pipeline Job done !")

