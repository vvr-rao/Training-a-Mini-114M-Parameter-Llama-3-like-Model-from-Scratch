import boto3
import os

aws_access_key = "<AWS ACCESS KEY>"
aws_secret_key = "<AWS SECRET KEY>"

s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)




def upload_file_to_s3(file_path, bucket_name, destination):
  try:
      s3.upload_file(file_path, bucket_name, destination)
      print(f"File {file_path} successfully uploaded to S3 bucket {bucket_name}.")
  except Exception as e:
      print(f"Error uploading file to S3: {e}")


def download_file_from_s3(bucket_name, file_key, local_path):
  # Download the file from S3
  try:
      s3.download_file(bucket_name, file_key, local_path)
      print(f"File {file_key} downloaded successfully to {local_path}")
  except Exception as e:
      print(f"Error downloading file: {e}")


def download_dataset(bucket_name, folder_name, local_dir):
  bucket_and_folder_name = bucket_name + "/" + folder_name
  response = s3.list_objects_v2(Bucket=bucket_name)
  objects = response.get('Contents', [])
  for obj in objects:
    filename = os.path.join(obj['Key'])
    s3.download_file(bucket_name, obj['Key'], filename)
    print(f"Downloaded {obj['Key']} to {filename}")

