# codec/s3_utils.py
import os
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from pathlib import Path

class S3Uploader:
    """
    A utility class to handle file uploads and deletions for an S3-compatible
    object storage service, such as DigitalOcean Spaces.
    """
    def __init__(self, endpoint_url: str, access_key: str, secret_key: str, bucket_name: str):
        """
        Initializes the S3 client.

        Args:
            endpoint_url: The endpoint URL of the S3 service (e.g., 'https://nyc3.digitaloceanspaces.com').
            access_key: The access key ID for the service.
            secret_key: The secret access key for the service.
            bucket_name: The name of the bucket (or space) to interact with.
        """
        self.bucket_name = bucket_name
        self.client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(s3={'addressing_style': 'virtual'})
        )
        # Construct the base part of the public URL for files.
        # For DO Spaces, this transforms 'https://nyc3.digitaloceanspaces.com'
        # into 'https://your-bucket-name.nyc3.digitaloceanspaces.com'
        self.public_url_base = f"{endpoint_url.replace('digitaloceanspaces.com', f'{bucket_name}.digitaloceanspaces.com')}"

    def upload(self, file_path: str, object_name: str) -> str:
        """
        Uploads a file to the S3 bucket and makes it publicly readable.

        Args:
            file_path: The local path of the file to upload.
            object_name: The desired name for the object in the bucket (including any 'folders').

        Returns:
            The full public URL of the uploaded file.

        Raises:
            ClientError: If the upload fails.
        """
        try:
            # The 'ACL': 'public-read' argument is crucial for making the file
            # accessible via its public URL, which OpenAI's model needs.
            self.client.upload_file(
                file_path,
                self.bucket_name,
                object_name,
                ExtraArgs={'ACL': 'public-read'}
            )
            public_url = f"{self.public_url_base}/{object_name}"
            print(f"Successfully uploaded {Path(file_path).name} to S3 as {object_name}")
            return public_url
        except ClientError as e:
            print(f"Error uploading to S3: {e}")
            raise

    def delete(self, object_name: str):
        """
        Deletes an object from the S3 bucket.

        Args:
            object_name: The name of the object to delete from the bucket.
        """
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=object_name)
            print(f"  - Deleted S3 object: {object_name}")
        except ClientError as e:
            # Fail gracefully so that cleanup can continue with other files.
            print(f"  - Failed to delete S3 object {object_name}: {e}")