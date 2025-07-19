# codec/s3utils.py
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
    def __init__(self, endpoint_url: str, access_key: str, secret_key: str, bucket_name: str, public_url_base: str):
        """
        Initializes the S3 client.

        Args:
            endpoint_url: The endpoint URL for the S3 SDK (e.g., 'https://sfo3.digitaloceanspaces.com').
            access_key: The access key ID for the service.
            secret_key: The secret access key for the service.
            bucket_name: The name of the bucket (or space) to interact with.
            public_url_base: The full public base URL for accessing files, typically the CDN endpoint
                             (e.g., 'https://my-bucket.sfo3.cdn.digitaloceanspaces.com').
        """
        self.bucket_name = bucket_name
        self.client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(s3={'addressing_style': 'virtual'})
        )
        # Use the explicit public URL base provided from the configuration.
        # This is the correct, public-facing URL for services like OpenAI to access.
        self.public_url_base = public_url_base

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
            # accessible via its public URL, which is required by the OpenAI model.
            self.client.upload_file(
                file_path,
                self.bucket_name,
                object_name,
                ExtraArgs={'ACL': 'public-read'}
            )
            # Construct the final URL using the correct public base.
            public_url = f"{self.public_url_base}/{object_name}"
            print(f"Successfully uploaded {Path(file_path).name} to S3. URL: {public_url}")
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