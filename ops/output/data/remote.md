# Data Remote Configuration

This repository uses **DVC** to manage full-resolution datasets and model artifacts. To keep the repository lightweight, large files are stored in an object store remote. Operators should provision a bucket (for example, `s3://scytonemin-thesis-data`) that exposes versioned storage.

## Recommended Remote
- Provider: Amazon S3 (or any S3-compatible service such as MinIO, Wasabi)
- Bucket: `s3://scytonemin-thesis-data`
- Access: read-only credentials for reviewers, read/write for maintainers

## Setup
```bash
# Activate the local virtual environment first
source .venv/bin/activate

# Configure the default remote (replace with actual bucket URI)
dvc remote add -d thesis-data s3://scytonemin-thesis-data

# Optionally configure credentials via environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

# Push full datasets once populated
dvc push
```

Document any alternative remotes or credential storage decisions in this file so that downstream operators can mirror the data layout.
