terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Storage bucket (optional for CSVs)
resource "google_storage_bucket" "data" {
  name          = var.bucket_name
  location      = var.region
  force_destroy = true
}

# BigQuery dataset
resource "google_bigquery_dataset" "football" {
  dataset_id                 = "football"
  location                   = "US"
  delete_contents_on_destroy = true
}