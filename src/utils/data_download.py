import requests
import os

# 1. Configuration
dataset_id = "9frgkybxhn"
version = "1"
num_files_to_download = 5  # Change this to the exact number of files you want

# Create a folder to store the downloaded files
output_dir = "mendeley_files"
os.makedirs(output_dir, exist_ok=True)

# 2. Fetch the dataset file list from Mendeley's public API
api_url = f"https://data.mendeley.com/public-api/datasets/{dataset_id}/versions/{version}"
print(f"Fetching dataset metadata...")

# Adding a standard User-Agent so the server doesn't block our script
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(api_url, headers=headers)
response.raise_for_status()

# 3. Parse the data
dataset_data = response.json()
files_list = dataset_data.get("files", [])

if not files_list:
    print("No files found in this dataset.")
else:
    total_files = len(files_list)
    limit = min(num_files_to_download, total_files)
    print(f"Found {total_files} files in total. Downloading the first {limit} files...\n")

    # 4. Download the specified number of files
    for i, file_info in enumerate(files_list[:limit]):
        filename = file_info.get("name", f"file_{i}.bin")

        # Extract the download link from the JSON response
        download_url = file_info.get("content_details", {}).get("download_url")

        # Fallback just in case the JSON structure puts the URL at the top level
        if not download_url:
            download_url = file_info.get("download_url")

        if filename and download_url:
            filepath = os.path.join(output_dir, filename)
            print(f"[{i + 1}/{limit}] Downloading {filename}...")

            # Stream the download (this prevents memory crashes for large files)
            with requests.get(download_url, headers=headers, stream=True) as r:
                r.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        else:
            print(f"[{i + 1}] Skipped: Could not find download URL for this file.")

print("\nAll requested files have been downloaded successfully!")