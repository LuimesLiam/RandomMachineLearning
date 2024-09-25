import requests
import tarfile
import pathlib

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

dataset_tar = "flower_photos.tar.gz"  

print("Downloading dataset...")
response = requests.get(dataset_url, stream=True)
if response.status_code == 200:
    with open(dataset_tar, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("Download complete.")
else:
    print(f"Failed to download dataset. Status code: {response.status_code}")

data_dir = pathlib.Path("flower_photos")  

print("Extracting dataset...")
with tarfile.open(dataset_tar, "r:gz") as tar:
    tar.extractall(path=data_dir.parent)
print("Extraction complete.")

print(f"Dataset extracted to: {data_dir}")
