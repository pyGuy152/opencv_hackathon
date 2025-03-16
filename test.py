import requests

# The URL of the Flask server where the file will be uploaded
url = 'https://ready-danya-hackathon-chi-8c014eb4.koyeb.app/api'

# Path to the file you want to upload
file_path = 'premium_photo-1674170065323-9f207919ea27.jpeg'

# Open the file in binary mode and send it in the POST request
with open(file_path, 'rb') as file:
    files = {'file': (file.name, file)}
    response = requests.post(url, files=files)

# Print the response from the server
print(response.text)
