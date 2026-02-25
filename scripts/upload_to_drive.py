import os
import json
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

def upload_to_drive():
    creds_json = os.environ.get('GDRIVE_CREDENTIALS')
    if not creds_json:
        print("Error: GDRIVE_CREDENTIALS environment variable not set.")
        return

    creds_file = 'client_secrets.json'
    with open(creds_file, 'w', encoding='utf-8') as f:
        f.write(creds_json)

    gauth = GoogleAuth()
    creds_dict = json.loads(creds_json)
    gauth.settings['client_config_backend'] = 'service'
    gauth.settings['service_config'] = {
        'client_json_dict': creds_dict,
        'client_user_email': creds_dict.get('client_email', '')
    }
    
    try:
        gauth.ServiceAuth()
    except Exception as e:
        print(f"Failed to authenticate with Google Drive: {e}")
        return

    drive = GoogleDrive(gauth)
    target_dir = 'mlruns/0'
    files_uploaded = 0
    
    if os.path.exists(target_dir):
        for run_id in os.listdir(target_dir):
            run_path = os.path.join(target_dir, run_id, 'artifacts', 'model', 'conda.yaml')
            if os.path.exists(run_path):
                print(f"Uploading {run_path} to Google Drive...")
                file_drive = drive.CreateFile({'title': f'model_conda_{run_id}.yaml'})
                file_drive.SetContentFile(run_path)
                file_drive.Upload()
                print(f"File uploaded successfully! ID: {file_drive['id']}")
                files_uploaded += 1
                
    if files_uploaded == 0:
        print("No models found in mlruns/0. Uploading the MLProject file instead for verification.")
        if os.path.exists("MLProject"):
            file_drive = drive.CreateFile({'title': 'MLProject_File'})
            file_drive.SetContentFile("MLProject")
            try:
                file_drive.Upload()
                print("MLProject file uploaded successfully to Google Drive.")
            except Exception as e:
                print(f"Failed to upload: {e}")
        else:
            print("Nothing to upload.")

if __name__ == '__main__':
    upload_to_drive()
