import os
import gdown
import zipfile
import shutil

def download_data():
    # Ottieni la directory corrente del progetto
    project_dir = os.getcwd()  # Es: 'project/'


    # Percorso ZIP
    zip_path = os.path.join(project_dir, "dataset.zip")


    # Scarica il dataset se non esiste
    if not os.path.exists(zip_path):
        print("📦 Dataset non trovato, avvio download da Google Drive...")

        # Link diretto al file
        url = "https://drive.google.com/file/d/13vafFIf65g5sqnC2jMY82JIJsSsOC2QQ/view?usp=drive_link"

        # Scarica con gdown
        gdown.download(url, zip_path, quiet=False, fuzzy= True)

        print("🗜 Estrazione dei dati...")

        # Estrai
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(project_dir)
        
        # Elimina lo ZIP
        os.remove(zip_path)
        print("🧹 Pulizia completata. Dataset pronto!")
    else:
        print("✔ Dataset già presente nella directory:", zip_path)

if __name__ == "__main__":
    download_data()