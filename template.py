import os 
from pathlib import Path
import logging 

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s]: %(message)s")

project_name = "icebreaker"

list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/modules/__init__.py",
    f"{project_name}/modules/app.py",
    f"{project_name}/modules/config.py",
    f"{project_name}/modules/main.py",
    f"{project_name}/modules/data_extraction.py",
    f"{project_name}/modules/data_processing.py",
    f"{project_name}/modules/llm_interface.py",
    f"{project_name}/modules/query_engine.py",
    f"{project_name}/modules/output_schema.py",
    ".env"
]

for file_path in list_of_files:
    file_path =  Path(file_path)
    file_dir,file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for file: {file_name}")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path,"w") as f:
            pass
            logging.info(f"Creating an empty file: {file_path}")
    
    else:
        logging.info(f"{file_name} already exists")


