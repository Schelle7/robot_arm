from pathlib import Path
import shutil

def snapshot_model_files(model_path: str, output_directory: str) -> None:
    source_file = Path(model_path).resolve()
    source_directory = source_file.parent
    destination_directory = Path(output_directory) / "model"

    shutil.copytree(
        source_directory,
        destination_directory,
        dirs_exist_ok=True,
    )