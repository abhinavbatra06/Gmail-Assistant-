
import os
import yaml

class StorageManager:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.paths = self.cfg["paths"]
        self._ensure_directories()

    def _ensure_directories(self):
        # Create all required folders
        for key, path in self.paths.items():
            if key == "db_path":
                os.makedirs(os.path.dirname(path), exist_ok=True)
            else:
                os.makedirs(path, exist_ok=True)

    @property
    def db_path(self):
        return self.paths["db_path"]

    def path_for_email(self, msg_id):
        return os.path.join(self.paths["raw_emails"], f"{msg_id}.eml")

    def path_for_metadata(self, msg_id):
        return os.path.join(self.paths["metadata"], f"{msg_id}.json")

    def path_for_docling(self, msg_id):
        return os.path.join(self.paths["docling"], f"{msg_id}.json")

    def path_for_attachment(self, msg_id, filename):
        clean = filename.replace("/", "_")
        return os.path.join(self.paths["attachments"], f"{msg_id}_{clean}")
