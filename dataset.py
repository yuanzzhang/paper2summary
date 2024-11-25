import json
import pickle
from pathlib import Path

from torch.utils.data import Dataset



class ScientificPapersDataset(Dataset):
    def __init__(self, file_path):
        """
        Args:
            file_path (str): Path to the dataset file (.txt)
        """
        self.file_path = Path(file_path)
        self.index_file_path = self.file_path.with_suffix('.idx')  # Store the index as .idx
        self.file = None  # Placeholder for file handle
        self.line_positions = self._load_index() or self._create_index()

        # The total number of examples is the length of the index
        self.num_examples = len(self.line_positions)


    def _load_index(self):
        """Load the precomputed index from a file if it exists."""
        if self.index_file_path.exists():
            print(f"Loading index from {self.index_file_path}...")
            with open(self.index_file_path, 'rb') as f:
                return pickle.load(f)
        return None


    def _create_index(self):
        """Create a new index by reading the dataset file line by line."""
        print(f"Creating index for {self.file_path}...")
        line_positions = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            pos = 0
            while f.readline():
                line_positions.append(pos)
                pos = f.tell()

        # Save the index for future use
        print(f"Saving index to {self.index_file_path}...")
        with open(self.index_file_path, 'wb') as f:
            pickle.dump(line_positions, f)

        return line_positions


    def __len__(self):
        return self.num_examples


    def __getitem__(self, idx):
        if self.file is None:
            # Open the file if not already open
            self.file = open(self.file_path, 'r', encoding='utf-8')

        # Seek to the correct position and read one line
        self.file.seek(self.line_positions[idx])
        line = self.file.readline()
        data = json.loads(line)

        return {
            'article_id': data['article_id'],
            'article': '\n'.join(data['article_text']),
            'abstract': '\n'.join(data['abstract_text']).replace('<S>', '').replace('</S>', '')
        }


    def close(self):
        """Close the file handle."""
        if self.file is not None:
            self.file.close()
            self.file = None


    def __del__(self):
        """Ensure the file handle is closed when the object is deleted."""
        self.close()
