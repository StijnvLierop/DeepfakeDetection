from pathlib import Path
from typing import List

from deepfake_detection.data import MapStyleDatasetMixin, Dataset, Instance, FileImageInstance
from deepfake_detection.data.annotation import Annotation


class CNNDetectDataset(MapStyleDatasetMixin, Dataset):
    """
    This class can be used to load the dataset presented by Wang et al. (2020).
    More info about the dataset can be found here: https://github.com/PeterWang512/CNNDetection
    """

    def __init__(self, path: str):
        super().__init__(name="CNNDetect")
        self.path = path

        # Index dataset
        self.instance_paths, self.authenticity_labels, self.source_labels, self.semantic_labels = self._index()

    def _index(self) -> (List[Path], List[int], List[str], List[str]):
        instance_paths = []
        authenticity_labels = []
        source_labels = []
        semantic_labels = []  # New list for semantic classes

        root = Path(self.path)
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        for img_path in root.rglob('*'):
            if img_path.suffix.lower() in extensions:
                parent_name = img_path.parent.name

                if parent_name in ['0_real', '1_fake']:
                    # Get path parts relative to root
                    parts = img_path.relative_to(root).parts

                    source = parts[0]
                    auth_label = parent_name.split('_')[1]

                    # Logic for semantic label
                    if len(parts) == 4:
                        # The semantic label is the second element
                        semantic = parts[1]
                    else:
                        # For biggan, gaugan, etc., there is no semantic sub-label
                        semantic = "none"

                    instance_paths.append(img_path)
                    authenticity_labels.append(auth_label)
                    source_labels.append(source)
                    semantic_labels.append(semantic)

        return instance_paths, authenticity_labels, source_labels, semantic_labels

    def __getitem__(self, idx: int) -> Instance:
        # Get instance path and labels
        path = self.instance_paths[idx]
        authenticity_label = self.authenticity_labels[idx]
        source_label = self.source_labels[idx]
        semantic_label = self.semantic_labels[idx]

        # Return instance
        return FileImageInstance(str(path),
                                 Annotation(authenticity_label=authenticity_label,
                                            source_label=source_label),
                                 meta={'semantic_content': semantic_label}
                                 )

    def __len__(self):
        return len(self.instance_paths)
