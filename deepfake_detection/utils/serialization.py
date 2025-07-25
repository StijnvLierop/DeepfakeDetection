from json import JSONEncoder, JSONDecoder

import numpy as np

from deepfake_detection.data import ImageInstance, FileImageSequenceInstance, FileVideoInstance, \
    FileImageInstance


class FileImageInstanceEncoder(JSONEncoder):
    """
    Class that encodes FileImageInstances into a JSON serializable format.
    Only metadata is encoded, but data itself is not serialized.
    """
    def default(self, instance: FileImageInstance):
        return {'path': str(instance.path),
                'label': instance.label}


class FileImageInstanceDecoder(JSONDecoder):
    """
    Class that decodes JSON data into a FileImageInstance.
    Only metadata is decoded, but data itself is not stored in JSON format.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.entry_object_hook, *args, **kwargs)

    def entry_object_hook(self, obj):
        return FileImageInstance(**obj)

class ImageInstanceEncoder(JSONEncoder):
    """
    Class that encodes ImageInstances into a JSON serializable format.
    Only metadata is encoded, but data itself is not serialized.
    """
    def default(self, instance: FileImageInstance):
        return {'data': np.array(instance.data).tolist(),
                'label': instance.label}


class ImageInstanceDecoder(JSONDecoder):
    """
    Class that decodes JSON data into a ImageInstance.
    Only metadata is decoded, but data itself is not stored in JSON format.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.entry_object_hook, *args, **kwargs)

    def entry_object_hook(self, obj):
        return ImageInstance(**obj)

class FileVideoInstanceEncoder(JSONEncoder):
    """
    Class that encodes FileVideoInstances into a JSON serializable format.
    Only metadata is encoded, but data itself is not serialized.
    """
    def default(self, instance: FileVideoInstance):
        return {'path': str(instance.path),
                'label': instance.label}


class FileVideoInstanceDecoder(JSONDecoder):
    """
    Class that decodes JSON data into a FileVideoInstance.
    Only metadata is decoded, but data itself is not stored in JSON format.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.entry_object_hook, *args, **kwargs)

    def entry_object_hook(self, obj):
        return FileVideoInstance(**obj)

class FileImageSequenceInstanceEncoder(JSONEncoder):
    """
    Class that encodes FileImageSequenceInstances into a JSON serializable format.
    Only metadata is encoded, but data itself is not serialized.
    """
    def default(self, instance: FileImageSequenceInstance):
        return {'path': str(instance.path),
                'label': instance.label}


class FileImageSequenceInstanceDecoder(JSONDecoder):
    """
    Class that decodes JSON data into a FileImageSequenceInstance.
    Only metadata is decoded, but data itself is not stored in JSON format.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.entry_object_hook, *args, **kwargs)

    def entry_object_hook(self, obj):
        return FileImageSequenceInstance(**obj)