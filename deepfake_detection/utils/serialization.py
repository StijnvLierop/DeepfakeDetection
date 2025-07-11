from json import JSONEncoder, JSONDecoder

from deepfake_detection.data import Instance, ImageInstance, ImageSequenceInstance, VideoInstance


KEY2INSTANCE = {'image': ImageInstance, 'image_sequence': ImageSequenceInstance, 'video': VideoInstance}
INSTANCE2KEY = {ImageInstance : 'image', ImageSequenceInstance: 'image_sequence', VideoInstance: 'video'}

class InstanceEncoder(JSONEncoder):
    """
    Class that encodes instances into a JSON serializable format.
    Only metadata is encoded, but data itself is not serialized.
    """
    def default(self, instance: Instance):
        return {'path': str(instance.path),
                'label': instance.label,
                'instance_type': INSTANCE2KEY[instance.__class__]}


class InstanceDecoder(JSONDecoder):
    """
    Class that decodes JSON data into an instance.
    Only metadata is decoded, but data itself is not stored in JSON format.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.entry_object_hook, *args, **kwargs)

    def entry_object_hook(self, obj):
        return KEY2INSTANCE[obj['instance_type']](label=obj['label'], path=obj['path'])