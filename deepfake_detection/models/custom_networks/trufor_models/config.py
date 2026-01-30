class AttrNamespace(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __contains__(self, item):
        return item in self.keys()


def default_config() -> AttrNamespace:
    extra = AttrNamespace(
        BACKBONE="mit_b2",
        DECODER="MLPDecoder",
        DECODER_EMBED_DIM=512,
        PREPRC="imagenet",
        BN_EPS=0.001,
        BN_MOMENTUM=0.1,
        DETECTION="confpool",
        CONF=True,
        NP_WEIGHTS="",
    )

    model = AttrNamespace(
        NAME="detconfcmx",
        MODS=("RGB", "NP++"),
        PRETRAINED="",
        EXTRA=extra,
    )

    dataset = AttrNamespace(NUM_CLASSES=2)

    return AttrNamespace(MODEL=model, DATASET=dataset)