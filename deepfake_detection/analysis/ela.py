import tempfile

from PIL import Image, ImageChops, ImageEnhance


def ela(img: Image.Image, quality: int = 95) -> Image.Image:
    """
    Computes the Error Level Analysis (ELA) image of a given image instance.

    :param img: Input PIL image.
    :param quality: JPEG compression quality (1–95) used for recompression. Lower values
                    produce stronger compression artefacts, making differences more visible.
    """

    # Temporary save filename for compressed image
    tmp_savename = tempfile.NamedTemporaryFile(delete=True).name

    img = img.convert("RGB")

    # Save compressed image
    img.save(tmp_savename, "JPEG", quality=quality)

    # Open compressed image
    resaved_im = Image.open(tmp_savename)

    # Calculate difference between image and compressed image
    ela_im = ImageChops.difference(img, resaved_im)

    # Get maximum pixels and calculate ela brightness scale
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff

    # Enhance difference image with given scale
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    return ela_im
