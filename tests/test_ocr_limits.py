import unittest
from unittest.mock import MagicMock

try:
    from PIL import Image

    from services.documents import ocr
except ModuleNotFoundError:
    ocr = None


@unittest.skipUnless(ocr, "requires Pillow")
class OcrImageLimitTest(unittest.TestCase):
    def test_oversized_image_is_rejected_before_decoding(self):
        image = MagicMock(spec=Image.Image)
        image.size = (10_000, 10_000)
        engine = MagicMock()

        with self.assertRaisesRegex(ValueError, "分辨率过大"):
            ocr.ocr_image(image, engine)

        image.convert.assert_not_called()
        engine.ocr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
