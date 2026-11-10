import unittest
from unittest.mock import MagicMock

from PIL import Image

from services.documents import ocr


class OcrImageLimitTest(unittest.TestCase):
    def test_oversized_image_is_rejected_before_decoding(self):
        image = MagicMock(spec=Image.Image)
        image.size = (10_000, 10_000)
        engine = MagicMock()

        with self.assertRaisesRegex(ValueError, "分辨率过大"):
            ocr.ocr_image(image, engine)

        image.convert.assert_not_called()
        engine.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
