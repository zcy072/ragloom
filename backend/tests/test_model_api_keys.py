import os
import sqlite3
import unittest
from unittest.mock import patch

from cryptography.fernet import Fernet

import app_storage
from services.core import model_api_keys
from tests.support import TEST_MODEL_API_KEY_ENCRYPTION_KEY, IsolatedStorageTestCase

ENV = model_api_keys.ENCRYPTION_KEY_ENV


class ModelApiKeyEncryptionTest(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {ENV: TEST_MODEL_API_KEY_ENCRYPTION_KEY})
        environment.start()
        self.addCleanup(environment.stop)

    def test_roundtrip_with_the_configured_key(self):
        ciphertext = model_api_keys.encrypt_api_key("sk-roundtrip")

        self.assertTrue(ciphertext.startswith(model_api_keys.ENCRYPTED_PREFIX))
        self.assertNotIn("sk-roundtrip", ciphertext)
        self.assertEqual(model_api_keys.decrypt_api_key(ciphertext), "sk-roundtrip")

    def test_same_plaintext_encrypts_to_different_ciphertexts(self):
        first = model_api_keys.encrypt_api_key("same")
        second = model_api_keys.encrypt_api_key("same")

        self.assertNotEqual(first, second)
        self.assertEqual(model_api_keys.decrypt_api_key(second), "same")

    def test_empty_values_stay_empty(self):
        self.assertEqual(model_api_keys.encrypt_api_key(""), "")
        self.assertEqual(model_api_keys.encrypt_api_key(None), "")
        self.assertEqual(model_api_keys.decrypt_api_key(""), "")
        self.assertEqual(model_api_keys.decrypt_api_key(None), "")

    def test_input_that_looks_encrypted_is_treated_as_a_plain_string(self):
        ciphertext = model_api_keys.encrypt_api_key("inner")
        for value in ("enc:v1:not-a-token", ciphertext):
            wrapped = model_api_keys.encrypt_api_key(value)
            self.assertNotEqual(wrapped, value)
            self.assertEqual(model_api_keys.decrypt_api_key(wrapped), value)

    def test_decryption_rejects_wrong_key(self):
        ciphertext = model_api_keys.encrypt_api_key("secret")
        with patch.dict(os.environ, {ENV: Fernet.generate_key().decode()}):
            with self.assertRaises(model_api_keys.ModelApiKeyEncryptionError) as raised:
                model_api_keys.decrypt_api_key(ciphertext)
        self.assertNotIn(ciphertext, str(raised.exception))

    def test_decryption_rejects_tampering_unknown_prefix_and_plaintext(self):
        ciphertext = model_api_keys.encrypt_api_key("secret")
        token = ciphertext.removeprefix(model_api_keys.ENCRYPTED_PREFIX)
        tampered = ciphertext[:-4] + ("AAAA" if not ciphertext.endswith("AAAA") else "BBBB")
        for value in (
            tampered,
            "enc:v2:" + token,
            token,
            "sk-plaintext",
            # 旧版本的本地占位符同样不再被放行。
            "ollama",
            model_api_keys.ENCRYPTED_PREFIX + "é",
        ):
            with self.subTest(value=value[:20]):
                with self.assertRaises(model_api_keys.ModelApiKeyEncryptionError):
                    model_api_keys.decrypt_api_key(value)

    def test_missing_or_malformed_master_key_is_rejected(self):
        for value in ("", "not-a-fernet-key", "a" * 44):
            with self.subTest(value=value), patch.dict(os.environ, {ENV: value}):
                with self.assertRaisesRegex(model_api_keys.ModelApiKeyEncryptionError, ENV):
                    model_api_keys.validate_master_key()
                with self.assertRaises(model_api_keys.ModelApiKeyEncryptionError):
                    model_api_keys.encrypt_api_key("secret")


class MasterKeyStartupTest(IsolatedStorageTestCase):
    def test_missing_or_invalid_master_key_blocks_startup(self):
        for value in (None, "too-short"):
            with self.subTest(value=value):
                environment = {ENV: value} if value is not None else {}
                with patch.dict(os.environ, environment):
                    if value is None:
                        os.environ.pop(ENV, None)
                    with self.assertRaises(model_api_keys.ModelApiKeyEncryptionError):
                        app_storage.initialize_app_storage()

    def test_startup_does_not_decrypt_stored_keys(self):
        app_storage.initialize_app_storage()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE models_config SET api_key = 'enc:v1:corrupt' WHERE provider_name = 'OpenAI'"
            )
        # 单条坏密文只在检查或使用该凭证时报错，不阻止应用启动。
        with patch.dict(os.environ, {ENV: Fernet.generate_key().decode()}):
            app_storage.initialize_app_storage()


if __name__ == "__main__":
    unittest.main()
