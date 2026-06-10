import pytest
from types import SimpleNamespace

# Skip all tests if transformers is not installed
pytest.importorskip("transformers")

from promptmask import PromptMask
from promptmask.backends.transformers_backend import TransformersBackend


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def test_extract_entities_supports_bioes_labels():
    backend = TransformersBackend.__new__(TransformersBackend)
    backend.model = SimpleNamespace(config=SimpleNamespace(id2label={
        0: "O",
        1: "B-private_person",
        2: "E-private_person",
        3: "S-private_email",
    }))

    text = "My name is John Doe at john@example.com"
    predictions = [_Scalar(i) for i in [0, 0, 0, 1, 2, 0, 3]]
    offset_mapping = [
        (_Scalar(0), _Scalar(2)),
        (_Scalar(2), _Scalar(7)),
        (_Scalar(7), _Scalar(10)),
        (_Scalar(10), _Scalar(15)),
        (_Scalar(15), _Scalar(19)),
        (_Scalar(19), _Scalar(23)),
        (_Scalar(22), _Scalar(39)),
    ]

    entities = backend._extract_entities(text, predictions, offset_mapping)

    assert entities == [
        {"type": "private_person", "start": 11, "end": 19, "text": "John Doe"},
        {"type": "private_email", "start": 23, "end": 39, "text": "john@example.com"},
    ]


class TestTransformersBackend:
    @pytest.fixture
    def transformers_config(self):
        return {
            "llm_api": {
                "backend": "transformers",
                "model": "openai/privacy-filter",
                "device": "cpu"
            }
        }

    def test_backend_initialization(self, transformers_config):
        """Test that transformers backend initializes successfully."""
        pm = PromptMask(config=transformers_config)
        assert pm.backend_type == "transformers"
        assert pm.transformers_backend is not None
        assert pm.client is None

    def test_mask_name_pii(self, transformers_config):
        """Test masking of person names."""
        pm = PromptMask(config=transformers_config)
        text = "My name is John Doe"
        masked, mask_map = pm.mask_str(text)

        assert "John Doe" not in masked
        assert len(mask_map) > 0
        assert "John Doe" in mask_map

    def test_mask_email(self, transformers_config):
        """Test masking of email addresses."""
        pm = PromptMask(config=transformers_config)
        text = "Contact me at john@example.com"
        masked, mask_map = pm.mask_str(text)

        assert "john@example.com" not in masked
        assert len(mask_map) > 0

    def test_mask_multiple_pii(self, transformers_config):
        """Test masking multiple PII types."""
        pm = PromptMask(config=transformers_config)
        text = "My name is Jane Smith and email is jane.smith@example.com"
        masked, mask_map = pm.mask_str(text)

        assert "Jane Smith" not in masked or "jane.smith@example.com" not in masked
        assert len(mask_map) >= 1

    def test_no_pii(self, transformers_config):
        """Test text with no PII returns empty mask map."""
        pm = PromptMask(config=transformers_config)
        text = "The weather is nice today."
        masked, mask_map = pm.mask_str(text)

        assert masked == text
        assert len(mask_map) == 0

    def test_empty_text(self, transformers_config):
        """Test empty text handling."""
        pm = PromptMask(config=transformers_config)
        masked, mask_map = pm.mask_str("")

        assert masked == ""
        assert mask_map == {}

    def test_mask_unmask_cycle(self, transformers_config):
        """Test full mask-unmask cycle."""
        pm = PromptMask(config=transformers_config)
        original_text = "My name is Alice and email is alice@test.com"

        masked, mask_map = pm.mask_str(original_text)
        unmasked = pm.unmask_str(masked, mask_map)

        assert unmasked == original_text

    @pytest.mark.asyncio
    async def test_async_mask_str(self, transformers_config):
        """Test async masking."""
        pm = PromptMask(config=transformers_config)
        text = "My name is Bob"

        masked, mask_map = await pm.async_mask_str(text)

        assert "Bob" not in masked or len(mask_map) == 0

    def test_mask_messages(self, transformers_config):
        """Test masking messages."""
        pm = PromptMask(config=transformers_config)
        messages = [
            {"role": "user", "content": "My name is Charlie"},
            {"role": "assistant", "content": "Hello Charlie"}
        ]

        masked_messages, mask_map = pm.mask_messages(messages)

        assert len(masked_messages) == 2
        if len(mask_map) > 0:
            assert "Charlie" not in masked_messages[0]["content"]
