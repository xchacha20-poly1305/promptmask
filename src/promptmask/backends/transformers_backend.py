import asyncio
from typing import Dict, Tuple
from collections import defaultdict

class TransformersBackend:
    def __init__(self, model_name: str, device: str = "auto", mask_wrapper: dict = None):
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            import torch
        except ImportError:
            raise ImportError(
                "transformers backend requires additional dependencies. "
                "Install with: pip install 'promptmask[transformers]'"
            )

        self.model_name = model_name
        self.device = device
        self.mask_wrapper = mask_wrapper or {"left": "${", "right": "}"}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if device == "auto":
            self.model = AutoModelForTokenClassification.from_pretrained(model_name, device_map="auto")
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.model.to(device)

        self.model.eval()
        self.torch = torch

    def mask_text(self, text: str) -> Tuple[str, Dict[str, str]]:
        if not text:
            return "", {}

        inputs = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        predictions = self.torch.argmax(outputs.logits, dim=-1)[0]

        entities = self._extract_entities(text, predictions, offset_mapping)
        mask_map = self._generate_mask_map(entities)
        masked_text = self._apply_masks(text, entities, mask_map)

        return masked_text, mask_map

    async def async_mask_text(self, text: str) -> Tuple[str, Dict[str, str]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.mask_text, text)

    def _extract_entities(self, text: str, predictions, offset_mapping):
        entities = []
        current_entity = None
        label_names = self.model.config.id2label

        for idx, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            if start == end:
                continue

            label = label_names[pred.item()]

            if label == "O":
                if current_entity:
                    self._append_entity(entities, text, current_entity)
                    current_entity = None
                continue

            prefix, entity_type = label.split("-", 1)

            if prefix == "S":
                if current_entity:
                    self._append_entity(entities, text, current_entity)
                    current_entity = None
                self._append_entity(entities, text, {
                    "type": entity_type,
                    "start": start.item(),
                    "end": end.item(),
                })
            elif prefix == "B":
                if current_entity:
                    self._append_entity(entities, text, current_entity)
                current_entity = {
                    "type": entity_type,
                    "start": start.item(),
                    "end": end.item(),
                }
            elif prefix in {"I", "E"}:
                if current_entity:
                    current_entity["end"] = end.item()
                else:
                    current_entity = {
                        "type": entity_type,
                        "start": start.item(),
                        "end": end.item(),
                    }
                if prefix == "E":
                    self._append_entity(entities, text, current_entity)
                    current_entity = None

        if current_entity:
            self._append_entity(entities, text, current_entity)

        return entities

    def _append_entity(self, entities, text: str, entity):
        start, end = self._normalize_entity_span(text, entity["start"], entity["end"])
        if start >= end:
            return
        entities.append({
            "type": entity["type"],
            "start": start,
            "end": end,
            "text": text[start:end],
        })

    def _normalize_entity_span(self, text: str, start: int, end: int):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    def _generate_mask_map(self, entities):
        type_counters = defaultdict(int)
        mask_map = {}
        mask_left, mask_right = self.mask_wrapper["left"], self.mask_wrapper["right"]

        for entity in entities:
            entity_text = entity["text"]
            entity_type = entity["type"].upper().replace("-", "_")
            type_counters[entity_type] += 1
            mask_name = f"{mask_left}{entity_type}_{type_counters[entity_type]}{mask_right}"
            mask_map[entity_text] = mask_name

        return mask_map

    def _apply_masks(self, text: str, entities, mask_map):
        sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)
        masked_text = text

        for entity in sorted_entities:
            mask = mask_map[entity["text"]]
            masked_text = masked_text[:entity["start"]] + mask + masked_text[entity["end"]:]

        return masked_text
