"""GLiNER2 classification export wrapper and schema preprocessing.

Origin: FluidInference/gliner2-5-small-coreml (Apache-2.0), see NOTICE.
``prepare_tasks_with_processor`` is a kleinhirn addition for multi-task
schemas; it exposes the collator's per-marker group ids (cls_group_index).

For goldens the wrapper runs UNPATCHED in fp32. The patched attention
(``coreml_safe_attention_forward``, static scale, frozen build_rpos) is
allowed only for trace-based exports, and only after showing patched vs
unpatched logits stay within 1e-4 (see convert-coreml.py upstream).
"""
import numpy as np
import torch
from torch import nn
from gliner2 import Schema
from gliner2.models.base import load_extractor_tokenizer
from gliner2.processor import SchemaTransformer
from gliner2.training.trainer import ExtractorCollator
from transformers.models.deberta_v2 import modeling_deberta_v2


def coreml_safe_attention_forward(
    self, hidden_states, attention_mask, output_attentions=False,
    query_states=None, relative_pos=None, rel_embeddings=None,
):
    """Native DeBERTa attention with a finite mask sentinel for FP16 Core ML."""
    if query_states is None:
        query_states = hidden_states
    query = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    key = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    value = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    factor = 1 + int("c2p" in self.pos_att_type) + int("p2c" in self.pos_att_type)
    scale = modeling_deberta_v2.scaled_size_sqrt(query, factor)
    scores = torch.bmm(query, key.transpose(-1, -2) / scale.to(dtype=query.dtype))
    if self.relative_attention:
        relative = self.disentangled_attention_bias(
            query, key, relative_pos, self.pos_dropout(rel_embeddings), factor
        )
        scores = scores + relative
    scores = scores.view(-1, self.num_attention_heads, scores.size(-2), scores.size(-1))
    scores = scores.masked_fill(~attention_mask.bool(), -1e4)
    probabilities = self.dropout(torch.softmax(scores, dim=-1))
    context = torch.bmm(probabilities.view(-1, probabilities.size(-2), probabilities.size(-1)), value)
    context = context.view(-1, self.num_attention_heads, context.size(-2), context.size(-1))
    context = context.permute(0, 2, 1, 3).contiguous()
    context = context.view(context.size()[:-2] + (-1,))
    return (context, probabilities) if output_attentions else (context, None)


class GLiNER2ClassificationExport(nn.Module):
    """Native GLiNER2 classification path with explicit marker routing."""

    def __init__(self, native: nn.Module):
        super().__init__()
        self.encoder = native.encoder
        self.classifier = native.classifier
        self.temperature = float(native.boundary_settings.classification_temperature)

    def forward(self, input_ids, attention_mask, marker_indices, marker_mask):
        hidden = self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long()).last_hidden_state
        indices = marker_indices.long().unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        states = hidden.gather(1, indices)
        logits = self.classifier(states).squeeze(-1) / self.temperature
        logits = torch.where(marker_mask > 0.5, logits, torch.full_like(logits, -1e4))
        return logits, torch.softmax(logits, dim=-1)


def load_processor(tokenizer_dir: str):
    """Load only the tokenizer and schema formatter needed by the exported model."""
    return SchemaTransformer(tokenizer=load_extractor_tokenizer(tokenizer_dir), token_pooling="first")


def native_batch(native, text: str, task: str, labels: list[str], length: int):
    schema = Schema().classification(task, labels)
    collator = ExtractorCollator(native.processor, is_training=False, max_len=length, architecture=native.architecture)
    return collator([(text, schema.build())])


def prepare_classification(native, text: str, task: str, labels: list[str], length: int, max_options: int):
    return prepare_with_processor(native.processor, text, task, labels, length, max_options)


def prepare_with_processor(processor, text: str, task: str, labels: list[str], length: int, max_options: int):
    return prepare_tasks_with_processor(processor, text, {task: labels}, length, max_options)


def prepare_tasks_with_processor(processor, text: str, tasks: dict[str, list[str]], length: int, max_options: int):
    """Pad a (possibly multi-task) classification schema into a fixed bucket.

    Returns input_ids/attention_mask [1, length], marker_indices/marker_mask
    [1, max_options] and marker_groups [1, max_options]: the classification
    group each marker belongs to (index into ``tasks`` order). Softmax runs
    per group, matching the native per-task decode.
    """
    total_labels = sum(len(labels) for labels in tasks.values())
    if not 1 <= total_labels <= max_options:
        raise ValueError(f"Expected 1..{max_options} labels in total, got {total_labels}")
    schema = Schema()
    for task, labels in tasks.items():
        schema = schema.classification(task, labels)
    collator = ExtractorCollator(processor, is_training=False, max_len=length, architecture="boundary")
    batch = collator([(text, schema.build())])
    ids = batch.input_ids.numpy()
    attention = batch.attention_mask.numpy()
    indices = batch.cls_marker_indices.numpy()
    mask = batch.cls_marker_mask.numpy()
    groups = batch.cls_group_index.numpy()
    seq_len = ids.shape[1]
    if seq_len > length or indices.shape[1] != total_labels or int(mask.sum()) != total_labels:
        raise ValueError("Input exceeds bucket or classification markers were truncated")
    # cls_group_index counts schema groups, not just classifications: map the
    # raw group ids onto the dense classification order 0..n_tasks-1.
    task_types = batch.task_types[0]
    cls_group_order = [g for g, t in enumerate(task_types) if t == "classifications"]
    remap = {g: i for i, g in enumerate(cls_group_order)}
    dense_groups = np.vectorize(remap.get)(groups).astype(np.int64)
    ids = np.pad(ids, ((0, 0), (0, length - ids.shape[1])), constant_values=processor.tokenizer.pad_token_id)
    attention = np.pad(attention, ((0, 0), (0, length - attention.shape[1])))
    indices = np.pad(indices, ((0, 0), (0, max_options - indices.shape[1])))
    mask = np.pad(mask, ((0, 0), (0, max_options - mask.shape[1])))
    dense_groups = np.pad(dense_groups, ((0, 0), (0, max_options - dense_groups.shape[1])))
    return {
        "input_ids": ids.astype(np.int32),
        "attention_mask": attention.astype(np.int32),
        "marker_indices": indices.astype(np.int32),
        "marker_mask": mask.astype(np.float32),
        "marker_groups": dense_groups.astype(np.int32),
        "seq_len": seq_len,
    }
