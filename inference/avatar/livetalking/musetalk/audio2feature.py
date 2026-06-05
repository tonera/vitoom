"""Whisper feature extractor (vendored from LiveTalking
``avatars/musetalk/whisper/audio2feature.py``)。

去掉了上游：

* ``from .whisper import load_model`` —— 死代码（``audio2feat`` 实际只用
  HuggingFace ``transformers`` 的 ``WhisperModel``）
* ``__main__`` 调试块
* ``sys.path.append("..")`` —— 跨模块的 sys.path hack 没必要

上游有 ``get_sliced_feature_sparse`` / ``feature2chunks`` 方法，sidecar 实际
只用 ``audio2feat`` 这一个入口（被我们的 ``feature_buffer.py`` 调用），但
保留以便万一 runtime 想切到 sparse 模式不用回 vendor。
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoFeatureExtractor, WhisperModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


class Audio2Feature():
    def __init__(self, model_path: str, whisper_model_type: str = "tiny"):
        # whisper_model_type 仅作为标记保留（上游接口形参），实际使用
        # transformers 的 from_pretrained 路径（model_path）。
        del whisper_model_type
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        self.whisper = WhisperModel.from_pretrained(model_path)
        self.whisper = self.whisper.to(device=device, dtype=weight_dtype).eval()
        self.whisper.requires_grad_(False)

    def get_sliced_feature(self, feature_array, vid_idx,
                           audio_feat_length=(2, 2), fps: int = 25):
        """Get sliced features based on a given video frame index."""
        length = len(feature_array)
        selected_feature = []
        selected_idx = []

        center_idx = int(vid_idx * 50 / fps)
        left_idx = center_idx
        right_idx = center_idx + (audio_feat_length[0] + audio_feat_length[1] + 1) * 2

        for idx in range(left_idx, right_idx):
            idx = max(0, idx)
            idx = min(length - 1, idx)
            x = feature_array[idx]
            selected_feature.append(x)
            selected_idx.append(idx)

        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, 384)
        return selected_feature, selected_idx

    def feature2chunks(self, feature_array, fps: int, batch_size: int,
                       audio_feat_length=(2, 2), start: int = 0):
        whisper_chunks = []
        for i in range(batch_size):
            selected_feature, _ = self.get_sliced_feature(
                feature_array=feature_array, vid_idx=i + start,
                audio_feat_length=audio_feat_length, fps=fps,
            )
            whisper_chunks.append(selected_feature)
        return whisper_chunks

    def audio2feat(self, wav_data):
        """Float32 16k mono PCM (numpy) → whisper hidden-state stack (N, K, 384)."""
        input_feature = self.feature_extractor(
            wav_data, return_tensors="pt", sampling_rate=16000,
        ).input_features
        input_feature = input_feature.to(device).to(weight_dtype)
        whisper_feature = self.whisper.encoder(
            input_feature, output_hidden_states=True,
        ).hidden_states
        whisper_feature = torch.stack(whisper_feature, dim=2)
        return whisper_feature.squeeze(0).cpu().numpy()


__all__ = ["Audio2Feature"]
