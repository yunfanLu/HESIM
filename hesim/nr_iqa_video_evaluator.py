#!/usr/bin/env python3

"""
nr_iqa_video_evaluator.py
Evaluate NR-IQA on a video (given ordered frame paths):
  - NIQE (↓), BRISQUE (↓), MUSIQ (↑), NRQM (↑) via pyiqa
  - CLIP-IQA (↑) via torchmetrics
"""

import os
from typing import Callable, Dict, List

import cv2
import numpy as np
import pyiqa
import torch
import torchvision.transforms as T
from PIL import Image
from torchmetrics.multimodal import CLIPImageQualityAssessment


class NoRefQualityEvaluator:
    """
    Args:
        metrics: e.g., ["niqe","clip_iqa","musiq","brisque","nrqm"]
        device: "cuda" or "cpu"
    """

    def __init__(self, metrics: List[str], device: str = "cuda") -> None:
        self.metrics = metrics
        self.device = device
        self._metric_factory: Dict[str, Callable] = {
            "niqe": self._compute_niqe,
            "clip_iqa": self._compute_clip_iqa,
            "musiq": self._compute_musiq,
            "brisque": self._compute_brisque,
            "nrqm": self._compute_nrqm,
        }
        self._cache = {}

    def evaluate(self, frame_paths: List[str]) -> Dict[str, float]:
        results = {}
        for key in self.metrics:
            if key not in self._metric_factory:
                raise ValueError(f"Unsupported metric: {key}")
            try:
                results[key] = self._metric_factory[key](frame_paths)
            except Exception as e:
                results[key] = float("nan")
                print(f"[WARN] {key} failed: {e}")
        return results

    @staticmethod
    def _imread_rgb(path: str) -> Image.Image:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    @staticmethod
    def _to_tensor(img_pil: Image.Image) -> torch.Tensor:
        return T.ToTensor()(img_pil).unsqueeze(0)

    def _framewise_mean(self, frame_paths: List[str], scorer) -> float:
        scores = []
        for p in frame_paths:
            if not os.path.exists(p):
                continue
            try:
                score = float(scorer(p))
                if score is not None:
                    scores.append(score)
                else:
                    print(f"[WARN] score is None for {p}")
            except Exception:
                continue
        return float(np.mean(scores))

    def _compute_niqe(self, frame_paths: List[str]) -> float:
        if "niqe" not in self._cache:
            self._cache["niqe"] = pyiqa.create_metric("niqe", device=self.device)
        metric = self._cache["niqe"]

        def _score(p):
            x = self._to_tensor(self._imread_rgb(p)).to(self.device)
            return metric(x).item()

        return self._framewise_mean(frame_paths, _score)

    def _compute_brisque(self, frame_paths: List[str]) -> float:
        if "brisque" not in self._cache:
            self._cache["brisque"] = pyiqa.create_metric("brisque", device=self.device)
        metric = self._cache["brisque"]

        def _score(p):
            x = self._to_tensor(self._imread_rgb(p)).to(self.device)
            return metric(x).item()

        return self._framewise_mean(frame_paths, _score)

    def _compute_musiq(self, frame_paths: List[str]) -> float:
        if "musiq" not in self._cache:
            self._cache["musiq"] = pyiqa.create_metric("musiq", device=self.device)
        metric = self._cache["musiq"]

        def _score(p):
            x = self._to_tensor(self._imread_rgb(p)).to(self.device)
            return metric(x).item()

        return self._framewise_mean(frame_paths, _score)

    def _compute_nrqm(self, frame_paths: List[str]) -> float:
        if "nrqm" not in self._cache:
            self._cache["nrqm"] = pyiqa.create_metric("nrqm", device=self.device)
        metric = self._cache["nrqm"]

        def _score(p):
            x = self._to_tensor(self._imread_rgb(p)).to(self.device)
            return metric(x).item()

        return self._framewise_mean(frame_paths, _score)

    def _compute_clip_iqa(self, frame_paths: List[str]) -> float:
        if "clip_iqa" not in self._cache:

            self._cache["clip_iqa"] = CLIPImageQualityAssessment(model_name_or_path="clip_iqa").to(self.device).eval()
        metric = self._cache["clip_iqa"]

        scores = []
        with torch.no_grad():
            for p in frame_paths:
                if not os.path.exists(p):
                    continue
                x = self._to_tensor(self._imread_rgb(p)).to(self.device)
                s = metric(x)
                scores.append(float(s.item()))
        return float(np.mean(scores)) if scores else float("nan")
