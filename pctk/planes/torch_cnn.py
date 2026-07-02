"""Optional CNN backend for fetal-plane classification (PyTorch).

Mirrors the :class:`~pctk.planes.model.PlaneClassifier` API
(``train`` / ``evaluate`` / ``predict`` / ``save`` / ``load``) but trains a small
convolutional network instead of the scikit-learn baseline. Torch is imported
lazily inside methods, so importing this module (and all of ``pctk.planes``)
never requires torch; a clear error is raised only if you actually use the CNN
backend without it installed.

Install with:  ``pip install torch torchvision``  (CPU wheels are fine for a
demo; a GPU makes real FETAL_PLANES_DB training practical).

Still sex-neutral: the network's output layer is over anatomical plane classes,
never sex.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _require_torch():
    if not torch_available():
        raise ImportError(
            "the CNN backend needs PyTorch. Install it with:\n"
            "    pip install torch torchvision\n"
            "or use the default scikit-learn backend (PlaneClassifier).")


@dataclass
class CNNConfig:
    size: int = 64
    epochs: int = 8
    batch_size: int = 32
    lr: float = 1e-3
    device: Optional[str] = None      # "cuda" / "cpu" / None=auto
    seed: int = 0
    classes: List[str] = field(default_factory=list)


def _build_net(n_classes: int):
    import torch.nn as nn

    class SmallCNN(nn.Module):
        def __init__(self, n_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.3),
                nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return SmallCNN(n_classes)


def _load_tensor(path: str, size: int):
    import torch
    arr = np.asarray(Image.open(path).convert("L").resize((size, size)),
                     dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)   # (1, H, W)


class CNNPlaneClassifier:
    """PyTorch CNN with the same interface as the sklearn PlaneClassifier."""

    backend = "torch"

    def __init__(self, config: Optional[CNNConfig] = None):
        self.config = config or CNNConfig()
        self.net = None
        self.classes_: List[str] = list(self.config.classes)

    def _device(self):
        import torch
        if self.config.device:
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _batches(self, paths, labels=None):
        import torch
        idx = np.arange(len(paths))
        bs = self.config.batch_size
        for s in range(0, len(idx), bs):
            sel = idx[s:s + bs]
            X = torch.stack([_load_tensor(paths[i], self.config.size) for i in sel])
            if labels is None:
                yield X, None
            else:
                y = torch.tensor([labels[i] for i in sel], dtype=torch.long)
                yield X, y

    def train(self, df: pd.DataFrame, split: str = "train") -> Dict:
        _require_torch()
        import torch
        import torch.nn as nn

        torch.manual_seed(self.config.seed)
        sub = df[df["split"] == split] if "split" in df.columns else df
        self.classes_ = sorted(sub["label"].unique().tolist())
        cls_to_i = {c: i for i, c in enumerate(self.classes_)}
        paths = sub["image_path"].tolist()
        labels = [cls_to_i[c] for c in sub["label"]]

        device = self._device()
        self.net = _build_net(len(self.classes_)).to(device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)
        loss_fn = nn.CrossEntropyLoss()

        self.net.train()
        for epoch in range(self.config.epochs):
            order = np.random.permutation(len(paths))
            p = [paths[i] for i in order]
            l = [labels[i] for i in order]
            total = 0.0
            for X, y in self._batches(p, l):
                X, y = X.to(device), y.to(device)
                opt.zero_grad()
                loss = loss_fn(self.net(X), y)
                loss.backward()
                opt.step()
                total += float(loss) * len(y)
            print(f"  epoch {epoch + 1}/{self.config.epochs} "
                  f"loss={total / len(paths):.4f}")
        return {"n_train": len(paths), "classes": self.classes_,
                "device": device}

    def evaluate(self, df: pd.DataFrame, split: str = "test") -> Dict:
        _require_torch()
        import torch
        from sklearn.metrics import accuracy_score, classification_report

        sub = df[df["split"] == split] if "split" in df.columns else df
        cls_to_i = {c: i for i, c in enumerate(self.classes_)}
        paths = sub["image_path"].tolist()
        y_true = [cls_to_i[c] for c in sub["label"]]
        device = self._device()
        self.net.eval()
        preds = []
        with torch.no_grad():
            for X, _ in self._batches(paths):
                preds.extend(self.net(X.to(device)).argmax(1).cpu().tolist())
        return {
            "n_test": len(paths),
            "accuracy": float(accuracy_score(y_true, preds)),
            "report": classification_report(
                y_true, preds, target_names=self.classes_, zero_division=0),
            "labels": self.classes_,
        }

    def predict(self, image) -> Dict:
        _require_torch()
        import torch
        import torch.nn.functional as F

        device = self._device()
        if isinstance(image, str):
            x = _load_tensor(image, self.config.size).unsqueeze(0)
        else:
            arr = np.asarray(Image.fromarray(np.asarray(image)).convert("L")
                             .resize((self.config.size, self.config.size)),
                             dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        self.net.eval()
        with torch.no_grad():
            proba = F.softmax(self.net(x.to(device)), dim=1)[0].cpu().numpy()
        i = int(proba.argmax())
        return {"label": self.classes_[i], "confidence": float(proba[i]),
                "proba": {c: float(p) for c, p in zip(self.classes_, proba)}}

    def save(self, path: str) -> None:
        _require_torch()
        import torch
        torch.save({"state_dict": self.net.state_dict(),
                    "classes": self.classes_, "config": self.config,
                    "backend": "torch"}, path)

    @classmethod
    def load(cls, path: str) -> "CNNPlaneClassifier":
        _require_torch()
        import torch
        blob = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(blob.get("config"))
        obj.classes_ = blob["classes"]
        obj.net = _build_net(len(obj.classes_))
        obj.net.load_state_dict(blob["state_dict"])
        return obj
