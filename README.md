# Damo-Embedding

[![Deploy to GitHub Pages](https://github.com/uopensail/damo-embedding/actions/workflows/gh-pages.yml/badge.svg)](https://uopensail.github.io/damo-embedding/docs/Intro) [![Build and upload to PyPI](https://github.com/uopensail/damo-embedding/actions/workflows/main.yml/badge.svg?event=release)](https://pypi.org/project/damo-embedding/)
# Quick Install

```shell
pip install damo-embedding
```

# Example

## DeepFM

```python
import torch
import torch.nn as nn

from damo_embedding import Embedding


class DeepFM(torch.nn.Module):
    def __init__(
        self,
        emb_size: int,
        fea_size: int,
        hid_dims=[256, 128],
        num_classes=1,
        dropout=[0.2, 0.2],
        **kwargs,
    ):
        super(DeepFM, self).__init__()
        self.emb_size = emb_size
        self.fea_size = fea_size

        initializer = {
            "name": "truncate_normal",
            "mean": float(kwargs.get("mean", 0.0)),
            "stddev": float(kwargs.get("stddev", 0.0001)),
        }

        optimizer = {
            "name": "adam",
            "gamma": float(kwargs.get("gamma", 0.001)),
            "beta1": float(kwargs.get("beta1", 0.9)),
            "beta2": float(kwargs.get("beta2", 0.999)),
            "lambda": float(kwargs.get("lambda", 0.0)),
            "epsilon": float(kwargs.get("epsilon", 1e-8)),
        }

        self.w = Embedding(
            1,
            initializer=initializer,
            optimizer=optimizer,
        )

        self.v = Embedding(
            self.emb_size,
            initializer=initializer,
            optimizer=optimizer,
        )
        self.w0 = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        self.dims = [fea_size * emb_size] + hid_dims

        self.layers = nn.ModuleList()
        for i in range(1, len(self.dims)):
            self.layers.append(nn.Linear(self.dims[i - 1], self.dims[i]))
            self.layers.append(nn.BatchNorm1d(self.dims[i]))
            self.layers.append(nn.BatchNorm1d(self.dims[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout[i - 1]))
        self.layers.append(nn.Linear(self.dims[-1], num_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            input (torch.Tensor): input tensor

        Returns:
            tensor.Tensor: deepfm forward values
        """
        assert input.shape[1] == self.fea_size
        w = self.w.forward(input)
        v = self.v.forward(input)
        square_of_sum = torch.pow(torch.sum(v, dim=1), 2)
        sum_of_square = torch.sum(v * v, dim=1)
        fm_out = (
            torch.sum((square_of_sum - sum_of_square)
                      * 0.5, dim=1, keepdim=True)
            + torch.sum(w, dim=1)
            + self.w0
        )

        dnn_out = torch.flatten(v, 1)
        for layer in self.layers:
            dnn_out = layer(dnn_out)
        out = fm_out + dnn_out
        out = self.sigmoid(out)
        return out

```

## Save Model

```python
from damo_embedding import save_model
model = DeepFM(8, 39)
save_model(model, "./", training=False)
```
# Document
[Doc Website](https://uopensail.github.io/damo-embedding/docs/Intro)