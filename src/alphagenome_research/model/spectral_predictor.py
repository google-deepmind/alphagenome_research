# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contact map predictor wrapper with spectral regularization integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from alphagenome_research.model.spectral_regularizer import (
    SpectralContactRegularizer,
)


@dataclass(frozen=True)
class SpectralRegularizationConfig:
  """Configuration for spectral regularization."""

  lambda_low: float = 0.05
  lambda_high: float = 0.1
  lambda_sym: float = 0.01
  spectral_operator: Literal["fft", "dct", "laplacian"] = "fft"
  low_freq_cutoff: float = 0.15
  high_freq_cutoff: float = 0.6
  adaptive: bool = False


class SpectralEnhancedContactPredictor(nn.Module):
  """
  Predictor de contactos con regularización espectral integrada.
  Se puede usar como reemplazo directo en AlphaGenome.
  """

  def __init__(
      self,
      base_predictor: nn.Module,
      config: SpectralRegularizationConfig | None = None,
      data_loss_fn: nn.Module | None = None,
  ) -> None:
    super().__init__()

    self.base = base_predictor
    self.config = config or SpectralRegularizationConfig()
    self.data_loss_fn = data_loss_fn or nn.MSELoss()

    if self.config.spectral_operator != "fft":
      raise NotImplementedError(
          "Only FFT-based spectral regularization is implemented."
      )

    self.spectral_reg = SpectralContactRegularizer(
        lambda_high=self.config.lambda_high,
        lambda_low=self.config.lambda_low,
        lambda_sym=self.config.lambda_sym,
        freq_threshold_high=self.config.high_freq_cutoff,
        freq_threshold_low=self.config.low_freq_cutoff,
    )

  def forward(
      self,
      sequence_embedding: torch.Tensor,
      targets: torch.Tensor | None = None,
      training: bool = True,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
    """
    Args:
        sequence_embedding: [batch, L, D] - Embedding de secuencia
        targets: [batch, L, L] - Ground truth, opcional
        training: Si True, calcula pérdidas de regularización

    Returns:
        contact_map: [batch, L, L] - Mapa de contacto predicho
        loss_dict: Pérdidas desglosadas (si training y targets provistos)
    """
    # 1. Predicción base
    contact_map = self.base(sequence_embedding)

    if not training or targets is None:
      return contact_map

    loss_data = self._data_loss(contact_map, targets)
    loss_spectral, spectral_metrics = self.spectral_reg(
        contact_map, contact_target=targets
    )
    loss_sym = spectral_metrics["symmetry_loss"]

    total_loss = (
        loss_data
        + self.config.lambda_low * spectral_metrics["low_freq_loss"]
        + self.config.lambda_high * spectral_metrics["high_freq_loss"]
        + self.config.lambda_sym * loss_sym
    )

    loss_dict = {
        "total": total_loss,
        "data": loss_data,
        "spectral_low": spectral_metrics["low_freq_loss"],
        "spectral_high": spectral_metrics["high_freq_loss"],
        "symmetry": loss_sym,
    }

    return contact_map, loss_dict

  def _data_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return self.data_loss_fn(pred, target)

  def predict_with_uncertainty(
      self, sequence_embedding: torch.Tensor, n_samples: int = 10
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Predicción con estimación de incertidumbre usando dropout.

    Returns:
        mean_pred: Predicción promedio
        uncertainty: Incertidumbre por pixel
    """
    self.train()  # Activar dropout

    predictions = []
    for _ in range(n_samples):
      with torch.no_grad():
        predictions.append(self.base(sequence_embedding))

    self.eval()

    predictions_stack = torch.stack(predictions)  # [n_samples, batch, L, L]
    mean_pred = predictions_stack.mean(dim=0)
    uncertainty = predictions_stack.std(dim=0)

    return mean_pred, uncertainty
