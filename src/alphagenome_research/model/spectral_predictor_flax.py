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

"""Flax spectral predictor wrapper for Hi-C contact map models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from flax import linen as nn
import jax.numpy as jnp


@dataclass(frozen=True)
class SpectralRegularizationConfig:
  """Configuration for spectral regularization (Flax)."""

  lambda_low: float = 0.05
  lambda_high: float = 0.1
  lambda_sym: float = 0.01
  spectral_operator: Literal["fft"] = "fft"
  low_freq_cutoff: float = 0.15
  high_freq_cutoff: float = 0.6
  adaptive: bool = False


def fft2(x: jnp.ndarray) -> jnp.ndarray:
  """2D FFT over the last two axes."""
  return jnp.fft.fft2(x, axes=(-2, -1))


def split_frequencies(
    spectrum: jnp.ndarray, low_cut: float, high_cut: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Split spectrum into low- and high-frequency radial bands."""
  height, width = spectrum.shape[-2:]
  yy, xx = jnp.meshgrid(
      jnp.arange(height) - height / 2,
      jnp.arange(width) - width / 2,
      indexing="ij",
  )
  radius = jnp.sqrt(xx**2 + yy**2)
  r_norm = radius / jnp.max(radius)

  low_mask = r_norm <= low_cut
  high_mask = r_norm >= high_cut

  low = spectrum * low_mask
  high = spectrum * high_mask

  return low, high


def l2_loss(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
  return jnp.mean((a - b) ** 2)


def symmetry_loss(x: jnp.ndarray) -> jnp.ndarray:
  return jnp.mean((x - jnp.swapaxes(x, -1, -2)) ** 2)


class SpectralEnhancedContactPredictor(nn.Module):
  """
  Flax wrapper that adds spectral regularization to a base Hi-C predictor.

  The base predictor must return a contact map of shape: (B, N, N).
  """

  base_predictor: nn.Module
  spectral_config: SpectralRegularizationConfig

  @nn.compact
  def __call__(
      self,
      inputs: Any,
      targets: jnp.ndarray | None = None,
      *,
      training: bool = True,
      return_losses: bool = False,
  ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray] | None] | jnp.ndarray:
    if self.spectral_config.spectral_operator != "fft":
      raise ValueError("Only FFT-based spectral regularization is supported.")

    preds = self.base_predictor(inputs, training=training)

    if (not training) or (targets is None):
      return preds if not return_losses else (preds, None)

    loss_data = l2_loss(preds, targets)

    spec_pred = fft2(preds)
    spec_true = fft2(targets)

    low_pred, high_pred = split_frequencies(
        spec_pred,
        self.spectral_config.low_freq_cutoff,
        self.spectral_config.high_freq_cutoff,
    )
    low_true, high_true = split_frequencies(
        spec_true,
        self.spectral_config.low_freq_cutoff,
        self.spectral_config.high_freq_cutoff,
    )

    loss_low = l2_loss(jnp.abs(low_pred), jnp.abs(low_true))
    loss_high = l2_loss(jnp.abs(high_pred), jnp.abs(high_true))
    loss_sym = symmetry_loss(preds)

    total_loss = (
        loss_data
        + self.spectral_config.lambda_low * loss_low
        + self.spectral_config.lambda_high * loss_high
        + self.spectral_config.lambda_sym * loss_sym
    )

    losses = {
        "total": total_loss,
        "data": loss_data,
        "spectral_low": loss_low,
        "spectral_high": loss_high,
        "symmetry": loss_sym,
    }

    return preds, losses
