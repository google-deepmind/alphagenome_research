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

"""Spectral regularization utilities for genomic contact maps."""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralContactRegularizer(nn.Module):
  """
  Regularización espectral para mapas de contacto genómicos.

  Principio: Penalizar ruido de alta frecuencia mientras se preserva
  estructura de baja frecuencia (dominios TADs, bucles).

  Basado en:
  - Fourier-domain regularization (Chiang et al. 2021)
  - Multi-scale genomic signals (Zhou et al. 2022)
  """

  def __init__(
      self,
      lambda_high: float = 0.1,
      lambda_low: float = 0.05,
      lambda_sym: float = 0.01,
      freq_threshold_high: float = 0.3,
      freq_threshold_low: float = 0.1,
  ) -> None:
    super().__init__()

    self.lambda_high = lambda_high
    self.lambda_low = lambda_low
    self.lambda_sym = lambda_sym

    # Estos se computan en forward (dependen del tamaño de entrada)
    self.freq_threshold_high = freq_threshold_high
    self.freq_threshold_low = freq_threshold_low

    # Cache para máscaras frecuenciales (optimización)
    self.register_buffer("_freq_masks", None)

  def _create_frequency_masks(self, length: int, device: torch.device) -> torch.Tensor:
    """Crea máscaras frecuenciales una vez por tamaño length."""
    if self._freq_masks is not None:
      if self._freq_masks.shape[-1] == length and self._freq_masks.device == device:
        return self._freq_masks

    # Espacio frecuencial 2D
    frequencies = torch.fft.fftfreq(length, device=device)
    kx, ky = torch.meshgrid(frequencies, frequencies, indexing="ij")

    # Frecuencia radial (distancia al centro)
    radial_freq = torch.sqrt(kx**2 + ky**2)
    radial_freq = torch.fft.fftshift(radial_freq)  # Centro en 0

    # Máscaras
    low_freq_mask = (radial_freq <= self.freq_threshold_low).float()
    high_freq_mask = (radial_freq >= self.freq_threshold_high).float()

    # Máscara banda media (opcional para análisis)
    mid_freq_mask = (
        (radial_freq > self.freq_threshold_low)
        & (radial_freq < self.freq_threshold_high)
    ).float()

    masks = torch.stack([low_freq_mask, mid_freq_mask, high_freq_mask])
    self._freq_masks = masks

    return masks

  def forward(
      self,
      contact_pred: torch.Tensor,
      contact_target: torch.Tensor | None = None,
      weight_mask: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        contact_pred: Tensor [batch, L, L] - Predicción del modelo
        contact_target: Tensor [batch, L, L] - Ground truth (opcional)
        weight_mask: Tensor [batch, L, L] - Máscara de pesos (ej: regiones válidas)

    Returns:
        loss: Tensor escalar - Pérdida de regularización
        metrics: Dict - Métricas de diagnóstico
    """
    _, length, _ = contact_pred.shape

    # 1. Asegurar que sea simétrico (Hi-C es simétrico)
    sym_loss = torch.tensor(0.0, device=contact_pred.device)
    if self.lambda_sym > 0:
      sym_diff = contact_pred - contact_pred.transpose(-1, -2)
      sym_loss = torch.mean(sym_diff**2)

    # 2. Transformada de Fourier 2D
    pred_fft = torch.fft.fft2(contact_pred.float())
    pred_fft = torch.fft.fftshift(pred_fft)  # Frecuencias centradas

    # 3. Obtener máscaras frecuenciales
    masks = self._create_frequency_masks(length, contact_pred.device)
    low_mask, _, high_mask = masks

    # 4. Pérdida de alta frecuencia (penalizar ruido o diferenciar con target)
    high_freq_loss = torch.tensor(0.0, device=contact_pred.device)
    if contact_target is None:
      high_freq_energy = torch.abs(pred_fft * high_mask.unsqueeze(0)) ** 2
      if weight_mask is not None:
        # Ponderar por importancia (ej: regiones con alta cobertura)
        high_freq_energy = high_freq_energy * weight_mask
      high_freq_loss = torch.mean(high_freq_energy)
    else:
      target_fft = torch.fft.fft2(contact_target.float())
      target_fft = torch.fft.fftshift(target_fft)

      high_freq_pred = pred_fft * high_mask.unsqueeze(0)
      high_freq_target = target_fft * high_mask.unsqueeze(0)

      high_freq_diff = torch.abs(high_freq_pred - high_freq_target) ** 2
      if weight_mask is not None:
        high_freq_diff = high_freq_diff * weight_mask
      high_freq_loss = torch.mean(high_freq_diff)

    # 5. Pérdida de baja frecuencia (preservar estructura)
    low_freq_loss = torch.tensor(0.0, device=contact_pred.device)
    if contact_target is not None and self.lambda_low > 0:
      target_fft = torch.fft.fft2(contact_target.float())
      target_fft = torch.fft.fftshift(target_fft)

      low_freq_pred = pred_fft * low_mask.unsqueeze(0)
      low_freq_target = target_fft * low_mask.unsqueeze(0)

      low_freq_diff = torch.abs(low_freq_pred - low_freq_target) ** 2

      if weight_mask is not None:
        low_freq_diff = low_freq_diff * weight_mask

      low_freq_loss = torch.mean(low_freq_diff)

    # 6. Pérdida total
    total_loss = (
        self.lambda_high * high_freq_loss
        + self.lambda_low * low_freq_loss
        + self.lambda_sym * sym_loss
    )

    # 7. Métricas de diagnóstico
    metrics = {
        "high_freq_loss": high_freq_loss,
        "low_freq_loss": low_freq_loss,
        "symmetry_loss": sym_loss,
        "total_spectral_loss": total_loss,
    }

    return total_loss, metrics

  def analyze_frequency_content(self, contact_map: torch.Tensor) -> dict[str, float]:
    """
    Analiza contenido frecuencial para diagnóstico.

    Returns:
        dict con distribución de energía por bandas frecuenciales
    """
    length = contact_map.shape[-1]

    # Transformada
    contact_fft = torch.fft.fft2(contact_map.float())
    contact_fft = torch.fft.fftshift(contact_fft)
    power_spectrum = torch.abs(contact_fft) ** 2

    # Máscaras
    masks = self._create_frequency_masks(length, contact_map.device)
    low_mask, mid_mask, high_mask = masks

    # Energía por banda
    total_energy = torch.sum(power_spectrum)
    low_energy = torch.sum(power_spectrum * low_mask.unsqueeze(0))
    mid_energy = torch.sum(power_spectrum * mid_mask.unsqueeze(0))
    high_energy = torch.sum(power_spectrum * high_mask.unsqueeze(0))

    return {
        "energy_low_freq": (low_energy / total_energy).item(),
        "energy_mid_freq": (mid_energy / total_energy).item(),
        "energy_high_freq": (high_energy / total_energy).item(),
        "total_energy": total_energy.item(),
    }


class MultiScaleSpectralRegularizer(nn.Module):
  """
  Regularización espectral multi-escala.

  Aplica regularización a diferentes resoluciones para capturar
  estructuras a diferentes escalas genómicas:
  - Escala fina: Bucles (~1-10kb)
  - Escala media: Dominios TADs (~50-500kb)
  - Escala gruesa: Compartimentos A/B (~1Mb+)
  """

  def __init__(self, scales: list[float] | None = None, **kwargs: float) -> None:
    super().__init__()

    if scales is None:
      scales = [1.0, 0.5, 0.25]

    self.scales = scales
    self.regularizers = nn.ModuleList(
        [SpectralContactRegularizer(**kwargs) for _ in scales]
    )

  def forward(
      self,
      contact_pred: torch.Tensor,
      contact_target: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses = []
    all_metrics: dict[str, torch.Tensor] = {}

    for i, scale in enumerate(self.scales):
      # Redimensionar a escala
      if scale != 1.0:
        length_orig = contact_pred.shape[-1]
        length_scaled = int(length_orig * scale)

        # Interpolación simple (podría mejorarse)
        pred_scaled = torch.nn.functional.interpolate(
            contact_pred.unsqueeze(1),
            size=(length_scaled, length_scaled),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        if contact_target is not None:
          target_scaled = torch.nn.functional.interpolate(
              contact_target.unsqueeze(1),
              size=(length_scaled, length_scaled),
              mode="bilinear",
              align_corners=False,
          ).squeeze(1)
        else:
          target_scaled = None
      else:
        pred_scaled = contact_pred
        target_scaled = contact_target

      # Aplicar regularizador a esta escala
      loss, metrics = self.regularizers[i](pred_scaled, target_scaled)

      losses.append(loss)

      # Renombrar métricas por escala
      for key, value in metrics.items():
        all_metrics[f"scale_{scale}_{key}"] = value

    # Promedio ponderado (pesos podrían aprenderse)
    total_loss = sum(losses) / len(losses)

    return total_loss, all_metrics
