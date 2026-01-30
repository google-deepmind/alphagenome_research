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

"""Example experiment configuration for spectral regularization."""

from __future__ import annotations

from alphagenome_research.model.spectral_predictor import (
    SpectralEnhancedContactPredictor,
)

# NOTE: Replace these placeholders with real model definitions in your project.
# from your_project.models import SimpleContactPredictor
# base_predictor = SimpleContactPredictor(input_dim=256, hidden_dim=512)

# EXPERIMENTO 1: Validación básica del regularizador espectral

# 1. Cargar datos reales
train_files = [
    "data/GM12878/hic/GSE63525_GM12878_insitu_primary.hic",  # Convertir a .cool
    "data/K562/hic/ENCFF123ABC.cool",
]

# 2. Configurar modelo
# model = SpectralEnhancedContactPredictor(
#     base_predictor=base_predictor,
#     use_spectral_reg=True,
#     spectral_kwargs={
#         "lambda_high": 0.1,
#         "lambda_low": 0.05,
#         "lambda_sym": 0.01,
#     },
# )

# 3. Entrenamiento comparativo
experiments = {
    "baseline": "model_without_spectral",
    "spectral_low": "model_with_spectral_low_weight",
    "spectral_high": "model_with_spectral_high_weight",
    "spectral_adaptive": "model_with_adaptive_weights",
}

# 4. Métricas de evaluación
metrics = {
    "Pearson_correlation": "Correlación entre predicción y ground truth",
    "MSE": "Error cuadrático medio",
    "SSIM": "Structural Similarity Index",
    "TAD_boundary_F1": "Detección de bordes de TADs",
    "Loop_recall": "Recuperación de bucles conocidos",
    "HiCRep_similarity": "Similaridad estructural (HiCRep)",
    "Spectral_energy_ratio": "Proporción energía baja/alta frecuencia",
}
