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
    SpectralRegularizationConfig,
)

# NOTE: Replace these placeholders with real model definitions in your project.
# from your_project.models import SimpleContactPredictor
# base_predictor = SimpleContactPredictor(input_dim=256, hidden_dim=512)

# EXPERIMENT 1: Basic validation of the spectral regularizer

# 1. Load real data
train_files = [
    "data/GM12878/hic/GSE63525_GM12878_insitu_primary.hic",  # Convert to .cool
    "data/K562/hic/ENCFF123ABC.cool",
]

# 2. Configure model
# spectral_config = SpectralRegularizationConfig(
#     lambda_high=0.1,
#     lambda_low=0.05,
#     lambda_sym=0.01,
#     low_freq_cutoff=0.15,
#     high_freq_cutoff=0.6,
# )
# model = SpectralEnhancedContactPredictor(
#     base_predictor=base_predictor,
#     config=spectral_config,
# )

# 3. Comparative training
experiments = {
    "baseline": "model_without_spectral",
    "spectral_low": "model_with_spectral_low_weight",
    "spectral_high": "model_with_spectral_high_weight",
    "spectral_adaptive": "model_with_adaptive_weights",
}

# 4. Evaluation metrics
metrics = {
    "Pearson_correlation": "Correlation between prediction and ground truth",
    "MSE": "Mean Squared Error",
    "SSIM": "Structural Similarity Index",
    "TAD_boundary_F1": "TAD boundary detection",
    "Loop_recall": "Recall of known loops",
    "HiCRep_similarity": "Structural similarity (HiCRep)",
    "Spectral_energy_ratio": "Low/high frequency energy ratio",
}
