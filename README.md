![AlphaGenome header image](docs/_static/header.png)

# AlphaGenome Research

![Presubmit Checks](https://github.com/google-deepmind/alphagenome_research/actions/workflows/presubmit_checks.yml/badge.svg)

[**Model Weights**](#model-weights) | [**Installation**](#installation) |
[**Quick Start**](#quick-start) |
[**Documentation**](https://www.alphagenomedocs.com/) |
[**Community**](https://www.alphagenomecommunity.com) |
[**Terms of Use**](https://deepmind.google.com/science/alphagenome/model-terms)

AlphaGenome is a unified DNA sequence model designed to advance regulatory
variant-effect prediction and shed light on genome function. It analyzes DNA
sequences of up to 1 million base pairs to deliver predictions at single
base-pair resolution across diverse modalities, including gene expression,
splicing patterns, chromatin features, and contact maps.

This repository provides the following research code:

-   An implementation of the AlphaGenome model, written in
    [JAX](https://github.com/google/jax).
-   An implementation of the
    [AlphaGenome API](https://deepmind.google.com/science/alphagenome) with
    accompanying variant scorers.
-   A dataset loader for reading AlphaGenome training data from TFRecords.
-   Colab notebooks for
    [running](https://colab.research.google.com/github/google-deepmind/alphagenome_research/blob/main/colabs/quick_start.ipynb)
    the model and
    [analysing](https://colab.research.google.com/github/google-deepmind/alphagenome_research/blob/main/colabs/variant_eval_examples.ipynb)
    evaluations.

We strongly recommend using our
[AlphaGenome API](https://deepmind.google.com/science/alphagenome) to interact
with the model without needing specialized hardware.

## Installation

<!-- mdformat off(disable for [!TIP] format) -->

> [!TIP]
> We strongly recommend you create a
> [Python Virtual Environment](https://docs.python.org/3/tutorial/venv.html) to
> prevent conflicts with your system's Python environment.

<!-- mdformat on -->

To install, clone a local copy of this repository and run `pip install`:

```bash
$ git clone https://github.com/google-deepmind/alphagenome_research.git
$ pip install -e ./alphagenome_research
```

This will install any required dependencies, including this repository in
[development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).

### Model weights

To use our pre-trained model weights, you can download them from either:

-   [Kaggle](https://www.kaggle.com/models/google/alphagenome) or
-   [Hugging Face](https://huggingface.co/collections/google/alphagenome)

Both require accepting our non-commercial
[model terms](https://deepmind.google.com/science/alphagenome/model-terms).
Requests are processed immediately.

### Model requirements

In order to run the model, we recommend running with at least an
[NVIDIA H100 GPU](https://docs.cloud.google.com/compute/docs/gpus#h100-gpus).

#### GPU installation (NVIDIA CUDA)

For local GPU support with CUDA 12, install JAX with CUDA and cuDNN:

```bash
$ pip install jax[cuda12_local] nvidia-cudnn-cu12
```

For more details, see the
[JAX installation documentation](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu).

#### Running the notebooks locally

If you plan to run the Colab notebooks locally, you will also need:

```bash
$ pip install ipywidgets
```

For training, we recommend running on
[Tensor Processing Units (TPUs) v3](https://docs.cloud.google.com/tpu/docs/v3)
or higher.

## Quick start

The easiest way to interact with the AlphaGenome model is using the provided
[DNA Model class](/src/alphagenome_research/model/dna_model.py). This wraps the core model and provides a
more intuitive set of functions for creating predictions, scoring variants,
performing in silico mutagenesis (ISM) and more.

It also provides the following factory functions to create a model instance
using our pre-trained weights:

```python
from alphagenome_research.model import dna_model

# To download from Kaggle:
model = dna_model.create_from_kaggle('all_folds')

# or Hugging Face:
model = dna_model.create_from_huggingface('all_folds')
```

Here's an example of making a variant prediction using model weights downloaded
from Kaggle:

```python
from alphagenome.data import genome
from alphagenome.visualization import plot_components
from alphagenome_research.model import dna_model
import matplotlib.pyplot as plt

model = dna_model.create_from_kaggle('all_folds')

interval = genome.Interval(chromosome='chr22', start=35677410, end=36725986)
variant = genome.Variant(
    chromosome='chr22',
    position=36201698,
    reference_bases='A',
    alternate_bases='C',
)

outputs = model.predict_variant(
    interval=interval,
    variant=variant,
    ontology_terms=['UBERON:0001157'],
    requested_outputs=[dna_model.OutputType.RNA_SEQ],
)

plot_components.plot(
    [
        plot_components.OverlaidTracks(
            tdata={
                'REF': outputs.reference.rna_seq,
                'ALT': outputs.alternate.rna_seq,
            },
            colors={'REF': 'dimgrey', 'ALT': 'red'},
        ),
    ],
    interval=outputs.reference.rna_seq.interval.resize(2**15),
    # Annotate the location of the variant as a vertical line.
    annotations=[plot_components.VariantAnnotation([variant], alpha=0.8)],
)
plt.show()
```

For further examples, please see our
[quick-start](https://colab.research.google.com/github/google-deepmind/alphagenome_research/blob/main/colabs/quick_start.ipynb)
notebook.

## Citing AlphaGenome

If you use AlphaGenome in your research, please cite using:

<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->

```bibtex
@article{alphagenome,
  title={Advancing regulatory variant effect prediction with {AlphaGenome}},
  author={Avsec, {\v Z}iga and Latysheva, Natasha and Cheng, Jun and Novati, Guido and Taylor, Kyle R. and Ward, Tom and Bycroft, Clare and Nicolaisen, Lauren and Arvaniti, Eirini and Pan, Joshua and Thomas, Raina and Dutordoir, Vincent and Perino, Matteo and De, Soham and Karollus, Alexander and Gayoso, Adam and Sargeant, Toby and Mottram, Anne and Wong, Lai Hong and Drot{\'a}r, Pavol and Kosiorek, Adam and Senior, Andrew and Tanburn, Richard and Applebaum, Taylor and Basu, Souradeep and Hassabis, Demis and Kohli, Pushmeet},
  journal={Nature},
  volume={649},
  number={8099},
  year={2026},
  doi={10.1038/s41586-025-10014-0},
  publisher={Nature Publishing Group UK London}
}
```

<!-- enableFinding(SNIPPET_INVALID_LANGUAGE) -->

## Acknowledgements

AlphaGenome's model release uses the following libraries and packages:

*   [Abseil](https://github.com/abseil/abseil-py)
*   [anndata](https://github.com/scverse/anndata)
*   [Chex](https://github.com/google-deepmind/chex)
*   [Einshape](https://github.com/google-deepmind/einshape)
*   [Etils](https://github.com/google/etils)
*   [Haiku](https://github.com/google-deepmind/dm-haiku)
*   [huggingface_hub](https://github.com/huggingface/huggingface_hub)
*   [JAX](https://github.com/jax-ml/jax)
*   [jaxtyping](https://github.com/patrick-kidger/jaxtyping)
*   [kagglehub](https://github.com/Kaggle/kagglehub)
*   [NumPy](https://numpy.org/)
*   [Orbax](https://github.com/google/orbax)
*   [pandas](https://pandas.pydata.org/)
*   [pyarrow](https://arrow.apache.org/)
*   [pyfaidx](https://github.com/mdshw5/pyfaidx)
*   [PyRanges](https://github.com/pyranges/pyranges)
*   [TensorFlow](https://www.tensorflow.org/)
*   [typeguard](https://github.com/agronholm/typeguard)

We thank all their contributors and maintainers!

## License and Disclaimer

Copyright 2026 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you
may not use this except in compliance with the Apache 2.0 license. You may
obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0. As noted above, model weights are
available via Kaggle and Hugging Face and are subject to the model terms at:
https://deepmind.google.com/science/alphagenome/model-terms.

Code examples and documentation to help you use the AlphaGenome model are
licensed under the Creative Commons Attribution 4.0 International License
(CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode.

Unless set out below under the *Training Data*, *Evaluation Data* or *Training
and Evaluation Data* headings, all other materials are licensed under the
Creative Commons Attribution-NonCommercial 4.0 International License (CC-BY-NC).
You may obtain a copy of the CC-BY-NC license at:
https://creativecommons.org/licenses/by-nc/4.0/legalcode.

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.

### Training Data

**FANTOM5:** This data has been reprocessed. The original FANTOM5 data is made
available at https://fantom.gsc.riken.jp/5/ under a CC-BY license (see link
above for a copy). Citation: *Lizio M, et al. Update of the FANTOM web resource:
expansion to provide additional transcriptome atlases. Nucleic Acids Res. 47:
D752–D758 (2019). https://doi.org/10.1093/nar/gky1099.*

**4D Nucleome:** This data has been reprocessed using the method set out in the
‘Methods’ section of the accompanying paper. The original 4D Nucleome data is
available from the 4DN Data Portal at https://data.4dnucleome.org/ and subject
to the Data Use Guidelines found there. The 4DN Data Portal is part of 4DN,
citation 4DN White Paper (https://www.nature.com/articles/nature23884) and 4DN
Data Portal Paper (https://www.nature.com/articles/s41467-022-29697-4).

### Evaluation Data

**This data includes: (i) list of variants; (ii) target values; and (iii)
AlphaGenome predicted score.**

**CAGI:** CAGI data can be obtained from
genomeinterpretation.org/challenges.html and is subject to the terms found here:
http://www.genomeinterpretation.org/data-use-agreement.html.

**GTEx v8:** GTEx v8 data can be obtained from: gtexportal.org/home. The data
used for the work described in this paper was obtained from:
https://github.com/calico/borzoi. Please visit the GTEx Portal for the most up
to date and accurate version of this data.

**GTEx v8 reprocessed into EMBL-EBI eQTL catalogue:** Data originally made
available at the GTEx Portal (see above) with modifications made by EMBL-EBI,
and provided under a CC-BY-4.0 license a copy of which can be found at
https://creativecommons.org/licenses/by/4.0/legalcode. Citation: *Kerimov, N.,
Hayhurst, J.D., Peikova, K. et al. A compendium of uniformly processed human
gene expression and splicing quantitative trait loci. Nat Genet 53, 1290–1299
(2021). https://doi.org/10.1038/s41588-021-00924-w.*

**ChromBPNet:** ChromBPNet data can be obtained at
https://www.synapse.org/Synapse:syn59449898/files/. Citation: *Pampari, A. et
al. ChromBPNet: bias factorized, base-resolution deep learning models of
chromatin accessibility reveal cis-regulatory sequence syntax, transcription
factor footprints and regulatory variants. BioRxiv, 2024–12 (2025).*

**ClinVar:** ClinVar data can be found at:
https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/ subject to this Data Use
Policy https://www.ncbi.nlm.nih.gov/clinvar/docs/maintenance_use/. Citation:
*Landrum, M. J. et al. ClinVar: improving access to variant interpretations and
supporting evidence. Nucleic Acids Res. 2018 Jan 4;46(D1):D1062-D1067. doi:
10.1093/nar/gkx1153.*

**MFASS:** MFASS data can be found at https://github.com/KosuriLab/MFASS.
Citation: *A Multiplexed Assay for Exon Recognition Reveals that an
Unappreciated Fraction of Rare Genetic Variants Cause Large-Effect Splicing
Disruptions; Chong, Rockie et al.; Molecular Cell, Volume 73, Issue 1, 183 -
194.e8.*

**eQTL:** eQTL data is provided with a CC-BY-4.0 license a copy of which can be
found here: https://creativecommons.org/licenses/by/4.0/legalcode. Citation:
*Kerimov, N., Hayhurst, J.D., Peikova, K. et al. A compendium of uniformly
processed human gene expression and splicing quantitative trait loci. Nat Genet
53, 1290–1299 (2021). https://doi.org/10.1038/s41588-021-00924-w.*

**Open Targets:** Open Targets data can be obtained at
https://platform-docs.opentargets.org/licence and is provided with a Creative
Commons 1.0 Universal license, a copy of which can be found here:
https://creativecommons.org/publicdomain/zero/1.0/legalcode.

**PolyA site annotations:** PolyA site annotations can be obtained here:
https://exon.apps.wistar.org/polya_db/v3/. The data used for this project is a
reprocessed version which can be found:
https://storage.googleapis.com/seqnn-share/helper/polyadb_human_v3.csv.gz.
Citation: *Linder, J., Srivastava, D., Yuan, H. et al. Predicting RNA-seq
coverage from DNA sequence as a unifying model of gene regulation. Nat Genet 57,
949–961 (2025). https://doi.org/10.1038/s41588-024-02053-6.*

### Training & Evaluation Data

**ENCODE:** This data has been reprocessed. The original ENCODE data is made
available at https://www.encodeproject.org/help/getting-started/#download
pursuant to the Data Use Policy at
https://www.encodeproject.org/help/citing-encode/. The specific data can be
found cited in the Supplementary Tables published as part of the *Advancing
regulatory variant effect prediction with AlphaGenome* paper. The data is
presented by the ENCODE Consortium, whose most recent publications are:

-   ENCODE integrative analysis (PMID: 22955616; PMCID: PMC3439153)
-   ENCODE portal (PMID: 41168159; PMCID: PMC12575607; PMID: 31713622; PMCID:
    PMC7061942)
-   ENCODE uniform processing pipelines:
    https://www.biorxiv.org/content/10.1101/2023.04.04.535623.

**GENCODE:** Copyright of the released GENCODE dataset is © 2024 EMBL-EBI. A
modified version of the GENCODE dataset (which can be found here:
https://www.gencodegenes.org/human/releases.html), is made available with
reference to the following:

-   Copyright © 2024 EMBL-EBI
-   The GENCODE dataset is subject to the EMBL-EBI terms of use, available at
    https://www.ebi.ac.uk/about/terms-of-use.
-   Citation: Frankish A, et al (2018) GENCODE reference annotation for the
    human and mouse genome.
-   Further details about GENCODE can be found at
    https://www.gencodegenes.org/human/releases.html, with additional citation
    information at https://www.gencodegenes.org/pages/publications.html and
    further acknowledgements can be found at
    https://www.gencodegenes.org/pages/gencode.html.

To prepare the dataset, the team followed the method set out here:
https://github.com/google-deepmind/alphagenome/blob/main/scripts/process_gtf.py

### Third-party software

Your use of any third-party software, libraries or code referenced in the
materials in this repository (including the libraries listed in the
[Acknowledgments](#acknowledgements) section) may be governed by separate terms
and conditions or license provisions. Your use of the third-party software,
libraries or code is subject to any such terms and you should check that you can
comply with any applicable restrictions or terms and conditions before use.
