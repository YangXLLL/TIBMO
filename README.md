# TIBMO
We present the Text-Image Bidirectional Mutual Optimization (TIBMO) attack. It iteratively optimizes image and text adversarially in both directions. To improve text quality, we replace BERT-MLM with GloVe embeddings for word candidate selection. Experiments show TIBMO achieves strong attack performance with natural text.

Quick Start
1. Install dependencies
See in requirements.txt.

2. Prepare datasets and models
- Dataset json files for downstream tasks [[ALBEF github]](https://github.com/salesforce/ALBEF)
- Finetuned checkpoint for ALBEF [[ALBEF github]](https://github.com/salesforce/ALBEF)
- Finetuned checkpoint for TCL [[TCL github]](https://github.com/uta-smile/TCL)
