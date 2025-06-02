## SVG Diffusion

This repository contains code for generating SVGs using diffusion models. The pipeline first generates images with a fine-tune of Stable Diffusion 3.5 Medium that has been trained with [DRaFT](https://arxiv.org/abs/2309.17400) to produce high-quality, SVG-style outputs. The images are then heuristically converted to rough SVGs, which are further refined using [diffvg](https://github.com/BachiLi/diffvg), a differentiable rasterizer.

This method was used to win [4th]((https://www.kaggle.com/competitions/drawing-with-llms/leaderboard)) place in the [Drawing with LLMs](https://www.kaggle.com/competitions/drawing-with-llms) kaggle competition. It was the only entrant in the top 5 that didn't "attack" the evaluation metrics: 1st and 2nd place figured out how to hide text in their SVGs while 3rd and 5th place found ways to directly optimize for the aesthetic score.  You can read more about the kaggle competition entry in [this discussion post](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581108).

To train a LoRA with DRaFT, run `draft.py.` The dataset of prompts used for training is available
[here](https://drive.google.com/file/d/12nMeHvJUWoEQBqJ9N0MLLWCtj0uRF0IU/view?usp=sharing). A Colab notebook for fine-tuning is available [here](https://colab.research.google.com/drive/16N2tihVUed5zPqRYWIpuoVZz_wJbaOLj?usp=sharing) (use an A100 runtime). The inference code is available on kaggle through [this package](https://www.kaggle.com/code/kevlark/4th-place-solution-sd3-5m-draft-diffvg).

Note: in addition to the packages in `requirements.txt`, you'll need to install `diffvg` (https://github.com/BachiLi/diffvg) separately.
