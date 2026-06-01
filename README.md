# BUA Labs

> Sample code and notebooks from [bualabs.com](https://www.bualabs.com/) — a Thai AI/ML learning community

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Notebooks](https://img.shields.io/badge/notebooks-130%2B-blue)

---

## About

BUA Labs is the companion code repository for [bualabs.com](https://www.bualabs.com/), a Thai-language deep learning and AI tutorial site. It contains 130+ Jupyter notebooks covering PyTorch, fastai, TensorFlow, NLP, geospatial analysis, ChatGPT prompting, and more — together with TensorFlow.js browser demos and Android mobile app examples.

---

## Repository Layout

```
bualabs/
├── nbs/          # Jupyter notebooks (main content)
├── web/          # TensorFlow.js browser demos (HTML)
└── app_android/  # Android mobile app projects
```

---

## Notebooks

Notebooks are in the `nbs/` directory and grouped by series number.

| Series | Topic | Notebooks |
|--------|-------|-----------|
| 00 | Getting Started | [Word Cloud (Thai)](nbs/00c_word_cloud_thai.ipynb) · [Play Audio](nbs/00b_play_horse_audio.ipynb) · [Jupyter Quickstart](nbs/00d_jupyter_notebook_quick_start.ipynb) |
| 01 | Image Classification | [Pets ResNet34](nbs/01a-image-classification-pets-resnet34.ipynb) · [Pets ResNet50](nbs/01b-image-classification-pets-resnet50.ipynb) · [MNIST ResNet18](nbs/01c-image-classification-mnist-resnet18.ipynb) · [Custom ResNet50](nbs/01d-image-classification-custom-resnet50.ipynb) · [Multilabel](nbs/01e-multilabel-image-classification-resnet50.ipynb) · [Heatmap / GradCAM](nbs/01f-heatmap-gradcam.ipynb) · [Histopathologic Cancer](nbs/01g_histopathologic_cancer_detection.ipynb) · [MixUp + Label Smoothing](nbs/01h-image-classification-pets-mixup-labelsmoothing.ipynb) · [Pets fastai2](nbs/01i_image_classification_pets_fastai2.ipynb) · [Skin Cancer HAM10000](nbs/01j_image_classification_skin_cancer_mnist_HAM10000.ipynb) · [Chest X-Ray Pneumonia](nbs/01k_chest_xray_images_pneumonia_classification.ipynb) |
| 02 | Training Fundamentals | [Learning Rate & Epochs](nbs/02a-learning-rate-epoch.ipynb) · [Batch Size](nbs/02b-batch-size.ipynb) |
| 03 | Image Segmentation | [CamVid U-Net ResNet34](nbs/03a-image-segmentation-camvid-unet-resnet34.ipynb) · [Pneumothorax](nbs/03b-image-segmentation-siim-acr-pneumothorax-segmentation.ipynb) |
| 04 | Sentiment Analysis | [IMDB AWD-LSTM](nbs/04a-sentiment-analysis-imdb-awd-lstm.ipynb) |
| 05 | Tabular / Structured Data | [Adult NN](nbs/05a-tabular-adult-nn.ipynb) · [Titanic Feature Engineering](nbs/05b-tabular-titanic-feature-engineering.ipynb) · [Rossmann Time Series](nbs/05c-tabular-rossmann-timeseries.ipynb) · [Customer Segmentation K-Means](nbs/05d_customer_segmentation_k_means.ipynb) |
| 06 | Collaborative Filtering | [MovieLens 100k](nbs/06a-collaborative-filtering-movielens100k.ipynb) |
| 07 | TensorBoard | [MNIST](nbs/07a-tensorboard-mnist.ipynb) · [Embedding](nbs/07b-tensorboard-embedding.ipynb) · [Embedding LearnerTensorboardWriter](nbs/07c-tensorboard-embedding-LearnerTensorboardWriter.ipynb) |
| 08 | Regression | [BIWI Head Pose ResNet34](nbs/08a-regression-biwi-headpose-resnet34.ipynb) |
| 09 | Regularization | [Data Augmentation](nbs/09a-data-augmentation-pets.ipynb) · [Dropout](nbs/09b-dropout-pets.ipynb) |
| 10 | Activation Functions | [Sigmoid](nbs/10a-activation-function-sigmoid.ipynb) · [Tanh](nbs/10b-activation-function-tanh.ipynb) · [ReLU](nbs/10c-activation-function-relu.ipynb) · [Mish](nbs/10d_mish_activation_function_nn.ipynb) |
| 11 | Hardware | [CPU vs GPU](nbs/11a-cpu-vs-gpu-train-deep-learning.ipynb) · [Webcam on Colab](nbs/11b_webcam_colab.ipynb) |
| 14 | Optimizers | [SGD / Linear Regression](nbs/14a-optimizer-sgd-linear-regression.ipynb) |
| 15 | Tensors | [Matrix Multiplication](nbs/15a-matrix-multiplication-tensor.ipynb) · [Numerical Operations](nbs/15b-numerical-operations-tensor.ipynb) · [Dimensions](nbs/15c-tensor-dimension.ipynb) · [Gather](nbs/15d_tensor_gather.ipynb) |
| 16 | Neural Networks from Scratch | [Basic NN](nbs/16a-neural-networks-basic.ipynb) · [Initialization](nbs/16b-neural-networks-initialization.ipynb) · [Vanishing / Exploding Gradient](nbs/16c-vanishing-gradient-exploding-gradient.ipynb) · [Training Loop](nbs/16d-neural-network-training-loop.ipynb) · [Parameters & Optimizer](nbs/16e-nn-parameter-optimizer.ipynb) · [Validation & Metrics](nbs/16f-nn-validation-metrics.ipynb) · [DataBunch & Learner](nbs/16g-databunch-learner.ipynb) · [CallbackHandler](nbs/16h-callbackhandler.ipynb) · [Callback Sample](nbs/16i-callback-sample.ipynb) · [LR Finder](nbs/16j-callback-lr_finder.ipynb) · [Fit-One-Cycle](nbs/16k-fit-one-cycle.ipynb) |
| 17 | Loss Functions | [MAE & RMSE](nbs/17a-loss-function-mae-rmse.ipynb) · [Cross-Entropy](nbs/17b-loss-function-cross-entropy.ipynb) |
| 18 | Softmax | [Softmax](nbs/18a-softmax.ipynb) |
| 19 | Metrics | [Confusion Matrix, Accuracy, Precision, Recall](nbs/19a-metrics-confusion-matrix-accuracy-precision-recall.ipynb) |
| 20 | Data APIs | [Dataset & DataLoader](nbs/20a-dataset-dataloader.ipynb) · [Sampler & Collate](nbs/20b-data-sampler-collate.ipynb) · [DataBlock API](nbs/20c-datablock-api.ipynb) · [DataBlock fastai2](nbs/20d_datablock_fastai2.ipynb) · [Imagenette fastai2](nbs/20e_imagenette_datablock_fastai2.ipynb) · [Pets fastai2](nbs/20f_pets_datablock_fastai2.ipynb) · [Siamese fastai2](nbs/20g_siamese_fastai2.ipynb) · [Echoing Transform](nbs/20h_echoing_transform.ipynb) |
| 21 | Preprocessing | [Fill NA](nbs/21a-preprocessing-fillna.ipynb) · [Normalization](nbs/21b-preprocessing-normalization.ipynb) · [Categorization](nbs/21c-preprocessing-categorization.ipynb) · [Cat Other](nbs/21d-preprocessing-cat-other.ipynb) |
| 22 | Python Fundamentals | [Callbacks](nbs/22a-python-callback.ipynb) · [Lambda](nbs/22b-python-lambda.ipynb) · [Partial](nbs/22c-python-partial.ipynb) · [*args / **kwargs](nbs/22d-python-args-kwargs.ipynb) |
| 23 | EDA | [Pandas Profiling](nbs/23a-pandas-profiling.ipynb) · [Pandas UI](nbs/23b_eda_pandas_ui.ipynb) |
| 24 | CNN Internals | [Lambda Layer](nbs/24a-cnn-lambda.ipynb) · [Hooks](nbs/24b-cnn-hook.ipynb) · [Histograms](nbs/24c-cnn-hist.ipynb) · [BatchNorm](nbs/24d-cnn-batchnorm.ipynb) · [LSUV Init](nbs/24e-init-lsuv.ipynb) · [Model Summary](nbs/24f-model-summary.ipynb) |
| 25 | Geospatial | [Intro](nbs/25a-geospatial-intro.ipynb) · [CRS](nbs/25b-geodata-crs.ipynb) · [Interactive Map (Folium)](nbs/25c-interactive-map-folium.ipynb) · [Folium Intro](nbs/25d-folium-intro.ipynb) · [Geocode & Spatial Join](nbs/25e-geocode-spatial-join.ipynb) · [Starbucks Big Data](nbs/25f-starbucks-reserve-roastery-big-data.ipynb) · [Proximity Analysis](nbs/25g-proximity-analysis.ipynb) · [Proximity NYC](nbs/25h-proximity-analysis-nyc-hospital-crash-collision.ipynb) |
| 26 | NLP | [Stop Words](nbs/26a_stop_words.ipynb) · [Stemming & Lemmatization](nbs/26b_stemming_lemmatization.ipynb) · [LSA / SVD / NMF](nbs/26c_lsa_svd_nmf.ipynb) · [Naive Bayes & LogReg](nbs/26d_sentiment_classification_using_naivebayes_logreg.ipynb) · [N-grams / Trigram](nbs/26e_sentiment_classification_ngrams_trigram.ipynb) · [Regex](nbs/26f_regex.ipynb) · [ULMFiT IMDB](nbs/26g_ulmfit_imdb.ipynb) · [Numbers RNN](nbs/26h_numbers_english_rnn.ipynb) · [Seq2Seq Translation](nbs/26i_seq2seq_translation.ipynb) · [Seq2Seq Attention](nbs/26j_seq2seq_attention.ipynb) · [Transformer](nbs/26k_translation_transformer.ipynb) · [GPT-2 fastai2](nbs/26l_transformers_gpt2_fastai2.ipynb) · [WikiText fastai2](nbs/26m_wikitext_fastai2.ipynb) · [Thai Wikipedia Text Gen](nbs/26n_thwiki_text_generation_fastai2.ipynb) |
| 27 | Thai NLP (PyThaiNLP) | [Characters / Date / Number](nbs/27a_pythainlp_thai_character_date_number_money.ipynb) · [Tokenize](nbs/27b_pythainlp_tokenize.ipynb) · [Spell Checking](nbs/27c_pythainlp_spellchecking.ipynb) · [Tagging](nbs/27d_pythainlp_tagging.ipynb) |
| 28 | Export to JavaScript | [Linear → JS](nbs/28a_linear_to_javascript.ipynb) · [Keras → TF.js JSON](nbs/28b_tfjs_keras_to_json.ipynb) |
| 29 | TFLite / Mobile | [Linear Regression](nbs/29a_tflite_linear_regression.ipynb) · [Transfer Learning](nbs/29b_tflite_transfer_learning.ipynb) · [Fashion MNIST](nbs/29c_tflite_fashion_mnist.ipynb) · [RPS](nbs/29d_tflite_convert_rps.ipynb) |
| 30 | Medical Imaging | [DICOM / Pneumothorax](nbs/30a_dicom_medical_imaging_fastai2_pneumothorax_classification.ipynb) |
| 31 | ChatGPT Prompting | [Intro](nbs/31a_chatgpt_intro.ipynb) · [Delimited](nbs/31b_chatgpt_prompt_delimited.ipynb) · [Format Output](nbs/31c_chatgpt_prompt_format_output.ipynb) · [Check Precondition](nbs/31d_chatgpt_prompt_check_precondition.ipynb) · [Few-Shot](nbs/31e_chatgpt_prompt_few_shot_prompting.ipynb) · [Zero-Shot CoT](nbs/31f_chatgpt_prompt_zero_shot_cot_prompting.ipynb) |
| 32 | LangChain | [Intro](nbs/32a_langchain_intro.ipynb) · [Memory](nbs/32b_langchain_memory.ipynb) |
| 33 | Botnoi Voice | [TTS API](nbs/33a_botnoi_voice.ipynb) |
| 34 | CUDA Programming | [Intro (PMPP)](nbs/34a_cuda_pmpp.ipynb) |

---

## Web Demos (TensorFlow.js)

Interactive demos in the `web/` directory — open any `.html` file directly in a browser, or serve locally with `python -m http.server`.

| File | Description |
|------|-------------|
| [10a_tfjs_linear_reg.html](web/10a_tfjs_linear_reg.html) | Linear regression in the browser |
| [10b_iris_classifier.html](web/10b_iris_classifier.html) | Iris flower classifier |
| [10c_wdbc_classifier.html](web/10c_wdbc_classifier.html) | Wisconsin Diagnostic Breast Cancer classifier |
| [10d_mnist_classifier.html](web/10d_mnist_classifier.html) | MNIST handwritten digit classifier |
| [10e_fashion_mnist.html](web/10e_fashion_mnist.html) | Fashion MNIST classifier |
| [10f_toxicity.html](web/10f_toxicity.html) | Text toxicity detection |
| [10g_mobilenet.html](web/10g_mobilenet.html) | MobileNet image classification |
| [10h_linear_converted_python.html](web/10h_linear_converted_python.html) | Python-trained linear model exported to TF.js |
| [10i_object-detection-coco-ssd.html](web/10i_object-detection-coco-ssd.html) | Real-time object detection with COCO-SSD |
| [10j_retrain.html](web/10j_retrain.html) | Transfer learning / retraining in the browser |

---

## Android Apps

Projects in the `app_android/` directory — open in Android Studio.

| Project | Description |
|---------|-------------|
| [cats_vs_dogs](app_android/cats_vs_dogs) | Binary image classifier: cats vs. dogs |
| [image_classification](app_android/image_classification) | General image classification |
| [object_detection](app_android/object_detection) | On-device object detection |
| [rps_classification](app_android/rps_classification) | Rock-Paper-Scissors gesture classifier |

---

## Getting Started

### Prerequisites

- Python 3.8+
- [Jupyter Notebook](https://jupyter.org/) or [JupyterLab](https://jupyterlab.readthedocs.io/)
- [PyTorch](https://pytorch.org/) + [fastai](https://docs.fast.ai/) (for series 01–30)
- [TensorFlow](https://www.tensorflow.org/) (for series 28–30 and web demos)

### Clone the repository

```bash
git clone https://github.com/gnoparus/bualabs.git
cd bualabs
```

### Run notebooks locally

```bash
jupyter notebook nbs/
```

### Run notebooks on Google Colab

Open any notebook on GitHub and replace `github.com` with `githubtocolab.com` in the URL, or use the badge below each notebook on [bualabs.com](https://www.bualabs.com/).

### Serve web demos locally

```bash
cd web
python -m http.server 8080
# then open http://localhost:8080 in your browser
```

### Open Android apps

Open any project folder under `app_android/` in [Android Studio](https://developer.android.com/studio).

---

## License

This project is licensed under the [MIT License](LICENSE).
