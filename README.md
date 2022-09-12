<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/transformers/master/docs/source/imgs/transformers_logo_name.png" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://huggingface.co/transformers/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/master/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p><b>English</b> | <a href="https://github.com/huggingface/transformers/blob/master/README_zh-hans.md">简体中文</a><p>
</h4>

<h3 align="center">
    <p>State-of-the-art Natural Language Processing for Jax, PyTorch and TensorFlow</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://raw.githubusercontent.com/huggingface/transformers/master/docs/source/imgs/course_banner.png"></a>
</h3>

# Table transformer to CSV

NOTE: This fork also contains code for launching your streamlit application to convert PDF-page-images that contain tables, to pandas dataframe that can be visualized and saved.

> Credits: Creators of table-transformer (MSR) https://github.com/microsoft/table-transformer
and https://github.com/NielsRogge/ who Integrated table-transformer to HuggingFace.

> My contribution: Added code to extract bounding boxes from the TSR model, use pytesseract and project to pandas dataframe.

The OCR is done by pytesseract (on Windows 11), please make necessary changes to the image paths and model checkpoint paths.
The model checkpoint can be found on HuggingFace: https://huggingface.co/nielsr/detr-table-detection and https://huggingface.co/nielsr/detr-table-structure-recognition


## Dashboard overview:
![image](https://user-images.githubusercontent.com/31631107/189661386-2a173cf4-9d0a-4c3b-9877-4f6439ed119d.png)
## Specific example:
### Table detection
![image](https://user-images.githubusercontent.com/31631107/189661619-1bac28bd-1d56-4989-81e6-726e93409194.png)
### Table Structure Recognition
![image](https://user-images.githubusercontent.com/31631107/189661745-32bb808d-e1a0-4549-bf41-2494d010ace2.png)
### Pandas Dataframe (can be exported to CSV)
![image](https://user-images.githubusercontent.com/31631107/189662133-272ce876-8efa-44de-bb43-8f900b3ee0bc.png)

