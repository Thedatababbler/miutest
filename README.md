# Medical Image Understanding with Pretrained Vision Language Models: A Comprehensive Study
<!-- select Model and/or Data and/or Code as needed>
Welcome to OpenMEDLab! 👋

<!--
**Here are some ideas to get you started:**
🙋‍♀️ A short introduction - what is your organization all about?
🌈 Contribution guidelines - how can the community get involved?
👩‍💻 Useful resources - where can the community find your docs? Is there anything else the community should know?
🍿 Fun facts - what does your team eat for breakfast?
🧙 Remember, you can do mighty things with the power of [Markdown](https://docs.github.com/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
-->


<!-- Insert the project banner here -->
<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="https://github.com/openmedlab/sampleProject/blob/main/banner_sample.png"></a>
</div>

---

<!-- Select some of the point info, feel free to delete -->

[![PyPI](https://img.shields.io/pypi/v/DI-engine)](https://pypi.org/project/DI-engine/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/DI-engine)
![PyTorch Version](https://img.shields.io/badge/dynamic/json?color=blue&label=pytorch&query=%24.pytorchVersion&url=https%3A%2F%2Fgist.githubusercontent.com/PaParaZz1/54c5c44eeb94734e276b2ed5770eba8d/raw/85b94a54933a9369f8843cc2cea3546152a75661/badges.json)


![Loc](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/HansBug/3690cccd811e4c5f771075c2f785c7bb/raw/loc.json)
![Comments](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/HansBug/3690cccd811e4c5f771075c2f785c7bb/raw/comments.json)


<!-- ![GitHub Org's stars](https://img.shields.io/github/stars/opendilab)
[![GitHub stars](https://img.shields.io/github/stars/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/network) -->
<!-- ![GitHub commit activity](https://img.shields.io/github/commit-activity/m/opendilab/DI-engine)
[![GitHub issues](https://img.shields.io/github/issues/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/pulls)
[![Contributors](https://img.shields.io/github/contributors/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/blob/master/LICENSE) -->

Updated on 2023.06.08



## Key Features

This is a repository for the ICLR2023 accepted paper -- Medical Image Understanding with Pretrained Vision Language Models: A Comprehensive Study.

key feature bulletin points here
- 1
- 2
- 3

## Links

- [Paper](https://)
- [Model](https://)
- [Code](https://) 
<!-- [Code] may link to your project at your institute>


<!-- give a introduction of your project -->
## Details

intro text here.

<!-- Insert a pipeline of your algorithm here if got one -->
<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="https://github.com/openmedlab/sampleProject/blob/main/diagram_sample.png"></a>
</div>

More intro text here.


## Dataset

Due to the license factor, we can not share all the datasets we used in our work, but we upload the polyp benchmark datasets as sample. If someone wants to use their own dataset, please refer to the polyp datasets to organize their data paths and annotation files. 

|Netdisk Type|Link|Password(optional)|
| ------ | ------ | ------ |
|BaiduNetDisk| [link](https://pan.baidu.com/s/1E7gbvs3ljXUsyy4yvZQT-A?pwd=s2nf )|s2nf |
|Google Drive|[link](https://drive.google.com/file/d/10ISx1yXxfE20nKq6UqquUAD5Egk3hyqi/view?usp=sharing)| N/A

After you download this zip file, please unzip it and place the folder at the project path.


## Get Started

**Main Requirements**  
Our project is based on the GLIP project, so please first setup the environment for the GLIP model following [this instruction](https://github.com/microsoft/GLIP#introduction). Next, please clone this repository and continue the installation guide in the next section.

**Installation**
```bash
git clone ---
pip install -r requirements.txt
```

#### Configuration Files
We follow the config file format used in the GLIP project. Please refer to the [sample config file]() we provided to create your own config file. **Note: The DATASETS.CAPTION_PROMPT content is ignore by our code, as our code use the automatically generated code instead of user inputted prompt.** 


#### Zero-shot Inference set-up guide
**Generate prompts with Masked Language Model(MLM) method**
In our work, we proposed three different methods to automatically generate prompts with expressive attributes. The first approach is the MLM method. To generate prompts with this approach, we need to use the pre-trained Language Models as our knowledge source. In this project, we use the [BiomedNLP-PubmedBERT-base](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) model as our specialized language model. Please use the following code to download the model to this repo:
```bash
git lfs install
git clone https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
```

After you setup the depedencies we need for automatically generate prompts with MLM method, you can now generate prompts for your dataset. However, our code currently only support generate prompts with three expressive attributes -- color, shape, and location. We may improve our code in the future to support more kinds of attributes, but we found the three attributes we included are the most useful attributes for now.

Now, run the following codes to get the prompts generated with our MLM method.
```bash
bash RUN/autoprompt/make_auto_mlm.sh
```
or
```bash
python make_autopromptsv2.py --dataset 'kvasir' \
      --cls_names 'polyp' \
      --vqa_names 'wound'\
      --mode 'lama'\
      --real_cls_names 'bump'
```
where ``--dataset`` is the dataset name that will be used for searching related data paths. ``--cls_names`` indicates the class name in the template that used to extract the attributes information from the language model. For example, in this case, we will ask the LM to predict the masked word in the following template:"The typical color of **polyp** is [MASK] color ". Then the LM will predict the MASK token considering the given class name. ``--vqa_names`` is similar to the ``cls_names`` above, execept it is used for asking the VQA models later. ``--mode`` argument decide which approach of automated generation will be used, and 'lama' refers to the MLM method. Finally, ``real_cls_names`` is the real class name that you will put into the prompt. Sometimes, we found substitude the terminologies with general vocabularies may imrpove the performance. For example, we use bump, instead of polyp, in our final prompts, and we observe a significant improvement.

After running the command before, you will receive several .json files saved in the 'autoprompt_json/' folder. These json files stored all the generated prompts for each image input. To run the final inferece code, please type the following codes:
```bash
#!/bin/bash
config_file=path/to/config/file.yaml
odinw_configs=path/to/config/file.yaml
output_dir=output/path
model_checkpoint=MODEL/glip_tiny_model_o365_goldg.pth
jsonFile=autoprompt_json/lama_kvasir_path_prompt_top1.json

python test_vqa.py --json ${jsonFile} \
      --config-file ${config_file} --weight ${model_checkpoint} \
      --task_config ${odinw_configs} \
      OUTPUT_DIR ${output_dir}\
      TEST.IMS_PER_BATCH 2 SOLVER.IMS_PER_BATCH 2 \
      TEST.EVAL_TASK detection \
      DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding \
      DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE False \
      DATASETS.USE_OVERRIDE_CATEGORY True \
      DATASETS.USE_CAPTION_PROMPT True\
```

**Generate image-specific prompts with VQA and Hybrid method**
Our approach need to use the OFA model for Visual-question answering tasks, and thus you need to follow this [guide](https://github.com/OFA-Sys/OFA/blob/main/transformers.md) to intall the OFA module with huggingface transformers. Note: We use the **OFA-base** model in this project. 
**For you convenience, you can simply run the following code to install the OFA model with huggingface transformers.** But we recommend you to refer to the user [guide](https://github.com/OFA-Sys/OFA/blob/main/transformers.md) in case there is any problem.
```bash
git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git
pip install OFA/transformers/
git clone https://huggingface.co/OFA-Sys/OFA-base 
```

**Download Model**


**Preprocess**
```bash
python DDD
```


**Training**
```bash
python DDD
```


**Validation**
```bash
python DDD
```


**Testing**
```bash
python DDD
```

## 🙋‍♀️ Feedback and Contact

- Email
- Webpage 
- Social media


## 🛡️ License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgement

A lot of code is modified from [monai](https://github.com/Project-MONAI/MONAI).

## 📝 Citation

If you find this repository useful, please consider citing this paper:
```
@article{Qin2022MedicalIU,
  title={Medical Image Understanding with Pretrained Vision Language Models: A Comprehensive Study},
  author={Ziyuan Qin and Huahui Yi and Qicheng Lao and Kang Li},
  journal={ArXiv},
  year={2022},
  volume={abs/2209.15517}
}
```

