#!/bin/bash
config_file=configs/glip_Swin_T_O365_GoldG_isbi.yaml
odinw_configs=configs/glip_Swin_T_O365_GoldG_isbi.yaml
output_dir=OUTPUTS/isic/zero_shot/llm/top3
model_checkpoint=MODEL/isic2016_best.pth
jsonFile=autoprompt_json/llm_isic_path_prompt_top3.json

python test.py --json ${jsonFile} \
      --config-file ${config_file} --weight ${model_checkpoint} \
      --task_config ${odinw_configs} \
      OUTPUT_DIR ${output_dir}\
      TEST.IMS_PER_BATCH 2 SOLVER.IMS_PER_BATCH 2 \
      TEST.EVAL_TASK detection \
      DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding \
      DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE False \
      DATASETS.USE_OVERRIDE_CATEGORY True \
      DATASETS.USE_CAPTION_PROMPT True\
      # MODEL.RETINANET.DETECTIONS_PER_IMG 300 MODEL.FCOS.DETECTIONS_PER_IMG 300 MODEL.ATSS.DETECTIONS_PER_IMG 300 MODEL.ROI_HEADS.DETECTIONS_PER_IMG 300