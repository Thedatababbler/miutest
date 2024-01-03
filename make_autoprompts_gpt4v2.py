import argparse
import torch
import os
from vqa_dataloader import make_dataloader
from PIL import Image
from torchvision import transforms
# from transformers import OFATokenizer, OFAModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
# from OFA.transformers.src.transformers.models.ofa.generate import sequence_generator
from collections import defaultdict
from tqdm import tqdm
import json
from langchain.llms import OpenAI, HuggingFaceHub,HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

import os

os.environ["OPENAI_API_KEY"] = "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"

parser = argparse.ArgumentParser()
#parser.add_argument('--cls_names', type=str, default='polyp',required=True)
parser.add_argument('--topk', type=int, default=3)
parser.add_argument('--dataset', type=str, default='dfuc', required=True)
parser.add_argument('--bert_type', type=str, default='pubmed')
parser.add_argument('--ofa_type', type=str, default='base')
parser.add_argument('--mode', type=str, default='hybrid', help='if both will generate lama and vqa and hybird')
parser.add_argument('--cls_names', action='append', required=True)
parser.add_argument('--real_cls_names', action='append', required=True)
parser.add_argument('--vqa_names', action='append', required=True)

args = parser.parse_args()

bert_map = {
    'pubmed': "./BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    'bert': None,
}
bert_path = bert_map[args.bert_type]

ofa_map ={
    'base': 'ofa-base/',
}
ofa_path = ofa_map[args.ofa_type]

DATASETMAP = {
    'dfuc': {'data_path': 'DATA/DFUC/images/dfuc2020_val', 'anno_path': 'DATA/DFUC/annotations/dfuc2020_val.json'},
    'isbi': {'data_path':'DATA/ISBI2016/images/test', 'anno_path':'DATA/ISBI2016/annotations/test.json'},
    'cvc300': {'data_path':'DATA/POLYP/test/CVC-300/images', 'anno_path':'DATA/POLYP/annotations/CVC-300_test.json'},
    'colondb': {'data_path':'DATA/POLYP/test/CVC-ColonDB/images', 'anno_path':'DATA/POLYP/annotations/CVC-ColonDB_test.json'},
    'clinicdb': {'data_path':'DATA/POLYP/test/CVC-ClinicDB/images', 'anno_path':'DATA/POLYP/annotations/CVC-ClinicDB_test.json'},
    'kvasir': {'data_path':'DATA/POLYP/test/Kvasir/images', 'anno_path':'DATA/POLYP/annotations/Kvasir_test.json'},
    'warwick': {'data_path': 'DATA/WarwickQU/images/test', 'anno_path': 'DATA/WarwickQU/annotations/test.json'},
    'bccd': {'data_path': 'DATA/BCCD/test', 'anno_path': 'DATA/BCCD/annotations/test.json'},
    'cpm17': {'data_path': 'DATA/Histopathy/cpm17/images/test', 'anno_path': 'DATA/Histopathy/cpm17/annotations/test.json'},
    'tbx':{'data_path':'DATA/TBX11K/imgs/', 'anno_path':'DATA/TBX11K/annotations/test.json'},
    'buv':{'data_path':'DATA/BUV2022/images/', 'anno_path':'DATA/BUV2022/annotations/test.json'},
    'bbone': {'data_path':'DATA/BackBonepng1/images', 'anno_path':'DATA/BackBonepng1/test_new.json'},
    'tn3k': {'data_path':'DATA/TN3k/images/test', 'anno_path':'DATA/TN3k/annotations/test.json'},
}

resolution = 384
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
patch_resize_transform = transforms.Compose([
    lambda image: image.convert('RGB'),
    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    #transforms.Normalize(mean=mean, std=std)
    ])

def build_model(bert_path, ofa_path,mode='hybrid'):
    '''
    if mode == hybrid, will generate lama and vqa hybrid prompt
    if mode == all, will generate lama prompt for whole data set
    '''
    if mode == 'hybrid' or mode == 'location' or mode == 'llm':
        return None, None, None, None
    #     tokenizer_lama = AutoTokenizer.from_pretrained(bert_path)
    #     model_lama = AutoModelForMaskedLM.from_pretrained(bert_path).to('cuda')
    #     model_lama.eval()

    #     tokenizer_vqa = OFATokenizer.from_pretrained(ofa_path, use_cache=True)
    #     model_vqa = OFAModel.from_pretrained(ofa_path, use_cache=False)
    #     model_vqa.eval()

    #     return model_lama,tokenizer_lama, model_vqa, tokenizer_vqa

    if mode == 'lama':
        tokenizer_lama = AutoTokenizer.from_pretrained(bert_path)
        model_lama = AutoModelForMaskedLM.from_pretrained(bert_path).to('cuda')
        model_lama.eval()

        return model_lama, tokenizer_lama, None, None

def masked_prompt(cls_names, model, tokenizer, mode='hybrid', topk=3):
    '''
    cls_names are the name of each class as a list
    return a prompt info dict:
                {'cls_1': {'location': top1, top2, top3}}
    '''
    res = defaultdict(dict)
    for cls_name in cls_names:
        if mode == 'hybrid' or mode =='llm':
            questions_dict = {
                #'location': f'[CLS] The location of {cls_name} is at [MASK] . [SEP]', #num of mask?
                'location': f'[CLS] Only [MASK] cells have a {cls_name}. [SEP]'
                # 'modality': in [mask] check, we will find polyp
                # 'color': f'[CLS] The typical color of {cls_name} is [MASK] . [SEP]',
                #'shape': f'[CLS] The shape of {cls_name} is [MASK] . [SEP]',
                #'def': f'{cls_name} is a  . [SEP]',
            }
            return res
        elif mode == 'lama':
            # questions_dict = {
            #     'location': f'[CLS] The location of {cls_name} is at [MASK] . [SEP]', #num of mask?
            #     'color': f'[CLS] In a fundus photography, the {cls_name} is in [MASK] color . [SEP]',
            #     'shape': f'[CLS] In a fundus photography, the {cls_name} is [MASK] shape . [SEP]',
            #     #'def': f'{cls_name} is a  . [SEP]',
            # }
            questions_dict = {
            #'location': f'[CLS] Only [MASK] cells have a {cls_name}. [SEP]', #num of mask?
            # 'location': f'[CLS] The {cls_name} normally appears at or near the [MASK] of a cell. [SEP]',
            # 'color': f'[CLS] When a cell is histologically stained, the {cls_name} are in [MASK] color. [SEP]',
            # 'shape': f'[CLS] Mostly the shape of {cls_name} is [MASK]. [SEP]',
            'location': f'[CLS] The location of {cls_name} is at [MASK]. [SEP]',
            'color': f'[CLS] The typical color of {cls_name} is [MASK]. [SEP]',
            'shape': f'[CLS] The typical shape of {cls_name} is [MASK]. [SEP]',
            #'def': f'{cls_name} is a  . [SEP]',
        }

        elif mode == 'location':
            return None


        res[cls_name] = defaultdict(list)
        for k, v in questions_dict.items():
            # import pdb; pdb.set_trace()
            predicted_tokens = []
            input_ids_translated = tokenizer(
                v,
                return_tensors = 'pt'
                ).input_ids.to('cuda')
            tokenized_text = tokenizer.tokenize(v)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            # Create the segments tensors.
            segments_ids = [0] * len(tokenized_text)
            
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
            segments_tensors = torch.tensor([segments_ids]).to('cuda')

            masked_index = tokenized_text.index('[MASK]')
            with torch.no_grad():
                predictions = model(tokens_tensor, segments_tensors)
            
            _, predicted_index = torch.topk(predictions[0][0][masked_index], topk)#.item()
            predicted_index = predicted_index.detach().cpu().numpy()
            #print(predicted_index)
            for idx in predicted_index:
                predicted_tokens.append(tokenizer.convert_ids_to_tokens([idx])[0])
            #print(predicted_tokens)
            temp_str = v.strip().split(' ')
            for i in range(topk):
                #print(predicted_tokens[i])
                #temp_str2 = [predicted_tokens[i] if s == '[MASK]' else s for s in temp_str]
                #print(temp_str2)
                #pred_translated = " ".join(temp_str2.copy())
                #print(pred_translated)
                res[cls_name][k].append(predicted_tokens[i])
            #print(pred_translated)
    return res

import base64
import requests
import json
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage


# OpenAI API Key
api_key = "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"
# api_key = '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'
# api_key = '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'

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def gpt4v(image64, concepts):
    headers = {
  "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }
    # import pdb; pdb.set_trace()
    ans = []
    # test_text = 'can you briefly describe the polyp in the attached image in three aspects: color, shape, and location. Make your description a sentence, so I can use it as a prompt'
    for concept in concepts:
        test_text = f'What is the color and shape of the {concept} in the image, and describe its color and shape in the following format.'\
            +'for example, you should return an output like this:{"color":"pale", "shape":"bump"}'
    # payload = {
    # "model": "gpt-4-vision-preview",
    # "messages": [
    #     {
    #     "role": "user",
    #     "content": [
    #         {
    #         "type": "text",
    #         "text": test_text,#'What is this image',
    #         },
    #         {
    #         "type": "image_url",
    #         "image_url": {
    #             "url": f"data:image/jpeg;base64,{image64}"
    #         }
    #         }
    #     ]
    #     }
    # ],
    # "max_tokens": 3000
    # }
        p =     [
                HumanMessage(
                    content=[
                        {"type": "text", "text": test_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image64}"
                            }
                        },
                    ]
                )
            ]
        chat = ChatOpenAI(model="gpt-4-vision-preview", request_timeout=15, temperature=0)
        res = chat.invoke(p)
        ans.append(res.content)
    
    #response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return ans


def gpt4(concepts):
    # OpenAI API Key
    OPENAI_API_KEY = "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"

    # import pdb;pdb.set_trace()
        

    client = OpenAI(api_key=OPENAI_API_KEY)



    sysinfo = 'You are an assitant who are working on a project to help interpreter and describe medical concept in common language.\
    I am looking for a detailed description of color and shape attributes for a medical concept/object. \
    I will provide you a medical concept/object and you should reply with the description regarding its color and shape attributes \
    You should desribe the attributes in common language rather than terminologies and response should be concise.'

    if type(concepts) is not list:
        test_text = f'Please provide a detailed description of the color and shape attributes for the visual features of {concepts}.'+ \
    'For example, you should return an output following this format:{"color":["pale white", "red", "pink"], "shape":["round", "oval"], loc:["colon"]}. Just use comma to seperate different attribute value, do not use "or" \
    lease all three most common values for each attribute'

        response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": f"{sysinfo}"},
            {"role": "user", "content": f"{test_text}"},
        ],
        temperature=0,
        )

        return response.choices[0].message.content

    else:
        # import pdb;pdb.set_trace()
        ans = []
        for concept in concepts:
            test_text = f'Please provide a detailed description of the color, location, and shape attributes for the visual features of {concept}.'+ \
    'For example, you should return an output following this format:{"color":["pale white", "red", "pink"], "shape":["round", "oval"], loc:["colon"]}. Just use comma to seperate different attribute value, do not use "or" \
    lease all three most common values for each attribute. Try to use common words rather than some advanced words for the attributes.'


            response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": f"{sysinfo}"},
                {"role": "user", "content": f"{test_text}"},
            ],
            temperature=0,
            )

            ans.append(response.choices[0].message.content)

        return ans
def create_prompt(cls_names, imgs, paths, tokenizer, model, vqa_names, real_names, lama_knowledge, mode='hybrid',topk=3):
    import time
    cls_nums = len(cls_names)
    
    # if mode == 'hybrid':
    #     prompt_dict = {
    #     'color': [f'What is the color of these {vqa_names[i]}?' for i in range(cls_nums)],
    #     'shape': [f'What is the shape of these {vqa_names[j]}?' for j in range(cls_nums)],
    #                 }
    # elif mode == 'location':
    #     prompt_dict = {
    #     'location': [f'Where is this {vqa_names[j]} located on?' for j in range(cls_nums)],
    #     }

    #caption = {'prefix':'', 'name':'', 'suffix':''}
    captions, caption = dict(), dict()
    captions_llm = [dict() for _ in range(3)]
    #loc_captions = dict()
    ans_dict = dict()

    if mode=='hybrid':

        for p in paths:
            # import pdb; pdb.set_trace()
            base64_image = encode_image(p)
            response = gpt4v(base64_image, cls_names)
            caps = []
            for idx, cap in enumerate(response):
                try:
                    # cap = cap['choices'][0]['message']['content']
                    json_string = cap#['choices'][0]['message']['content']
                    data = json.loads(json_string)
                    color, shape = data['color'], data['shape']
                    cap = f'{color} color {shape} shape {cls_names[idx]} in fundus'
                    
                except:
                    cap = 'white round bump in fundus'
                if "can't" in cap:
                    cap = 'white round bump in fundus'
                caps.append(cap)

            
            caption['prefix'] = ['']
            caption['name'] = caps#[cap]
            caption['suffix'] = ['']
            caption['caption'] = '. '.join(caps)
            # time.sleep(5)

    elif mode=='llm':
        #concept = 'skin lesion'#'polyps in an endoscopic image'
        concept = cls_names
        # for p in paths:
        # import pdb; pdb.set_trace()
        if len(concept) <= 1:
            caps = ['' for _ in range(3)]
            response = gpt4(concept[0])
            try:
                # cap = cap['choices'][0]['message']['content']
                json_string = response
                data = json.loads(json_string)
                # colors, shapes, locs = data['color'], data['shape'], data['loc']
                colors, shapes = data['color'], data['shape']
                # for k, (color, shape, loc) in enumerate(zip(colors, shapes, locs)):
                for k, (color, shape) in enumerate(zip(colors, shapes)):
                    caps[k] = f'{color} color {shape} shape {real_names[0]}'#{loc}'
            except:
                caps = ['white round bump in fundus' for _ in range(3)]
            # if "can't" in cap:
            #     cap = 'white round bump in fundus'
            for i, cap in enumerate(caps):
                captions_llm[i]['prefix'] = ['']
                captions_llm[i]['name'] = [cap]
                captions_llm[i]['suffix'] = ['']
                captions_llm[i]['caption'] = cap
                # captions_llm[i]['caption'] = '. '.join([cap, cap])
        
        else:
            caps = [['' for _ in range(3)] for _ in range(len(concept))]
            responses = gpt4(concept)
            for ipos, response in enumerate(responses):
                try:
                    # cap = cap['choices'][0]['message']['content']
                    json_string = response
                    
                    data = json.loads(json_string)
                    colors, shapes, locs = data['color'], data['shape'], data['loc']
                    if len(locs) == 1:
                        new_locs = [locs[0] for _ in range(3)]
                    elif len(locs) == 2:
                        new_locs = [locs[0], locs[1], locs[1]]
                    else:
                        new_locs = locs[:]
                    for k, (color, shape, loc) in enumerate(zip(colors, shapes, new_locs)):
                        temp_b = ['smooth', 'uneven']
                        caps[ipos][k] = f'{color} {shape} {real_names[ipos]} with {temp_b[ipos]} edge'# on {loc}'
                except:
                    caps[ipos][k] = ['white round bump in fundus' for _ in range(3)]
            # if "can't" in cap:
            #     cap = 'white round bump in fundus'
            # import pdb; pdb.set_trace()
            # for capdict in caps:
            #     for i, cap in enumerate(capdict):
            # import pdb; pdb.set_trace()
            for i, subcaps in enumerate(zip(*caps)):
                captions_llm[i]['prefix'] = ['']
                captions_llm[i]['name'] = list(subcaps)
                captions_llm[i]['suffix'] = ['']
                captions_llm[i]['caption'] = '. '.join(subcaps)


        return captions_llm
            # time.sleep(1)

    return caption

def gen_prompt(bert_path, ofa_path, args):
    count = 0
    model_lama, tokenizer_lama, model_vqa, tokenizer_vqa = build_model(bert_path, ofa_path, args.mode)
    cls_nums = len(args.cls_names)
    lama_knowledge = masked_prompt(args.cls_names, model_lama, tokenizer_lama, args.mode, args.topk)
    root, annoFile = DATASETMAP[args.dataset]['data_path'], DATASETMAP[args.dataset]['anno_path']
    data_loader = make_dataloader(root, annoFile, patch_resize_transform)
    _iterator = tqdm(data_loader)
    # prompt_top1 = {'prompts': []}
    # prompt_top2 = {'prompts': []}
    # prompt_top3 = {'prompts': []}
    prompt_top1, prompt_top2, prompt_top3 = dict(), dict(), dict()
    # prompt_top1 = dict()
    #TODO automize this shit
    #import pdb; pdb.set_trace()
    llm_flag = True
    captions, caption = [], dict()
    for i, batch in enumerate(_iterator):
        # if count > 2:
        #     break
        # count += 1
        images, targets, paths, *_ = batch
        
        # import pdb; pdb.set_trace()
                    
        if args.mode == 'llm':
            # import pdb; pdb.set_trace()
            if llm_flag:
                llm_flag=False
                

                captions = create_prompt(args.cls_names, images, paths, tokenizer_vqa, model_vqa,
                                            args.vqa_names, args.real_cls_names,
                                            lama_knowledge, args.mode, args.topk)

            # import pdb; pdb.set_trace()
            for j, path in enumerate(paths): #paths batchsize=1
                if images[j] is None:
                    print(path)
                    continue

                prompt_top1[path] = captions[0]
                prompt_top2[path] = captions[1]
                prompt_top3[path] = captions[2]

        elif args.mode == 'hybrid':
            prompts_dict = create_prompt(args.cls_names, images, paths, tokenizer_vqa, model_vqa,
                                         args.vqa_names, args.real_cls_names,
                                        lama_knowledge, args.mode, args.topk)
            # prompt_top1['prompts'] += [prompts_dict[0]]
            # prompt_top2['prompts'] += [prompts_dict[1]]
            # prompt_top3['prompts'] += [prompts_dict[2]]
            #import pdb;pdb.set_trace()
            for i, path in enumerate(paths):
                prompt_top1[path] = prompts_dict
                # prompt_top2[path] = prompts_dict[1]
                # prompt_top3[path] = prompts_dict[2]
    
    # prompt_json1 = json.dumps(prompt_top1)
    # prompt_json2 = json.dumps(prompt_top2)
    # prompt_json3 = json.dumps(prompt_top3)
    #import pdb; pdb.set_trace()
    if args.mode == 'llm':
        with open(f'autoprompt_json/{args.mode}_{args.dataset}_path_prompt_top1.json', 'w') as f1:
            json.dump(prompt_top1, f1)
        
        with open(f'autoprompt_json/{args.mode}_{args.dataset}_path_prompt_top2.json', 'w') as f2:
            json.dump(prompt_top2, f2)

        with open(f'autoprompt_json/{args.mode}_{args.dataset}_path_prompt_top3.json', 'w') as f3:
            json.dump(prompt_top3, f3)

    elif args.mode=='hybrid':
        with open(f'autoprompt_json/{args.mode}_{args.dataset}_path_prompt_top1.json', 'w') as f1:
            json.dump(prompt_top1, f1)
        

gen_prompt(bert_path, ofa_path, args)