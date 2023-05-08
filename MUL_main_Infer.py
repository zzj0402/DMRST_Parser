import os
import torch
import numpy as np
import argparse
import os
import config
from transformers import AutoTokenizer, AutoModel
from model_depth import ParsingNet
import re
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.global_gpu_id)


def parse_args():
    parser = argparse.ArgumentParser()
    """ config the saved checkpoint """
    parser.add_argument('--ModelPath', type=str, default='depth_mode/Savings/multi_all_checkpoint.torchsave', help='pre-trained model')
    base_path = config.tree_infer_mode + "_mode/"
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--savepath', type=str, default=base_path + './Savings', help='Model save path')
    args = parser.parse_args()
    return args


def inference(model, tokenizer, input_sentences, batch_size):
    LoopNeeded = int(np.ceil(len(input_sentences) / batch_size))

    input_sentences = [tokenizer.tokenize(i, add_special_tokens=False) for i in input_sentences]
    all_segmentation_pred = []
    all_tree_parsing_pred = []

    with torch.no_grad():
        for loop in range(LoopNeeded):
            StartPosition = loop * batch_size
            EndPosition = (loop + 1) * batch_size
            if EndPosition > len(input_sentences):
                EndPosition = len(input_sentences)

            input_sen_batch = input_sentences[StartPosition:EndPosition]
            _, _, SPAN_batch, _, predict_EDU_breaks = model.TestingLoss(input_sen_batch, input_EDU_breaks=None, LabelIndex=None,
                                                                        ParsingIndex=None, GenerateTree=True, use_pred_segmentation=True)
            all_segmentation_pred.extend(predict_EDU_breaks)
            all_tree_parsing_pred.extend(SPAN_batch)
    return input_sentences, all_segmentation_pred, all_tree_parsing_pred


def segmented_edus(input,edu_breaks):
    edus = []
    first_edu=input[:edu_breaks[0]+1]
# replace _ with empty space
    first_edu=''.join(first_edu)
    edu=first_edu.replace('▁',' ')
    edu_counter=1
    edus.append(edu[1:])
    while edu_counter<len(edu_breaks):
        edu=input[edu_breaks[edu_counter-1]+1:edu_breaks[edu_counter]+1]
        edus.append(''.join(edu).replace('▁',' ')[1:])
        edu_counter+=1
    return edus
    

if __name__ == '__main__':

    args = parse_args()
    model_path = args.ModelPath
    batch_size = args.batch_size
    save_path = args.savepath

    """ BERT tokenizer and model """
    bert_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
    bert_model = AutoModel.from_pretrained("xlm-roberta-base")

    bert_model = bert_model.cuda()

    for name, param in bert_model.named_parameters():
        param.requires_grad = False

    model = ParsingNet(bert_model, bert_tokenizer=bert_tokenizer)

    model = model.cuda()
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    Test_InputSentences = open("./data/text_for_inference.txt").readlines()
    input_sentences, all_segmentation_pred, all_tree_parsing_pred = inference(model, bert_tokenizer, Test_InputSentences, batch_size)
    print('---EDU Boundaries:---')
    print(all_segmentation_pred[0])
    output_index=0
    print('---EDUs---')
    while output_index<len(input_sentences):
        edus=segmented_edus(input_sentences[output_index],all_segmentation_pred[output_index])
        edu_num=1
        for edu in edus:
            print(edu_num, edu)
            edu_num+=1
        output_index+=1
        print('---')
    print('---RST Relations---')
    relation_num=1
    for tree in all_tree_parsing_pred:
        for t in tree:
            # Find all the relations in parantheses
            pattern = re.compile("\((.*?)\)")
            relations = pattern.findall(t)
            for r in relations:
                print(relation_num,r)
                relation_num+=1