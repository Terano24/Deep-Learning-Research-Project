import os
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.cnn_architecture import arsitekturCNN
from src.data.matrix_encoders import (
    HLA_Dictionary, dictionary, construct_aaindex, 
    peptide_iterate, hla_iterate, label_index
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def inference(peptide, mhc):
    after_pca = np.loadtxt(os.path.join(BASE_DIR, 'data/after_pca.txt'))
    hla = pd.read_csv(os.path.join(BASE_DIR, 'data/ParatopeIMGTopsi.txt'), sep='\t')
    hla_dic = HLA_Dictionary(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dictionary(inventory)
    cnn_model = arsitekturCNN()
    cnn_model.load_weights(os.path.join(BASE_DIR, 'models/CNN_WEIGHT_OPSI/'))
    peptide_score = [peptide]
    hla_score = [mhc]
    immuno_score = ['0']
    ori_score = pd.DataFrame({'peptide':peptide_score,'HLA':hla_score,'immunogenicity':immuno_score})
    dataset_score = construct_aaindex(ori_score,hla_dic,after_pca,dic_inventory)
    input1_score = peptide_iterate(dataset_score)
    input2_score = hla_iterate(dataset_score)
    label_score = label_index(dataset_score)
    scoring = cnn_model.predict(x=[input1_score,input2_score])
    return float(scoring.item())

def file_process(upload, download):
    after_pca = np.loadtxt(os.path.join(BASE_DIR, 'data/after_pca.txt'))
    hla = pd.read_csv(os.path.join(BASE_DIR, 'data/ParatopeIMGTopsi.txt'), sep='\t')
    hla_dic = HLA_Dictionary(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dictionary(inventory)
    cnn_model = arsitekturCNN()
    cnn_model.load_weights(os.path.join(BASE_DIR, 'models/CNN_WEIGHT_OPSI/'))

    ori_score = pd.read_csv(upload, sep=',', header=None)
    ori_score.columns = ['peptide', 'HLA']
    ori_score['immunogenicity'] = ['0'] * ori_score.shape[0]
    dataset_score = construct_aaindex(ori_score, hla_dic, after_pca, dic_inventory)

    input1_score = peptide_iterate(dataset_score)
    input2_score = hla_iterate(dataset_score)
    label_score = label_index(dataset_score)
    scoring = cnn_model.predict(x=[input1_score, input2_score])
    ori_score['immunogenicity'] = scoring
    ori_score.to_csv(os.path.join(download, 'epoch(detailed).csv'), index=None)
