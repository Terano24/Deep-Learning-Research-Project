import os
import pandas as pd
import numpy as np
from pathlib import Path

from src.models.cnn_model import arsitekturCNN
from src.data.processing import (
    HLA_Dictionary, dictionary, construct_aaindex,
    peptide_iterate, hla_iterate, label_index
)

# Set base path dynamically relative to the project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def inference(peptide, mhc):
    after_pca_path = BASE_DIR / 'data' / 'raw' / 'after_pca.txt'
    hla_path = BASE_DIR / 'data' / 'raw' / 'ParatopeIMGTopsi.txt'
    model_weights_path = BASE_DIR / 'models' / 'CNN_WEIGHT_OPSI'

    after_pca = np.loadtxt(after_pca_path)
    hla = pd.read_csv(hla_path, sep='\t')
    hla_dic = HLA_Dictionary(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dictionary(inventory)
    
    cnn_model = arsitekturCNN()
    cnn_model.load_weights(str(model_weights_path) + '/')
    
    peptide_score = [peptide]
    hla_score = [mhc]
    immuno_score = ['0']
    
    ori_score = pd.DataFrame({'peptide':peptide_score, 'HLA':hla_score, 'immunogenicity':immuno_score})
    dataset_score = construct_aaindex(ori_score, hla_dic, after_pca, dic_inventory)
    
    input1_score = peptide_iterate(dataset_score)
    input2_score = hla_iterate(dataset_score)
    
    scoring = cnn_model.predict(x=[input1_score, input2_score])
    return float(scoring)

def file_process(upload, download):
    after_pca_path = BASE_DIR / 'data' / 'raw' / 'after_pca.txt'
    hla_path = BASE_DIR / 'data' / 'raw' / 'ParatopeIMGTopsi.txt'
    model_weights_path = BASE_DIR / 'models' / 'CNN_WEIGHT_OPSI'

    after_pca = np.loadtxt(after_pca_path)
    hla = pd.read_csv(hla_path, sep='\t')
    hla_dic = HLA_Dictionary(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dictionary(inventory)
    
    cnn_model = arsitekturCNN()
    cnn_model.load_weights(str(model_weights_path) + '/')

    ori_score = pd.read_csv(upload, sep=',', header=None)
    ori_score.columns = ['peptide', 'HLA']
    ori_score['immunogenicity'] = ['0'] * ori_score.shape[0]
    
    dataset_score = construct_aaindex(ori_score, hla_dic, after_pca, dic_inventory)

    input1_score = peptide_iterate(dataset_score)
    input2_score = hla_iterate(dataset_score)
    
    scoring = cnn_model.predict(x=[input1_score, input2_score])
    ori_score['immunogenicity'] = scoring
    
    output_path = os.path.join(download, 'epoch(detailed).csv')
    ori_score.to_csv(output_path, index=None)
