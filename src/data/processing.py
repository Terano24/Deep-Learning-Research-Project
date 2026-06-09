import numpy as np
import pandas as pd

def peptide_iterate(dataset):
    result = np.empty([len(dataset),10,12,1])
    for i in range(len(dataset)):
        result[i,:,:,:] = dataset[i][0]
    return result

def hla_iterate(dataset):
    result = np.empty([len(dataset),46,12,1])
    for i in range(len(dataset)):
        result[i,:,:,:] = dataset[i][1]
    return result
    
def label_index(dataset):
    result = np.empty([len(dataset),1])
    for i in range(len(dataset)):
        result[i,:] = dataset[i][2]
    return result

def aaindex(peptide,after_pca):
    amino = 'ARNDCQEGHILKMFPSTWYV-'
    matrix = np.transpose(after_pca)  
    encoded = np.empty([len(peptide), 12])  
    for i in range(len(peptide)):
        query = peptide[i]
        if query == 'X': query = '-'
        query = query.upper()
        encoded[i, :] = matrix[:, amino.index(query)]
    return encoded

def peptide_data_aaindex(peptide, after_pca):
    length = len(peptide)
    encode = None 

    if length == 10:
        encode = aaindex(peptide, after_pca)
    elif length == 9:
        peptide = peptide[:5] + '-' + peptide[5:]
        encode = aaindex(peptide, after_pca)

    if encode is not None:  
        encode = encode.reshape(encode.shape[0], encode.shape[1], -1)

    return encode

def dictionary(inventory):
    dicA, dicB, dicC = {}, {}, {}
    dic = {'A': dicA, 'B': dicB, 'C': dicC}

    for hla in inventory:
        type_ = hla[4] 
        first2 = hla[6:8] 
        last2 = hla[8:]  
        try:
            dic[type_][first2].append(last2)
        except KeyError:
            dic[type_][first2] = []
            dic[type_][first2].append(last2)
    return dic

def recover_hla(hla, dic_inventory):
    type_ = hla[4]
    first2 = hla[6:8]
    last2 = hla[8:]
    big_category = dic_inventory[type_]
    if not big_category.get(first2) == None:
        small_category = big_category.get(first2)
        distance = [abs(int(last2.replace(':', '')) - int(i.replace(':', ''))) for i in small_category]
        optimal = min(zip(small_category, distance), key=lambda x: x[1])[0]
        return 'HLA-' + str(type_) + '*' + str(first2) + str(optimal)
    else:
        small_category = list(big_category.keys())
        distance = [abs(int(first2.replace(':', '')) - int(i.replace(':', ''))) for i in small_category]
        optimal = min(zip(small_category, distance), key=lambda x: x[1])[0]
        return 'HLA-' + str(type_) + '*' + str(optimal) + str(big_category[optimal][0])

def hla_data_aaindex(hla_dic,hla_type,after_pca,dic_inventory):    
    try:
        seq = hla_dic[hla_type]
    except KeyError:
        hla_type = recover_hla(hla_type,dic_inventory)
        seq = hla_dic[hla_type]
    encode = aaindex(seq,after_pca)
    encode = encode.reshape(encode.shape[0], encode.shape[1], -1)
    return encode

def construct_aaindex(ori,hla_dic,after_pca,dic_inventory):
    series = []
    for i in range(ori.shape[0]):
        peptide = ori['peptide'].iloc[i]
        hla_type = ori['HLA'].iloc[i]
        immuno = np.array(ori['immunogenicity'].iloc[i]).reshape(1,-1)   

        encode_pep = peptide_data_aaindex(peptide,after_pca)   
        encode_hla = hla_data_aaindex(hla_dic,hla_type,after_pca,dic_inventory)   
        series.append((encode_pep, encode_hla, immuno))
    return series

def HLA_Dictionary(hla):
    dic = {}
    for i in range(hla.shape[0]):
        col1 = hla['HLA'].iloc[i] 
        col2 = hla['pseudo'].iloc[i]  
        dic[col1] = col2
    return dic
