"""
Pocket-anchor cross-attention network for peptide-HLA immunogenicity.

  * peptide (10 positions, 9-mers get a gap at P5 like the original encoder) and the
    46-residue HLA pseudo-sequence are embedded with learned residue + position embeddings
  * self-attention within each sequence, then cross-attention in both directions so peptide
    positions can attend to the HLA pocket residues they contact (and vice versa)
  * the peptide stream is flattened (position-specific, keeps anchor positions) and joined
    with the pooled HLA stream into a shared trunk
  * two heads on the trunk: MHC binding/presentation (used for transfer pretraining) and
    immunogenicity. The immunogenicity logit is  global(z) + lung_flag * adapter(z)  so the
    small lung-cancer subset can specialise without overwriting what the full data taught.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

AMINO = 'ACDEFGHIKLMNPQRSTVWY-'
AA_INDEX = {a: i for i, a in enumerate(AMINO)}
PEP_LEN, HLA_LEN = 10, 46


def tokenize_peptide(p):
    if len(p) == 9:
        p = p[:5] + '-' + p[5:]
    return [AA_INDEX.get(c, AA_INDEX['-']) for c in p]


def tokenize_hla(seq):
    return [AA_INDEX.get(c, AA_INDEX['-']) for c in seq]


class PositionIds(layers.Layer):
    """Returns [0..L-1] broadcast to the batch (keeps the graph Keras-functional)."""
    def call(self, x):
        n = tf.shape(x)[1]
        return tf.tile(tf.range(n)[None, :], [tf.shape(x)[0], 1])


def _embed(tokens, length, d, name):
    tok = layers.Embedding(len(AMINO), d, name=f'{name}_tok')(tokens)
    pos = layers.Embedding(length, d, name=f'{name}_pos')(PositionIds()(tokens))
    return layers.LayerNormalization(name=f'{name}_ln')(tok + pos)


def _attend(q, kv, d, heads, name, dropout):
    a = layers.MultiHeadAttention(heads, d // heads, dropout=dropout, name=name)(q, kv)
    return layers.LayerNormalization(name=f'{name}_ln')(q + a)


def build_model(d=32, heads=4, trunk=128, dropout=0.2, use_lung_adapter=True, n_aux=0):
    """n_aux > 0 adds a standardised biochemical / self-similarity feature vector
    (see src/data/features.py) that is concatenated into the shared trunk."""
    pep = layers.Input(shape=(PEP_LEN,), dtype='int32', name='peptide')
    hla = layers.Input(shape=(HLA_LEN,), dtype='int32', name='hla')
    lung = layers.Input(shape=(1,), dtype='float32', name='lung_flag')
    aux = layers.Input(shape=(max(n_aux, 1),), dtype='float32', name='aux')

    p = _embed(pep, PEP_LEN, d, 'pep')
    h = _embed(hla, HLA_LEN, d, 'hla')
    p = _attend(p, p, d, heads, 'pep_self', dropout)
    h = _attend(h, h, d, heads, 'hla_self', dropout)
    p2 = _attend(p, h, d, heads, 'pep_to_hla', dropout)   # peptide positions query the pocket
    h2 = _attend(h, p, d, heads, 'hla_to_pep', dropout)   # pocket residues query the peptide

    parts = [layers.Flatten()(p2), layers.GlobalAveragePooling1D()(h2)]
    if n_aux:
        parts.append(layers.Dense(16, activation='relu', name='aux_proj')(aux))
    z = layers.Concatenate()(parts) if len(parts) > 1 else parts[0]
    z = layers.Dense(trunk, activation='relu', name='trunk')(z)
    z = layers.Dropout(dropout)(z)

    binding = layers.Dense(1, activation='sigmoid', name='binding')(z)
    logit = layers.Dense(1, name='imm_global')(z)
    if use_lung_adapter:
        adapter = layers.Dense(1, name='imm_lung_adapter', kernel_initializer='zeros')(z)
        logit = logit + lung * adapter
    imm = layers.Activation('sigmoid', name='immunogenicity')(logit)
    return tf.keras.Model([pep, hla, lung, aux], [binding, imm])
