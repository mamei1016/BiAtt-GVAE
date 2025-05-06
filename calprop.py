"""使用RDKIT库可以计算分子的SAscore（Synthetic Accessibility score）。SAscore是评估分子合成易用性的指标，
通过估计分子中反应可能性和步骤数量来量化分子的合成难度。

以下是使用RDKIT计算SAscore的示例代码："""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors,Lipinski,Crippen
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA,CalcExactMolWt
from rdkit.Chem import QED
from metrics.SA_Score import sascorer
from metrics.NP_Score import npscorer
from rdkit.Chem import Draw

def cal_prop(smile):
    m = Chem.MolFromSmiles(smile)
    if m is None : return None
    SAscore = sascorer.calculateScore(m)
    NP_score =npscorer.scoreMol(m)
    qed =QED.qed(m)
    lop=MolLogP(m)
    mw = CalcExactMolWt(m)
    lip = obey_lipinski(m)

    return mw, lop, qed,SAscore, NP_score,lip


def obey_lipinski(mol):
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    logp = Crippen.MolLogP(mol)
    rule_4 = (logp>=-2) & (logp<=5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])
# 示例用法
'''
smile = 'NN1CCN(C(c2ccccc2)c2ccccc2)CC1'
gen_mol= Chem.MolFromSmiles(smile)
Draw.MolToImage(gen_mol, size=(150,150), kekulize=True)
Draw.ShowMol(gen_mol, size=(150,150), kekulize=True)
img = Draw.MolToFile(gen_mol, 'mqs.png')

Wt, MolLoP, Qed,SA,NP,lip = cal_prop(smile)
print(Wt, MolLoP, Qed, SA,NP, lip)
'''



