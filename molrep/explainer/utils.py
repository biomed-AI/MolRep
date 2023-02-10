

from rdkit import Chem
from rdkit import DataStructs

class XAIKeys:

    def __init__(self, fragment):
        self.fragment = fragment

    def GenXAIKeys(self, mol):
        maccsXAIKeys = [(None, 0)] * len(self.fragment)

        for idx, smarts in enumerate(self.fragment):
            patt, count = smarts, 0
            if patt != '?':
                sma = Chem.MolFromSmarts(patt)
                if not sma:
                    print('SMARTS parser error for key %s' % (patt))
                else:
                    maccsXAIKeys[idx] = sma, count

        res = DataStructs.SparseBitVect(len(maccsXAIKeys)+1)
        for i, (patt, count) in enumerate(maccsXAIKeys):
            if patt is not None:
                if count == 0:
                    res[i + 1] = mol.HasSubstructMatch(patt)
                else:
                    matches = mol.GetSubstructMatches(patt)
                    if len(matches) > count:
                        res[i + 1] = 1
        return res
