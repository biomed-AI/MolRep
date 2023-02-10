import torch
import rdkit.Chem as Chem
import networkx as nx

from molrep.explainer.attribution.utils.fuseprop.mol_graph import MolGraph
from molrep.explainer.attribution.utils.fuseprop.chemutils import *
from collections import defaultdict

class IncBase(object):

    def __init__(self, batch_size, node_fdim, edge_fdim, max_nodes, max_edges, max_nb):
        self.max_nb = max_nb
        self.graph = nx.DiGraph()
        self.graph.add_node(0) #make sure node is 1 index
        self.edge_dict = {None : 0} #make sure edge is 1 index

        self.fnode = torch.zeros(max_nodes * batch_size, node_fdim).cuda()
        self.fmess = self.fnode.new_zeros(max_edges * batch_size, edge_fdim)
        self.agraph = self.fnode.new_zeros(max_nodes * batch_size, max_nb).long()
        self.bgraph = self.fnode.new_zeros(max_edges * batch_size, max_nb).long()

    def add_node(self, feature):
        idx = len(self.graph)
        if idx >= len(self.fnode) - 1:
            self.fnode = torch.cat([self.fnode, self.fnode * 0], dim=0)
            self.agraph = torch.cat([self.agraph, self.agraph * 0], dim=0)

        self.graph.add_node(idx)
        self.fnode[idx, :len(feature)] = feature
        return idx

    def can_expand(self, idx):
        return self.graph.in_degree(idx) < self.max_nb

    def add_edge(self, i, j, feature=None):
        if (i,j) in self.edge_dict: 
            return self.edge_dict[(i,j)]

        self.graph.add_edge(i, j)
        self.edge_dict[(i,j)] = idx = len(self.edge_dict)

        if idx >= len(self.fmess) - 1:
            self.fmess = torch.cat([self.fmess, self.fmess * 0], dim=0)
            self.bgraph = torch.cat([self.bgraph, self.bgraph * 0], dim=0)

        self.agraph[j, self.graph.in_degree(j) - 1] = idx
        if feature is not None:
            self.fmess[idx, :len(feature)] = feature

        in_edges = [self.edge_dict[(k,i)] for k in self.graph.predecessors(i) if k != j]
        self.bgraph[idx, :len(in_edges)] = self.fnode.new_tensor(in_edges)

        for k in self.graph.successors(j):
            if k == i: continue
            nei_idx = self.edge_dict[(j,k)]
            self.bgraph[nei_idx, self.graph.in_degree(j) - 2] = idx

        return idx


class IncGraph(IncBase):

    def __init__(self, avocab, batch_size, node_fdim, edge_fdim, max_nodes=20, max_edges=50, max_nb=6):
        super(IncGraph, self).__init__(batch_size, node_fdim, edge_fdim, max_nodes, max_edges, max_nb)
        self.avocab = avocab
        self.mol = Chem.RWMol()
        self.mol.AddAtom( Chem.Atom('C') ) #make sure node is 1 index, consistent to self.graph
        self.batch = defaultdict(list)
        self.interior_atoms = defaultdict(list)

    def get_mol(self):
        mol_list = [None] * len(self.batch)
        for batch_idx, batch_atoms in self.batch.items():
            mol = get_sub_mol(self.mol, batch_atoms)
            mol = sanitize(mol, kekulize=False)
            if mol is None: 
                mol_list[batch_idx] = None
            else:
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                mol_list[batch_idx] = Chem.MolToSmiles(mol)
        return mol_list

    def get_tensors(self):
        return self.fnode, self.fmess, self.agraph, self.bgraph

    def add_mol(self, bid, smiles):
        mol = get_mol(smiles)  #Important: must kekulize!
        root_atoms = []
        amap = {}
        for atom in mol.GetAtoms():
            symbol, charge = atom.GetSymbol(), atom.GetFormalCharge()
            nth_atom = atom.GetAtomMapNum()
            idx = self.add_atom(bid, (symbol, charge), nth_atom)
            amap[atom.GetIdx()] = idx
            if nth_atom > 0:
                root_atoms.append( (idx, nth_atom) )
            else:
                self.interior_atoms[bid].append(idx)

        for bond in mol.GetBonds():
            a1 = amap[bond.GetBeginAtom().GetIdx()]
            a2 = amap[bond.GetEndAtom().GetIdx()]
            bt = bond.GetBondType()
            self.add_bond(a1, a2, MolGraph.BOND_LIST.index(bt))

        root_atoms = sorted(root_atoms, key=lambda x:x[1])
        root_atoms = next(zip(*root_atoms))
        return root_atoms

    def add_atom(self, bid, atom_type, nth_atom=None):
        if nth_atom is None:
            nth_atom = len(self.batch[bid]) - len(self.interior_atoms[bid]) + 1
        new_atom = Chem.Atom(atom_type[0])
        new_atom.SetFormalCharge(atom_type[1])
        atom_feature = self.get_atom_feature(new_atom, nth_atom)
        aid = self.mol.AddAtom( new_atom )
        assert aid == self.add_node( atom_feature )
        self.batch[bid].append(aid)
        return aid
        
    def add_bond(self, a1, a2, bond_pred):
        assert 1 <= bond_pred <= 3
        if a1 == a2: return
        if self.can_expand(a1) == False or self.can_expand(a2) == False:
            return
        if self.mol.GetBondBetweenAtoms(a1, a2) is not None:
            return

        atom1, atom2 = self.mol.GetAtomWithIdx(a1), self.mol.GetAtomWithIdx(a2)
        if valence_check(atom1, bond_pred) and valence_check(atom2, bond_pred):
            bond_type = MolGraph.BOND_LIST[bond_pred]
            self.mol.AddBond(a1, a2, bond_type)
            self.add_edge( a1, a2, self.get_mess_feature(self.fnode[a1], bond_pred) )
            self.add_edge( a2, a1, self.get_mess_feature(self.fnode[a2], bond_pred) )
        
        # TOO SLOW!
        #if sanitize(self.mol.GetMol(), kekulize=False) is None:
        #    self.mol.RemoveBond(a1, a2)
        #    return


    def get_atom_feature(self, atom, nth_atom):
        nth_atom = min(MolGraph.MAX_POS - 1, nth_atom)
        f_atom = torch.zeros(self.avocab.size())
        f_pos = torch.zeros( MolGraph.MAX_POS )
        symbol, charge = atom.GetSymbol(), atom.GetFormalCharge()
        f_atom[ self.avocab[(symbol,charge)] ] = 1
        f_pos[ nth_atom ] = 1
        return torch.cat( [f_atom, f_pos], dim=-1 ).cuda()

    def get_mess_feature(self, atom_fea, bond_type):
        bond_fea = torch.zeros(len(MolGraph.BOND_LIST)).cuda()
        bond_fea[ bond_type ] = 1
        return torch.cat( [atom_fea, bond_fea], dim=-1 )

