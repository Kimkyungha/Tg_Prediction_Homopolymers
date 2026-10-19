import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, InMemoryDataset
from rdkit import Chem
from src3.smiles_utils import *
from dgl import DGLGraph
import dgl
import networkx as nx

def bond_features(bond):
    bt = bond.GetBondType()
    # 결합 유형에 따라 숫자 할당
    bond_order = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 1.5  # 방향족 결합을 1.5로 표현
    }

    numeric_bond_type = bond_order[bt]
    
    # 수치화된 결합 유형 및 반지 여부를 텐서로 반환
    return numeric_bond_type
    
def isNaN(num):
    return num != num
    
def mol2dgl_single(smi):
    if isNaN(smi):
        smi = "C"
    # else:
    n_nodes = 0
    n_edges = 0
    bond_x = []
    mol = Chem.MolFromSmiles(smi)
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()
    g = DGLGraph()
    # g=dgl.graph()
    nodeF = []
    for i, atom in enumerate(mol.GetAtoms()):
        assert i == atom.GetIdx()
        nodeF.append(np.concatenate([atom_symbol_HNums(atom),
                                            atom_degree(atom),
                                            atom_Aroma(atom),
                                            atom_Hybrid(atom),
                                            atom_ring(atom),
                                            atom_FC(atom)],
                                           axis=0))
    g.add_nodes(n_atoms)

    bond_src = []
    bond_dst = []
    for i, bond in enumerate(mol.GetBonds()):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        begin_idx = a1.GetIdx()
        end_idx = a2.GetIdx()
        features = bond_features(bond)

        bond_src.append(begin_idx)
        bond_dst.append(end_idx)
        bond_x.append(features)
        bond_src.append(end_idx)
        bond_dst.append(begin_idx)
        bond_x.append(features)
    g.add_edges(bond_src, bond_dst)
    g.ndata['h'] = torch.Tensor(nodeF)
    g.edata["e"] = torch.Tensor(bond_x)
    
    g = dgl.add_self_loop(g, fill_data=1)
    return g
    
def pol2hig_mk2(polG):
    # 1. NetworkX로부터 상위 그래프 생성 (기존 결합 및 루프 정보 포함)
    pg = dgl.from_networkx(polG.to_networkx(), edge_attrs=["degree"])
    
    # 2. 현재 셀프 루프가 있는 노드와 없는 노드 식별
    u, v = pg.edges()
    self_loop_mask = (u == v)
    nodes_with_loop = torch.unique(u[self_loop_mask])
    
    all_nodes = pg.nodes()
    mask = torch.ones(pg.num_nodes(), dtype=torch.bool)
    mask[nodes_with_loop] = False
    nodes_without_loop = all_nodes[mask] # 루프가 없는 노드들만 추출

    # 3. 루프가 없는 노드에 대해서만 선별적으로 추가
    if len(nodes_without_loop) > 0:
        # 엣지 추가 (구조 생성)
        pg.add_edges(nodes_without_loop, nodes_without_loop)
        
        # 기존 degree 피처의 차원 확인 (스칼라면 1, 벡터면 해당 차원)
        if pg.edata['degree'].dim() > 1:
            feat_dim = pg.edata['degree'].shape[1]
            # 49차원 벡터라면 (N, 49) 형태로 생성
            new_loop_features = torch.ones((len(nodes_without_loop), feat_dim))
        else:
            # 스칼라라면 (N,) 형태로 생성
            new_loop_features = torch.ones(len(nodes_without_loop))
        
        # 새로 생긴 엣지 슬롯(마지막 부분)에만 가중치 1 부여
        pg.edata['degree'][-len(nodes_without_loop):] = new_loop_features

    # --- 이하 하위 노드 통합 로직 ---
    src_list, dst_list = [], []
    nodeF, edgeF = [], []
    
    src_h, dst_h = pg.edges()
    src_list.append(src_h)
    dst_list.append(dst_h)
    
    total_num_node = pg.num_nodes()
    nodeF.append(torch.ones((total_num_node, 49)))
    edgeF.append(pg.edata["degree"])
    
    for i, node in enumerate(polG.nodes):
        # ... (이전과 동일한 mol2dgl_single 호출 및 하위 노드 추가 로직) ...
        smiles = node.fragment.smiles.replace("[R]","[Mg:1]").replace("[Q]","[Mg:2]").replace("[T]","[Mg:3]").replace("[U]","[Mg:4]")
        g = mol2dgl_single(smiles)
        
        # 하위 노드 정보 취합
        src, dst = g.edges()
        src_list.append(src + total_num_node)
        dst_list.append(dst + total_num_node)
        nodeF.append(g.ndata["h"])
        edgeF.append(g.edata["e"])
        
        # 상위-하위 연결 (Star graph)
        src_list.append(g.nodes() + total_num_node)
        dst_list.append(torch.tensor([i] * g.num_nodes()))
        edgeF.append(torch.tensor([1] * g.num_nodes()))
        
        total_num_node += g.num_nodes()

    # 최종 그래프 생성
    nodeF = torch.cat(nodeF)
    final_graph = dgl.graph((torch.cat(src_list), torch.cat(dst_list)), num_nodes=total_num_node)
    final_graph.ndata['h'] = nodeF.float()
    final_graph.edata["e"] = torch.cat(edgeF).float()
    
    return final_graph
    
def mon2hig_mk2(smiles):
    mg = dgl.from_networkx(nx.MultiGraph())
    mg.add_nodes(1)
    src_list, dst_list, nodeF, edgeF = [], [], [], []
    total_num_node = 1
    nodeF.append(torch.ones((total_num_node, 49)))

    g = mol2dgl_single(smiles)
    
    src, dst = g.edges()
    node = g.nodes() + total_num_node
    
    src_list.append(src + total_num_node)
    dst_list.append(dst + total_num_node)
    
    nodeF.append(g.ndata["h"])
    edgeF.append(g.edata["e"])
    
    src_list.append(node)
    dst_list.append(torch.tensor([0] * g.num_nodes()))

    edgeF.append(torch.tensor([1] * g.num_nodes()))
    total_num_node += g.num_nodes()
    nodeF = torch.cat(nodeF)
    final_src = torch.cat(src_list)
    final_dst = torch.cat(dst_list)
    final_graph = dgl.graph((final_src, final_dst), num_nodes=total_num_node)
    final_graph.ndata['h'] = torch.Tensor(nodeF)
    final_graph.edata["e"] = torch.cat(edgeF)

    return final_graph        
    
def smiles_to_graph(smi):
    """
    SMILES 문자열을 받아 단분자 DGL 그래프로 변환하는 함수
    (기존 mol2dgl_single과 동일한 로직)
    """
    if isNaN(smi):
        smi = "C"
        
    bond_x = []
    mol = Chem.MolFromSmiles(smi)
    n_atoms = mol.GetNumAtoms()
    
    g = DGLGraph()
    nodeF = []
    
    # 1. 노드(원자) 피처 추출
    for i, atom in enumerate(mol.GetAtoms()):
        assert i == atom.GetIdx()
        nodeF.append(np.concatenate([
            atom_symbol_HNums(atom),
            atom_degree(atom),
            atom_Aroma(atom),
            atom_Hybrid(atom),
            atom_ring(atom),
            atom_FC(atom)
        ], axis=0))
    g.add_nodes(n_atoms)

    # 2. 엣지(결합) 피처 및 구조 추출 (양방향)
    bond_src = []
    bond_dst = []
    for i, bond in enumerate(mol.GetBonds()):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        begin_idx = a1.GetIdx()
        end_idx = a2.GetIdx()
        features = bond_features(bond)

        bond_src.append(begin_idx)
        bond_dst.append(end_idx)
        bond_x.append(features)
        
        bond_src.append(end_idx)
        bond_dst.append(begin_idx)
        bond_x.append(features)
        
    g.add_edges(bond_src, bond_dst)
    
    # 3. 그래프에 피처 할당
    g.ndata['h'] = torch.Tensor(nodeF)
    g.edata["e"] = torch.Tensor(bond_x)
    
    # 4. Self-loop 추가 (GCN 등의 학습을 위함)
    g = dgl.add_self_loop(g, fill_data=1)
    
    return g

def smiles_to_periodic_graph(smi):
    """
    SMILES 문자열을 받아 주기성(Periodic Boundary Condition)이 반영된 
    고분자 DGL 그래프로 변환하는 함수
    """
    if isNaN(smi):
        smi = "C"
        
    mol = Chem.MolFromSmiles(smi)
    # 3D 좌표나 더 정확한 구조를 원한다면 Chem.AddHs(mol)을 추가할 수 있습니다.
    n_atoms = mol.GetNumAtoms()
    
    # --- [핵심 1] 더미 원자(*)와 연결된 실제 원자(양 끝단) 찾기 ---
    nearest_neighbors = []
    for atom in mol.GetAtoms():
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 0:
                nearest_neighbors.append(atom.GetIdx())
    nearest_neighbors = list(set(nearest_neighbors))

    g = dgl.graph(([], [])) # 빈 그래프 생성 (DGLGraph()는 최신버전에서 deprecate되어 dgl.graph 권장)
    nodeF = []
    
    # --- [핵심 2] 노드(원자) 피처 추출 및 더미 원자 무효화 ---
    for i, atom in enumerate(mol.GetAtoms()):
        assert i == atom.GetIdx()
        
        # 원래의 특성 벡터 생성
        feat = np.concatenate([
            atom_symbol_HNums(atom),
            atom_degree(atom),
            atom_Aroma(atom),
            atom_Hybrid(atom),
            atom_ring(atom),
            atom_FC(atom)
        ], axis=0)
        
        # 더미 원자(*)인 경우 특성을 모두 0으로 덮어씌움 (Zero-padding)
        if atom.GetAtomicNum() == 0:
            feat = np.zeros_like(feat)
            
        nodeF.append(feat)

    g.add_nodes(n_atoms)

    bond_src = []
    bond_dst = []
    bond_x = []

    # --- [핵심 3-1] 실제 엣지(결합) 추출 (양방향) ---
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        features = bond_features(bond)

        bond_src.extend([a1, a2])
        bond_dst.extend([a2, a1])
        bond_x.extend([features, features])
        
    # --- [핵심 3-2] 주기적 가상 엣지(Virtual Bond) 추가 ---
    for a1 in range(n_atoms):
        for a2 in range(a1 + 1, n_atoms):
            # 두 원자가 모두 양 끝단(nearest_neighbors)에 속하는 경우
            if (a1 in nearest_neighbors) and (a2 in nearest_neighbors):
                # 이미 결합이 없다면 가상 결합 추가
                if mol.GetBondBetweenAtoms(a1, a2) is None:
                    # a1이 더미 원자와 맺고 있던 결합 종류를 가져옴
                    virtual_bond_feat = 1.0 # 기본값
                    for neighbor in mol.GetAtomWithIdx(a1).GetNeighbors():
                        if neighbor.GetAtomicNum() == 0:
                            dummy_bond = mol.GetBondBetweenAtoms(a1, neighbor.GetIdx())
                            virtual_bond_feat = bond_features(dummy_bond)
                            break
                    
                    # 가상 결합도 양방향으로 추가
                    bond_src.extend([a1, a2])
                    bond_dst.extend([a2, a1])
                    bond_x.extend([virtual_bond_feat, virtual_bond_feat])

    # 그래프에 엣지 추가
    g.add_edges(bond_src, bond_dst)
    
    # 텐서 변환 및 그래프에 피처 할당
    g.ndata['h'] = torch.Tensor(nodeF)
    # edge feature가 1차원 리스트이므로 unsqueeze를 통해 [num_edges, 1] 형태로 만들어주는 것이 GNN 학습에 유리할 수 있습니다.
    g.edata["e"] = torch.Tensor(bond_x).unsqueeze(-1) 
    
    # 4. Self-loop 추가
    # 주의: edata가 추가된 상태에서 self loop를 넣으면 edge feature 차원이 맞지 않아 에러가 날 수 있습니다.
    # fill_data를 통해 self loop의 엣지 피처값을 1로 채워줍니다.
    g = dgl.add_self_loop(g, fill_data=1.0)
    
    return g