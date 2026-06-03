import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================= 配置 =================
'''训练数据'''
'''316data'''
# FILE_TRAIN_DATA = r'C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\training_data_120D.csv'
# SAVE_PATH_CAT_GRAPHS_YIELD = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\model_evaluation_strict-manual-GNN\yield\processed_cat_graphs_yield.pt"
# SAVE_PATH_LIG_GRAPHS_YIELD = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\model_evaluation_strict-manual-GNN\yield\processed_lig_graphs_yield.pt"
# MODEL_YIELD_PATH = r'C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\model_evaluation_strict-manual-GNN\yield\new\best_gnn_model_48-0130.pth' # 316data
# MODEL_SEL_PATH = r'C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\model_evaluation_strict-manual-GNN\selectivity\best_gnn_model_48.pth'

'''548data'''
# FILE_TRAIN_DATA = r'C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\6new_pre_data\training_data_120D.csv'
# SAVE_PATH_CAT_GRAPHS_YIELD = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\6new_pre_data\GNN_train_with_yield\processed_cat_graphs_yield.pt"
# SAVE_PATH_LIG_GRAPHS_YIELD = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\6new_pre_data\GNN_train_with_yield\processed_lig_graphs_yield.pt"
# MODEL_YIELD_PATH = r'C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\6new_pre_data\GNN_train_with_yield\new\best_gnn_model_48_0130.pth' # 548data

'''143data'''
FILE_TRAIN_DATA = r'C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\5Cat_1+N_Ligand\training_data_120D.csv'
SAVE_PATH_CAT_GRAPHS_YIELD = \
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\5Cat_1+N_Ligand\manul_GNN\processed_cat_graphs_yield.pt"
SAVE_PATH_LIG_GRAPHS_YIELD = \
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\5Cat_1+N_Ligand\manul_GNN\processed_lig_graphs_yield.pt"
MODEL_YIELD_PATH = r'C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\6new_pre_data\GNN_train_with_yield\new\best_gnn_model_48_0130.pth' # 548data


'''预测数据'''
# FILE_pred_DATA = r'C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\4\pred_pair_120D_data_923_ligand.csv'
# SAVE_PATH_CAT_GRAPHS_YIELD = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\4\processed_cat_graphs.pt"
# SAVE_PATH_LIG_GRAPHS_YIELD = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\4\processed_lig_graphs.pt"


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 必须重新定义模型类以加载权重 (保持与训练时一致)
'''     
yield: hidden_dim=48    
selectivity: hidden_dim=48'''
class DualTowerGNN(nn.Module):
    def __init__(self, hidden_dim=32):  # 减小维度防止过拟合
        super().__init__()
        self.embedding = nn.Embedding(120, hidden_dim)

        # 1. 边特征变换层 (用于 GINE)
        # 边特征是 1维 (键长)，映射到 hidden_dim
        self.edge_mlp1 = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

        # 2. GINE 卷积层 (Graph Isomorphism Network with Edge features)
        # 这是一个比 GCN 强得多的算子，专用于分子图
        self.conv1 = GINEConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)), edge_dim=hidden_dim)
        self.conv2 = GINEConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)), edge_dim=hidden_dim)

        # 3. 初始节点特征映射 (Mass+XYZ -> Hidden)
        self.node_map = nn.Linear(4, hidden_dim)

        # 4. Global Info 处理
        self.global_mlp = nn.Sequential(nn.Linear(4, 16), nn.ReLU())

        # 5. 预测头 (Cat_Graph + Lig_Graph + Cat_Global + Lig_Global)
        # 输入维度: 32 + 32 + 16 + 16 = 96
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.4),  # 增加 Dropout
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward_tower(self, data):
        # A. 节点特征初始化: Embedding(AtomNum) + Linear(Mass, XYZ)
        h1 = self.embedding(data.atom_nums)
        h2 = self.node_map(data.x)
        h = h1 + h2  # 相加融合

        # B. 图卷积 (带边特征)
        # 注意：GINE 需要边属性
        edge_attr = data.edge_attr

        # Conv 1
        edge_emb1 = self.edge_mlp1(edge_attr)
        h = self.conv1(h, data.edge_index, edge_attr=edge_emb1)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)

        # Conv 2
        edge_emb2 = self.edge_mlp2(edge_attr)
        h = self.conv2(h, data.edge_index, edge_attr=edge_emb2)
        h = F.relu(h)

        # C. 全局池化
        graph_vec = global_add_pool(h, data.batch)  # 维度: 48 (hidden_dim)

        # D. Global Info
        glob_vec = self.global_mlp(data.global_attr.view(-1, 4))  # # 维度: 16 (硬编码)

        return graph_vec, glob_vec  # 48 + 16 = 64

    def forward(self, cat_batch, lig_batch):
        g_cat, glob_cat = self.forward_tower(cat_batch)
        g_lig, glob_lig = self.forward_tower(lig_batch)

        # 拼接所有特征
        combined = torch.cat([g_cat, g_lig, glob_cat, glob_lig], dim=1)
        return self.predictor(combined)


# 建立 filename -> embedding vector 的映射
def extract_embeddings(model, graph_list):
    embeddings = {}
    loader = DataLoader(graph_list, batch_size=32, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            # 使用 forward_tower 提取特征 [Batch, 48+16] = [Batch, 64]
            # 2. 将它们拼接起来 (dim=1 表示在特征维度拼接)
            # g_vec 维度是 [Batch, 48], glob_vec 维度是 [Batch, 16]
            # 拼接后 vecs 维度是 [Batch, 64]

            g_vec, glob_vec = model.forward_tower(batch)
            vecs = torch.cat([g_vec, glob_vec], dim=1)

            # 3. 现在 vecs 是一个 Tensor，可以调用 .cpu()
            vecs = vecs.cpu().numpy()

            # 对应文件名
            for i, filename in enumerate(batch.filename):
                embeddings[filename] = vecs[i]
                # 注意：如果 filename 是列表或元组里的单个元素，通常不需要再处理
                # 但如果你的 batch.filename 是 list，这里没问题
                if hasattr(filename, 'lower'):
                    embeddings[filename.lower()] = vecs[i]

    return embeddings


def create_merged_csv(cat_emb, lig_emb, suffix):
    # 字典转 DF
    def dict_to_df(emb_map, type_name):
        data, ids = [], []
        first_key = next(iter(emb_map))
        dim = emb_map[first_key].shape[0]
        for key, vec in emb_map.items():
            ids.append(key)
            data.append(vec)
        feat_cols = [f'feat_gnn_{type_name}_{i + 1}' for i in range(dim)]
        df = pd.DataFrame(data, columns=feat_cols)
        df.insert(0, 'ID_Name', ids)
        return df.drop_duplicates(subset=['ID_Name'])

    df_cat = dict_to_df(cat_emb, "Cat")  # 获得cat的嵌入向量
    df_lig = dict_to_df(lig_emb, "Lig")  # 获得lig的嵌入向量

    df_base = pd.read_csv(FILE_TRAIN_DATA)
    # df_base = pd.read_csv(FILE_pred_DATA)

    '''这里因为只是用训练好得模型基于分子图生成向量，因此不需要构建inv-selectivity列'''

    # ID 匹配逻辑
    split_names = df_base['cat-ligand'].str.split('-', n=1, expand=True)
    df_base['temp_cat_id'] = split_names[0].str.lower()
    df_base['temp_lig_id'] = split_names[1]  # 注意大小写

    df_merged = pd.merge(df_base, df_cat, left_on='temp_cat_id', right_on='ID_Name', how='left')
    df_merged = pd.merge(df_merged, df_lig, left_on='temp_lig_id', right_on='ID_Name', how='left')

    df_merged = df_merged.drop(columns=['temp_cat_id', 'temp_lig_id', 'ID_Name_x', 'ID_Name_y'], errors='ignore')
    df_merged = df_merged.fillna(0)

    shape = cat_emb.get(list(cat_emb.keys())[0]).shape
    print(f"--------------------task_name: {suffix}; embedding shape: {shape}------------------------")

    # if suffix == "yield":
    # SAVE_DIR = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\model_evaluation_strict-manual-GNN\yield\new"
    # elif suffix == "selectivity" or 'Inv_Selectivity':
    #     SAVE_DIR = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\model_evaluation_strict-manual-GNN\selectivity"
    # save_path = os.path.join(SAVE_DIR, f"training_data_120D_with_GNN_{shape}D-0130.csv")

    SAVE_DIR = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\5Cat_1+N_Ligand\manul_GNN\new"
    # SAVE_DIR = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\6new_pre_data\GNN_train_with_yield\new"
    save_path = os.path.join(SAVE_DIR, f"training_data_120D_with_GNN_{shape}D_0130.csv")

    df_merged.to_csv(save_path, index=False)
    print(f"GNN_embedding_vec_with_120D 已经保存到 {save_path}")
    return df_merged, save_path


if __name__ == "__main__":
    # # --- A. 处理 Yield ---
    print("\n>>> Processing Yield Model...")
    # cat_list = torch.load(SAVE_PATH_CAT_GRAPHS_YIELD, weights_only=True)
    # lig_list = torch.load(SAVE_PATH_LIG_GRAPHS_YIELD, weights_only=True)
    cat_list = torch.load(SAVE_PATH_CAT_GRAPHS_YIELD, weights_only=False)
    lig_list = torch.load(SAVE_PATH_LIG_GRAPHS_YIELD, weights_only=False)

    model_yld = DualTowerGNN(hidden_dim=48).to(DEVICE)
    model_yld.load_state_dict(torch.load(MODEL_YIELD_PATH))
    # model_yld.load_state_dict(torch.load(MODEL_SEL_PATH))
    model_yld.eval()

    emb_cat_yld = extract_embeddings(model_yld, cat_list)
    emb_lig_yld = extract_embeddings(model_yld, lig_list)

    df_yld, path_yld = create_merged_csv(emb_cat_yld, emb_lig_yld, "yield")
    # df_yld, path_yld = create_merged_csv(emb_cat_yld, emb_lig_yld, "Inv_Selectivity")


