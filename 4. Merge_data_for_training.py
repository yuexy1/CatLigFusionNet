import os
import numpy as np
import pandas as pd
from ast import literal_eval
from tqdm import tqdm

# ================= 配置路径 (请修改为你实际的路径) =================
# 催化剂光谱文件
CAT_CSV = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\IrSpecInfo_catalyst.csv"
# LIG_CSV = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\IrSpecInfo_ligand_used.csv"
LIG_CSV = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\IrSpecInfo_ligand_945.csv"

# 实验数据文件
# EXP_CSV = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\exp_data_matched_updated.csv"  # 数据有问题
EXP_CSV = \
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\6new_pre_data\exp_data_matched-316-548-HJ.csv"
# r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\5Cat_1+N_Ligand\exp_data_matched_updated-145.csv"
# EXP_CSV = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\2\predict_pairs_2.csv"

# 输出文件路径
# OUT_CSV = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\training_data_200D.csv"  # 48 72 80 96 100
# OUT_CSV = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\2\predict_data_120D_to_model_2.csv"
OUT_CSV = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\6new_pre_data\training_data_120D.csv"

# 参数设置：生成 48维特征 (24 + 24)
# 光谱长度 3600 / 150 = 24 bins
SEG_LEN_POINTS = 60    # or 150 100 90 75 72 60 50 48 45 40 36


# ================= 功能函数 =================

def parse_list_literal(series: pd.Series) -> pd.Series:
    """将字符串形式的 Python 列表安全解析为 Python list"""
    return series.apply(literal_eval)


def load_ir_data(csv_path: str):
    """
    读取光谱CSV，返回一个字典: {Filename: numpy_array_of_intensities}
    同时执行归一化处理。
    """
    print(f"正在读取光谱文件: {csv_path} ...")
    df = pd.read_csv(csv_path, sep="\t", engine="python", encoding="utf-8")

    # 确定强度列名
    col_name = "IR_intens_normalized" if "IR_intens_normalized" in df.columns else "IR_intens"

    data_dict = {}
    # 解析列表
    df['parsed_intens'] = parse_list_literal(df[col_name].astype(str))

    # 检查是否有 freqs 列，如果有也需要解析
    if 'freqs' in df.columns:
        df['parsed_freqs'] = parse_list_literal(df['freqs'].astype(str))
    else:
        print(f"警告: {csv_path} 中没有 freqs 列，将使用默认 0-1 线性分布")
        df['parsed_freqs'] = None

    for idx, row in df.iterrows():
        name = str(row['Filename']).strip()
        val_intens = np.array(row['parsed_intens'], dtype=np.float32)

        # 处理频率轴
        if row['parsed_freqs'] is not None:
            val_freqs = np.array(row['parsed_freqs'], dtype=np.float32)
        else:
            # 如果没有频率，生成默认的 0到1
            val_freqs = np.linspace(0.0, 1.0, len(val_intens), dtype=np.float32)

        # 归一化检查 (如果是原始强度)
        if col_name == "IR_intens":
            mx = np.max(val_intens)
            if mx != 0:
                val_intens = val_intens / mx

        data_dict[name] = {
            'intens': val_intens,
            'freqs': val_freqs
        }

    return data_dict


def get_binned_features_trapz(intens_array, freqs_array, seg_len=SEG_LEN_POINTS):
    """
    对单条光谱进行分段积分（降维）。
    3600维 -> 24维 (如果 seg_len=150)
    """
    d = len(intens_array)
    if d % seg_len != 0:
        # 如果长度不是3600，简单截断或报错，这里假设都是3600
        new_len = (d // seg_len) * seg_len
        intens_array = intens_array[:new_len]
        freqs_array = freqs_array[:new_len]
        d = new_len

    n_bins = d // seg_len  # 3600//60=60
    # 重塑数组 (n_bins, seg_len)
    # 重塑数组: (段数, 每段点数)
    # 例如 3600 -> (24, 150)
    y_reshaped = intens_array.reshape(n_bins, seg_len)
    x_reshaped = freqs_array.reshape(n_bins, seg_len)

    # 对每一段做梯形积分
    # axis=1 表示沿着 150 个点的方向积分，得到 24 个面积值
    binned_feats = np.trapz(y_reshaped, x_reshaped, axis=1).astype(np.float32)

    return binned_feats  # 返回 numpy 数组


# ================= 主程序 =================

def main():
    # 1. 读取所有光谱数据到内存字典中
    cat_data = load_ir_data(CAT_CSV)
    lig_data = load_ir_data(LIG_CSV)

    print(f"已加载催化剂数据: {len(cat_data)} 个")
    print(f"已加载配体数据: {len(lig_data)} 个")

    # 2. 读取实验数据
    print(f"正在读取实验数据: {EXP_CSV} ...")
    df_exp = pd.read_csv(EXP_CSV, encoding='utf-8')  # 如果有乱码尝试 encoding='gbk'
    # df_exp = pd.read_excel(EXP_CSV)

    # 3. 准备结果列表
    final_data = []
    missing_log = []

    print("开始匹配并生成特征...")

    for idx, row in tqdm(df_exp.iterrows(), total=len(df_exp)):
        # 获取名称
        cat_name = str(row['Cat-name']).strip()
        lig_name = str(row['Matched_Ligand_Filename']).strip()

        # -------------------------------------------------------
        # [关键修改]：处理名称映射问题
        # 实验数据是 "Cat_X"，红外文件是 "cal_X"
        # 我们生成一个 cat_key 专门用于查字典
        cat_key = cat_name.replace("Cal_", "cal_")
        # -------------------------------------------------------

        # 构造 pair_name
        pair_name = f"{cat_name}-{lig_name}"

        # 检查是否存在对应的光谱
        if cat_key not in cat_data:
            # 记录缺失日志 (显示我们试图查找的 key)
            missing_log.append(f"Missing Cat IR: '{cat_name}' -> lookup '{cat_key}'")
            continue
        # if cat_name not in cat_ir_dict:
        #     missing_log.append(f"Missing Cat IR: {cat_name}")
        #     continue
        if lig_name not in lig_data:
            missing_log.append(f"Missing Ligand IR: {lig_name}")
            continue

        # 获取原始 3600维 光谱
        c_obj = cat_data[cat_key]
        l_obj = lig_data[lig_name]

        # [关键修改 2] 使用 trapz 计算特征，传入 freqs
        c_feat = get_binned_features_trapz(c_obj['intens'], c_obj['freqs'], SEG_LEN_POINTS)
        l_feat = get_binned_features_trapz(l_obj['intens'], l_obj['freqs'], SEG_LEN_POINTS)
        # 合并为 48维
        features_48d = np.concatenate([c_feat, l_feat])

        # '''不降维，直接合并cat-ligand对应的光谱强度'''
        # features_48d = np.concatenate([c_obj['intens'], l_obj['intens']])

        # 处理标签 Yields (去除百分号)
        '''
        对含有Yields和Selectivity实验数据的表格，运行下面的代码，不含有的则注释掉
        '''
        try:
            yield_str = str(row['Yields']).replace('%', '')
            y_yield = float(yield_str)
        except:
            y_yield = 0.0  # 或者 np.nan

        # 处理标签 Selectivity
        try:
            y_sel = float(row['Selectivity'])
        except:
            y_sel = 0.0

        # 存入字典
        entry = {
            'cat-ligand': pair_name,
            'Yields': y_yield,
            'Selectivity': y_sel
        }

        # 将 48 个特征展开为单独的列 (这是机器学习最标准的输入格式)
        # 如果你坚持要放在一列里存成列表，可以改写这里，但强烈建议展开
        for i in range(len(features_48d)):
            entry[f'feat_{i + 1}'] = features_48d[i]

        final_data.append(entry)

    # 4. 转换为 DataFrame 并保存
    if len(final_data) > 0:
        df_final = pd.DataFrame(final_data)

        # 调整列顺序：cat-ligand, feat_1...feat_48, Yields, Selectivity
        feat_cols = [f'feat_{i + 1}' for i in range(120)]  # 48 72 80 96 100 120 144 150 160 180 200
        cols_order = ['cat-ligand'] + feat_cols + ['Yields', 'Selectivity']
        # cols_order = ['cat-ligand'] + feat_cols
        df_final = df_final[cols_order]

        os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
        df_final.to_csv(OUT_CSV, index=False)
        print(f"\n成功! 文件已保存至: {OUT_CSV}")
        print(f"总样本数: {len(df_final)}")
    else:
        print("错误：没有匹配到任何数据。请检查名称是否一致。")

    # 打印缺失情况
    if missing_log:
        print(f"\n共发现 {len(missing_log)} 个匹配失败:")
        for msg in list(set(missing_log))[:10]:  # 只打印前10个
            print(msg)
        if len(missing_log) > 10: print("...")


if __name__ == "__main__":
    main()