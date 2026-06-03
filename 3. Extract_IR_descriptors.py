import linecache
import os
import pandas as pd
# import torch
import re
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Union


def ExtractLogSpectra(log_path: str, issues: Union[list, None] = None):
    """
    从一个log文件中获取红外光谱的 freqs, intens信息
    """
    if issues is None:
        issues = []
    freqs = []
    intens = []

    pattern_space = r"\s+"
    lines = linecache.getlines(log_path)
    meet_info = False

    for line in lines:
        line = line.strip()
        if line.startswith('Harmonic frequencies'):
            meet_info = True
            continue  # 跳到下一行
        if not meet_info:
            continue

        if line.startswith("Frequencies --"):
            for freq in re.split(pattern_space, line)[2:]:
                try:
                    freqs.append(float(freq))
                except ValueError:
                    issues.append(f"文件 {log_path} 中存在非数值形式的频率值: '{freq}'")

        elif line.startswith("IR Inten    --"):
            for inten in re.split(pattern_space, line)[3:]:
                try:
                    intens.append(float(inten))
                except ValueError:
                    issues.append(f"文件 {log_path} 中存在非数值形式的强度值: '{inten}'")
    return freqs, intens, issues


def lor_process(fs: List[float], ins: List[float], sigma: float = 5.0):
    """对提取的 freqs/intens 进行洛伦兹展宽"""
    """对提取的 freqs/intens 进行洛伦兹展宽 (使用 Numpy 替代 Torch)"""
    fs = np.array(fs, dtype=float)
    ins = np.array(ins, dtype=float)
    # 对应 torch.linspace(400, 4000, 3600)
    new_x = np.linspace(400, 4000, 3600)
    # 利用广播机制计算，对应 fs[:, None] - new_x[None, :]
    lx = fs[:, None] - new_x[None, :]
    ly = sigma / (lx ** 2 + sigma ** 2)
    maxm = np.max(ly)
    # 对应 torch.sum(..., dim=0)
    new_y = np.sum(ins[:, None] * ly / maxm, axis=0)

    return new_x.tolist(), new_y.tolist()


def normalize_spectrum(log_name: str, label: str, data):
    """   对光谱数据进行归一化处理，将数据缩放到 [0, 1]   """
    data = np.array(data, dtype=float)
    max_val = np.max(data) if data.size else 0.0
    if max_val == 0:
        print(f'{log_name}-{label}数值：np.max(data)==0')
        return data.tolist()
    return (data / max_val).tolist()   # ← 关键：转成 Python 列表


def _collect_all_logs(paths: Union[str, List[str]]) -> List[str]:
    """接受单个目录或目录列表，返回所有 .log 文件的绝对路径列表"""
    if isinstance(paths, str):
        paths = [paths]

    all_logs = []
    for p in paths:
        if not os.path.isdir(p):
            continue
        for f in os.listdir(p):
            if f.endswith(".log"):
                all_logs.append(os.path.join(p, f))
    return all_logs


def SaveLogInfoProcess(paths: Union[str, List[str]], save_path: str):
    """
    从一个或多个目录下批量提取 .log 文件的 IR 信息并合并保存
    - paths: str 或 [str, str, ...]
    - save_path: 输出 CSV 路径
    """
    log_paths = _collect_all_logs(paths)
    spectra_data = []
    issues = []

    with tqdm(total=len(log_paths), desc="Processing log files", unit="file") as pbar:
        for log_path in log_paths:
            pbar.update(1)
            log_name = os.path.basename(log_path)
            try:
                freqs, intens, local_issues = ExtractLogSpectra(log_path)
                # print(freqs)
                # print(intens)
                issues.extend(local_issues)

                if len(freqs) == 0 or len(intens) == 0:
                    issues.append(f"文件 {log_path} 未能解析到有效的频率/强度数据")
                    continue

                ir_freqs, ir_intens = lor_process(freqs, intens)
                ir_intens_normalized = normalize_spectrum(log_name, 'ir', ir_intens)

                spectra_data.append({
                    "Filename": os.path.splitext(log_name)[0],
                    "freqs": ir_freqs,
                    "IR_intens": ir_intens,
                    "IR_intens_normalized": ir_intens_normalized,
                })
            except Exception as e:
                issues.append(f"文件 {log_path} 处理时发生异常: {e}")

    spectra_df = pd.DataFrame(
        spectra_data,
        columns=["Filename", "freqs", "IR_intens", "IR_intens_normalized"]
    )
    spectra_df.to_csv(save_path, sep="\t", encoding="utf-8", index=False)

    if issues:
        print("\n==== 处理过程中发现的问题（汇总） ====")
        for msg in issues:
            print(msg)
    else:
        print("\n未发现问题。")


if __name__ == '__main__':
    # 849个配体的log文件路径：其中N配体共6个log文件路径:合计662个, P配体共3个log文件文件:合计163个, Fe-P配体仅一个路径:共24个
    N_1_path = r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\N_ligand_1\LOGs"
    N_2_path = r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\N_ligand_2\LOGs"
    N_3_path = r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\N_ligand_3\LOGs"
    N_4_path = r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\N_ligand_4\LOGs"
    N_5_path = r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\N_ligand_5\LOGs"
    N_6_path = r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\N_ligand_6\LOGs"
    P_1_path = r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\P_ligand_1\LOGs"
    P_2_path = r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\P_ligand_2\LOGs"
    P_3_path = r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\P_ligand_3\LOGs"
    P_Fe_path = r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\P_ligand_Fe\LOGs"
    # 第一次新增
    add_N_ligand_path_1 = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_1\new_ligand\N-ligand\LOGs"
    add_P_ligand_path_1 = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_1\new_ligand\P-ligand\LOGs"
    # 第二次新增
    add_N_ligand_path_2 = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\new_ligand\N-ligand\LOGs"
    add_O_ligand_path_2 = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\new_ligand\O-ligand\LOGs"
    add_P_ligand_path_2 = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\new_ligand\P-ligand\LOGs"
    add_P_Fe_ligand_path_2 = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\new_ligand\P_ligand_Fe\LOGs"
    # 第三次新增
    # add_N_ligand_path_3 = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\2\new_ligand\LOGs" 已经复制到add_N_ligand_path_2中
    # 第四次新增
    add_N_ligand_path_4 = \
        r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\5Cat_1+N_Ligand\new_ligand\LOGs"  # 21个
    add_P_ligand_path_4 = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\6new_pre_data\LOGs"  # 1个

    # 用法1：合并处理多个配体目录（以及可选的催化剂目录）
    all_dirs = [
        N_1_path, N_2_path, N_3_path, N_4_path, N_5_path, N_6_path,
        P_1_path, P_2_path, P_3_path, P_Fe_path,
        add_N_ligand_path_1, add_P_ligand_path_1,
        add_N_ligand_path_2, add_O_ligand_path_2, add_P_ligand_path_2, add_P_Fe_ligand_path_2,
        add_N_ligand_path_4, add_P_ligand_path_4,
    ]
    SaveLogInfoProcess(all_dirs, r'C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\IrSpecInfo_ligand_945.csv')

    # # 用法2：只处理单个文件夹
    # # ligand_path = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\logs\ligand"
    # # SaveLogInfoProcess(ligand_path,
    #                    # r'C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\IrSpecInfo_ligand_used.csv')
    #
    # cat_path = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\logs\cat"
    # SaveLogInfoProcess(cat_path,
    #                    r'C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\IrSpecInfo_catalyst_used.csv')
