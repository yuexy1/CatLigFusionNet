import os
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

# ================= 配置路径 =================
# 这里填入你存放SDF文件的目录（参考了你之前的代码路径）
LIGAND_DIRS = [
    # 注意：请确认是SDFs目录还是上级目录，需指向含.sdf的文件夹
    # 最初的化学空间
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\N_ligand_1\SDFs",
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\N_ligand_2\SDFs\renamed_sdfs",
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\N_ligand_3\SDFs\renamed_sdfs",
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\N_ligand_4\SDFs\renamed_sdfs",
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\N_ligand_5\SDFs\renamed_sdfs",
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\N_ligand_6\SDFs\renamed_sdfs",
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\P_ligand_1\SDFs",
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\P_ligand_2\SDFs\renamed_sdfs",
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\P_ligand_3\SDFs\renamed_sdfs",
    # 第一次新增
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_1\new_ligand\N-ligand\SDFs\renamed_sdfs",
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_1\new_ligand\P-ligand\SDFs\renamed_sdfs",
    # 第二次新增
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\new_ligand\N-ligand\SDFs\renamed_sdfs",
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\new_ligand\O-ligand\SDFs",
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\new_ligand\P-ligand\SDFs\renamed_sdfs",
    # 第三次新增 不在predict_1的数据集中，因此不需要识别
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\2\new_ligand\SDFs",  # N_713
    # 第四次新增
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\5Cat_1+N_Ligand\SDFs-10\renamed_sdfs",  # 10个
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\5Cat_1+N_Ligand\SDFs-11\renamed_sdfs",  # 11个
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\6new_pre_data\SDFs",  # 1个
]

# 实验数据文件路径
# EXP_DATA_PATH = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\1\predict_pairs_1.csv"
EXP_DATA_PATH = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\exp_data-316-548-HJ.csv"

# 输出结果路径
OUTPUT_DB_PATH = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\all_ligands_smiles_db_3+4.csv"  # 新增21个N配体
OUTPUT_MATCHED_PATH = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\exp_data_matched-316-548-HJ.csv"


# ================= 辅助函数 =================

def get_canonical_smiles(mol):
    """将RDKit Mol对象转换为标准Canonical SMILES"""
    if mol is None:
        return None
    try:
        # 使用 isomericSmiles=True 以保留手性信息
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except:
        return None


def standardize_smiles_string(smiles_str):
    """将SMILES字符串读取并重新标准化"""
    if pd.isna(smiles_str) or str(smiles_str).strip() == "":
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles_str))
        if mol:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except:
        pass
    return None


def collect_sdf_info(dirs):
    """遍历文件夹收集所有SDF的SMILES和文件名"""
    database = []
    failed_files = []  # 用于记录失败的文件
    total_files_detected = 0

    print("正在扫描 SDF 文件...")

    # # 1. 收集所有 sdf 文件的绝对路径
    sdf_files = []
    for d in dirs:
        if not os.path.exists(d):
            print(f"[警告] 路径不存在: {d}，请检查路径配置。")
            continue
        # 递归或非递归查找，根据实际文件夹结构，这里假设SDF就在该目录下或renamed_sdfs子目录下
        # 这里使用 os.walk 做深度遍历以防万一
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith('.sdf'):
                    sdf_files.append(os.path.join(root, f))

    total_files_detected = len(sdf_files)
    print(f"扫描完毕，共发现 {total_files_detected} 个 SDF 文件。开始解析...")

    for file_path in tqdm(sdf_files, desc="Parsing SDFs"):
        filename = os.path.basename(file_path)
        name_stem = os.path.splitext(filename)[0]  # 去除 .sdf 后缀，作为 ID

        mol = None
        error_reason = "Unknown"

        # 读取 SDF
        try:
            # 方法A: 尝试作为单个分子读取 (默认 strict 解析)
            # removeHs=False 尝试保留氢看是否读取成功，通常保持默认即可
            mol = Chem.MolFromMolFile(file_path)

            # 方法B: 如果A失败，尝试使用 Supplier (容错率稍微高一点，或者处理多分子文件)
            if mol is None:
                suppl = Chem.SDMolSupplier(file_path)
                if len(suppl) > 0:
                    # SDMolSupplier 即使有长度，取出的分子也可能是 None (如果解析失败)
                    mol = suppl[0]
                    if mol is None:
                        error_reason = "SDMolSupplier returned None (Structure Error)"
                else:
                    error_reason = "SDMolSupplier found no molecules (Empty File?)"

            # 方法C: 最后的尝试 - 关闭清洗 (Sanitize=False)
            # 这通常能读入结构，但生成的 SMILES 可能不标准，慎用，这里仅作调试用
            if mol is None:
                try:
                    mol = Chem.MolFromMolFile(file_path, sanitize=False)
                    if mol:
                        # 尝试手动清洗，看具体哪步报错
                        try:
                            Chem.SanitizeMol(mol)
                        except Exception as e:
                            error_reason = f"Sanitization Failed: {e}"
                            mol = None  # 既然清洗失败，视为无效，不录入数据库
                except:
                    pass

            if mol:
                smi = get_canonical_smiles(mol)
                if smi:
                    database.append({
                        "Calc_Filename": name_stem,
                        "Calc_SMILES": smi,
                        "File_Path": file_path
                    })
                else:
                    failed_files.append((filename, "SMILES Generation Failed"))

            else:
                # 如果没有被覆盖，记录默认原因
                if error_reason == "Unknown":
                    error_reason = "Parse Failed (Format or Valence Error)"
                failed_files.append((filename, error_reason))

        except Exception as e:
            failed_files.append((filename, f"Exception: {str(e)}"))
            print(f"[异常] 处理 {filename} 时出错: {e}")

    # 3. 输出统计报告
    success_count = len(database)
    print("\n" + "=" * 40)
    print(f"解析报告:")
    print(f"  - 发现 SDF 文件总数: {total_files_detected}")
    print(f"  - 成功提取 SMILES数: {success_count}")
    print(f"  - 失败/跳过 文件数 : {len(failed_files)}")
    print("=" * 40)

    if failed_files:
        print("\n[!] 以下文件解析失败，请检查结构：")
        for name, reason in failed_files:
            print(f"  ❌ {name} -> 原因: {reason}")
        print("\n建议: 用 ChemDraw 或 Avogadro 打开这些文件，检查是否有红色的价态错误或空文件。")
    print("=" * 40 + "\n")

    return pd.DataFrame(database)


# ================= 主程序 =================

def main():
    # 1. 构建计算配体的数据库 (Filename <-> SMILES)
    df_calc = collect_sdf_info(LIGAND_DIRS)

    # 保存一份中间结果，方便核查
    print(f"共提取到 {len(df_calc)} 个计算配体结构。")  # 不包括Fe  849-24=825个N/P配体smiles信息 新增27个N+P配体信息
    df_calc.to_csv(OUTPUT_DB_PATH, index=False, encoding='utf-8-sig')

    # 2. 读取实验数据
    if not os.path.exists(EXP_DATA_PATH):
        print(f"找不到实验数据文件: {EXP_DATA_PATH}")
        return

    df_exp = pd.read_csv(EXP_DATA_PATH, encoding='utf-8-sig')
    # 如果读取乱码，尝试 encoding='gbk'

    print("正在匹配实验数据...")

    # 3. 为实验数据添加一列标准化的SMILES
    tqdm.pandas(desc="Standardizing Exp SMILES")
    # 注意：如果SMILES为空(如Fe配体)，这里会返回None
    df_exp['Std_Exp_SMILES'] = df_exp['Ligand-Smiles'].progress_apply(standardize_smiles_string)

    # 4. 进行匹配
    matched_filenames = []
    match_types = []

    # 字典1: SMILES -> Filename (注意处理重复SMILES的情况，如果有异构体)
    # 既然要做一一对应，假设计算库里SMILES是唯一的。如果有重复，这里默认取第一个。
    smiles_to_filename = dict(zip(df_calc['Calc_SMILES'], df_calc['Calc_Filename']))

    # 集合: 所有的文件名 (用于标记验证)
    available_filenames = set(df_calc['Calc_Filename'])

    for index, row in df_exp.iterrows():
        exp_smi = row['Std_Exp_SMILES']
        exp_name = str(row['Ligand-name']).strip()

        found_name = None
        match_type = "None"

        # --- 策略 1: 基于 SMILES 匹配 (针对 N/P 配体) ---
        if exp_smi is not None:
            if exp_smi in smiles_to_filename:
                found_name = smiles_to_filename[exp_smi]
                match_type = "SMILES_Match"
            else:
                match_type = "SMILES_Not_Found_In_DB"

        # --- 策略 2: 直接使用 Ligand-name (针对 Fe 配体等无SMILES情况) ---
        # 修改说明：当没有 SMILES 且有 Ligand-name 时，不再检查 available_filenames，
        # 直接信任并使用 csv 中的 Ligand-name。
        elif exp_name and exp_name != 'nan' and exp_name != '':
            found_name = exp_name
            # 顺便标记一下这个名字是否在我们的SDF库里（仅作参考，不影响赋值）
            if exp_name in available_filenames:
                match_type = "Name_Direct_Verified"  # 在SDF库里找到了同名文件
            else:
                match_type = "Name_Direct_Unverified"  # 直接用了名字，但在SDF库里没找到对应文件(如Fe系列)

        else:
            match_type = "No_Info"

        matched_filenames.append(found_name)
        match_types.append(match_type)

    df_exp['Matched_Ligand_Filename'] = matched_filenames
    df_exp['Match_Status'] = match_types

    # 5. 保存结果
    df_exp.to_csv(OUTPUT_MATCHED_PATH, index=False, encoding='utf-8-sig')

    # 6. 统计报告
    total = len(df_exp)
    success = df_exp['Matched_Ligand_Filename'].notna().sum()
    print("\n" + "=" * 30)
    print(f"匹配完成！结果已保存至 {OUTPUT_MATCHED_PATH}")
    print(f"总数据条数: {total}")
    print(f"成功匹配数: {success}")
    print(f"失败匹配数: {total - success}")
    print("=" * 30)


if __name__ == '__main__':
    main()