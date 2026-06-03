import os
import shutil
import pandas as pd
from tqdm import tqdm

# ================= 配置路径 =================

# 1. 实验数据匹配结果文件 (包含 Matched_Ligand_Filename 列)
# CSV_PATH = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\2\predict_pairs_2.csv"
CSV_PATH = \
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\6new_pre_data\exp_data_matched-316-548-HJ.csv"
    # r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\5Cat_1+N_Ligand\exp_data_matched_updated-145.csv"

# 2. 目标文件夹 (你想把 log 文件复制到的地方)
DEST_DIR = r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\6new_pre_data\logs\ligand"

# 3. 源 LOG 文件夹列表
# 根据你的截图和之前的路径，我推测 LOGs 文件夹在 N_ligand_x 目录下
# 如果 P_ligand 也有 LOGs，请按同样的格式添加
SOURCE_LOG_DIRS = [
    # 原始库
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\N_ligand_1\LOGs",
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\N_ligand_2\LOGs",
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\N_ligand_3\LOGs",
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\N_ligand_4\LOGs",
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\N_ligand_5\LOGs",
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\N_ligand_6\LOGs",
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\P_ligand_1\LOGs",
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\P_ligand_2\LOGs",
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\P_ligand_3\LOGs",
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\P_ligand_Fe\LOGs",
    r"C:\Users\yxy\Desktop\hj-cooperation\BO-workflow\data\molecular\P_ligand_Fe\LOGs",
    # 第一次新增
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_1\new_ligand\N-ligand\LOGs",
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_1\new_ligand\P-ligand\LOGs",
    # 第二次新增（N-ligand\LOGs中含有predict中新增的1个分子）
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\new_ligand\N-ligand\LOGs",
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\new_ligand\O-ligand\LOGs",
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\new_ligand\P-ligand\LOGs",
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\new_ligand\P_ligand_Fe\LOGs",
    # 第四次新增
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\5Cat_1+N_Ligand\new_ligand\LOGs",  # 21个
    r"C:\Users\yxy\Desktop\hj-cooperation\ML-pred\data\round_2\predict\6new_pre_data\LOGs",  # 1个
]

# ================= 主程序 =================

def main():
    # 1. 准备目标文件夹
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"创建目标文件夹: {DEST_DIR}")
    else:
        print(f"目标文件夹已存在: {DEST_DIR}")

    # 2. 读取需要复制的文件名列表
    if not os.path.exists(CSV_PATH):
        print(f"错误: 找不到 CSV 文件 {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    # 提取 Matched_Ligand_Filename 列，去除空值，并去重
    if 'Matched_Ligand_Filename' not in df.columns:
        print("错误: CSV 中没有 'Matched_Ligand_Filename' 列")
        return

    target_filenames = df['Matched_Ligand_Filename'].dropna().unique()
    target_filenames = set(str(x).strip() for x in target_filenames)  # 转为集合，方便查找

    print(f"需要复制的唯一配体数量: {len(target_filenames)}")

    # 3. 扫描源文件夹，建立文件索引 {文件名(不含后缀): 完整路径}
    # 这样做比对每个目标文件都去遍历所有文件夹要快得多
    file_index = {}
    print("\n正在扫描源目录建立索引...")

    for folder in SOURCE_LOG_DIRS:
        if not os.path.exists(folder):
            print(f"[跳过] 路径不存在: {folder}")
            continue

        # 遍历文件夹下的文件
        for filename in os.listdir(folder):
            if filename.lower().endswith('.log'):
                # 获取文件名主体 (例如 "L001.log" -> "L001")
                stem = os.path.splitext(filename)[0]
                full_path = os.path.join(folder, filename)

                # 存入索引 (如果有重名文件，后面的会覆盖前面的，通常假设不同文件夹下没有重名ID)
                file_index[stem] = full_path

    print(f"索引建立完毕，共找到 {len(file_index)} 个 LOG 文件。")

    # 4. 开始复制
    copied_count = 0
    missing_files = []

    print("\n开始复制文件...")
    for target in tqdm(target_filenames, desc="Copying"):
        # 处理一下文件名，确保没有多余空格
        clean_target = target.strip()

        if clean_target in file_index:
            src_file = file_index[clean_target]
            # 目标路径
            dest_file = os.path.join(DEST_DIR, os.path.basename(src_file))

            try:
                shutil.copy2(src_file, dest_file)  # copy2 保留文件元数据(时间戳等)
                copied_count += 1
            except Exception as e:
                print(f"复制出错: {clean_target} -> {e}")
        else:
            missing_files.append(clean_target)

    # 5. 结果报告
    print("\n" + "=" * 40)
    print(f"处理完成")
    print(f"  - 目标清单数: {len(target_filenames)}")
    print(f"  - 成功复制数: {copied_count}")
    print(f"  - 缺失文件数: {len(missing_files)}")
    print("=" * 40)

    if missing_files:
        print("\n以下文件在 LOGs 文件夹中未找到 (可能只有 SDF 没有 LOG，或者文件名不匹配):")
        # 只打印前20个，防止刷屏
        for name in missing_files[:20]:
            print(f"  MISSING: {name}.log")
        if len(missing_files) > 20:
            print(f"  ... 以及其他 {len(missing_files) - 20} 个")

        # 保存缺失列表到文件，方便排查
        missing_log_path = os.path.join(DEST_DIR, "missing_logs_report.txt")
        with open(missing_log_path, "w") as f:
            for name in missing_files:
                f.write(f"{name}\n")
        print(f"\n完整缺失列表已保存至: {missing_log_path}")


if __name__ == '__main__':
    main()