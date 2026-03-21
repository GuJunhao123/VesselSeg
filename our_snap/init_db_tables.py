# our_snap/init_db_tables.py
import pandas as pd
import numpy as np
import os
from database import DatabaseManager

def create_and_import_tables():
    # === 关键修复：获取当前脚本所在的绝对路径 ===
    # 这样无论你在哪里运行命令，程序都能找到隔壁的Excel文件
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 拼接完整的文件路径
    file_patient = os.path.join(base_dir, '数据统计712.xlsx')
    file_indicators = os.path.join(base_dir, '检查检验指标.xlsx')

    print(f"📂 正在查找文件...")
    print(f"   -> 目标路径 1: {file_patient}")
    print(f"   -> 目标路径 2: {file_indicators}")

    # 检查文件到底在不在
    if not os.path.exists(file_patient):
        print(f"❌ 错误：找不到文件 '数据统计712.xlsx'")
        print("   请确认文件是否真的放在: " + base_dir)
        return
    if not os.path.exists(file_indicators):
        print(f"❌ 错误：找不到文件 '检查检验指标.xlsx'")
        print("   请确认文件是否真的放在: " + base_dir)
        return

    # 1. 获取数据库连接
    db = DatabaseManager()
    conn = db.get_connection()
    if not conn:
        print("❌ 严重错误：无法连接数据库，请检查 config.py 配置！")
        return
    
    cursor = conn.cursor()

    # =========================================================
    # 任务一：处理【数据统计712.xlsx】
    # =========================================================
    print("\n[1/2] 正在读取 '数据统计712.xlsx'...")
    try:
        df_patient = pd.read_excel(file_patient, sheet_name=0)
        df_patient = df_patient.replace({np.nan: None})

        cursor.execute("""
            IF OBJECT_ID('patient_info', 'U') IS NOT NULL DROP TABLE patient_info;
            CREATE TABLE patient_info (
                id INT PRIMARY KEY,
                patient_code NVARCHAR(100),
                name NVARCHAR(100),
                visit_type NVARCHAR(50),
                radiology_no NVARCHAR(100),
                visit_date NVARCHAR(50),
                lesion_type NVARCHAR(100),
                data_source NVARCHAR(100)
            )
        """)
        conn.commit()

        success_count = 0
        for index, row in df_patient.iterrows():
            pid = row.get('新编号')
            if pid is None: continue

            try:
                cursor.execute("""
                    INSERT INTO patient_info (id, patient_code, name, visit_type, radiology_no, visit_date, lesion_type, data_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    int(pid),
                    str(row.get('病人编号')) if row.get('病人编号') else None,
                    str(row.get('名称')) if row.get('名称') else None,
                    str(row.get('住院/门诊')) if row.get('住院/门诊') else None,
                    str(row.get('放射号')) if row.get('放射号') else None,
                    str(row.get('就诊时间')) if row.get('就诊时间') else None,
                    str(row.get('病变类型')) if row.get('病变类型') else None,
                    str(row.get('数据来源')) if row.get('数据来源') else None
                ))
                success_count += 1
            except Exception:
                pass # 跳过单行错误

        conn.commit()
        print(f"   ✅ 成功导入 {success_count} 条病人数据！")

    except Exception as e:
        print(f"   ❌ 任务一失败: {e}")

    # =========================================================
    # 任务二：处理【检查检验指标.xlsx】
    # =========================================================
    print("\n[2/2] 正在读取 '检查检验指标.xlsx'...")
    try:
        df_ind = pd.read_excel(file_indicators, sheet_name=0)
        df_ind = df_ind.replace({np.nan: None})

        cursor.execute("""
            IF OBJECT_ID('clinical_indicators', 'U') IS NOT NULL DROP TABLE clinical_indicators;
            CREATE TABLE clinical_indicators (
                id INT PRIMARY KEY,
                gender NVARCHAR(20),
                age NVARCHAR(20),
                afp NVARCHAR(50),
                cea NVARCHAR(50),
                ca50 NVARCHAR(50),
                ca199 NVARCHAR(50),
                ca125 NVARCHAR(50),
                ca242 NVARCHAR(50),
                ca724 NVARCHAR(50),
                ferritin NVARCHAR(50)
            )
        """)
        conn.commit()

        success_count = 0
        for index, row in df_ind.iterrows():
            cid = row.get('编号')
            if cid is None: continue

            try:
                def safe_str(val): return str(val).strip() if val is not None else None
                
                cursor.execute("""
                    INSERT INTO clinical_indicators (id, gender, age, afp, cea, ca50, ca199, ca125, ca242, ca724, ferritin)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    int(cid),
                    safe_str(row.get('性别')),
                    safe_str(row.get('年龄')),
                    safe_str(row.get('甲胎蛋白')),
                    safe_str(row.get('癌胚抗原')),
                    safe_str(row.get('糖类抗原50')),
                    safe_str(row.get('糖类抗原-199') or row.get('糖类抗原199')),
                    safe_str(row.get('糖类抗原-125') or row.get('糖类抗原125')),
                    safe_str(row.get('糖类抗原242')),
                    safe_str(row.get('糖类抗原724')),
                    safe_str(row.get('铁蛋白'))
                ))
                success_count += 1
            except Exception:
                pass

        conn.commit()
        print(f"   ✅ 成功导入 {success_count} 条检验指标数据！")

    except Exception as e:
        print(f"   ❌ 任务二失败: {e}")

    conn.close()

if __name__ == "__main__":
    create_and_import_tables()