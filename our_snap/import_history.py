import os
import re
from docx import Document
from database import DatabaseManager

# === 配置 ===
DOC_PATH = "bingshi.docx"  # 你的病史文件路径
DB_MANAGER = DatabaseManager()

def parse_docx_to_dict(doc_path):
    """
    读取 Word 文档，提取 {姓名: 病史文本} 的字典
    """
    if not os.path.exists(doc_path):
        print(f"❌ 找不到文件: {doc_path}")
        return {}

    doc = Document(doc_path)
    patient_history_map = {}
    current_name = None
    current_text = []

    # 正则表达式匹配姓名 (根据你的文档格式： "姓名:  ,赵才林")
    # 匹配逻辑：找 "姓名"，忽略中间的冒号、空格、逗号，提取后面的名字
    name_pattern = re.compile(r"姓名[:：\s,，]*(\S+)")

    full_text = [p.text for p in doc.paragraphs if p.text.strip()]
    
    for line in full_text:
        # 尝试在这一行里找名字
        match = name_pattern.search(line)
        if match:
            # 如果之前已经有正在记录的病人，先保存上一个
            if current_name:
                patient_history_map[current_name] = "\n".join(current_text)
            
            # 开始新病人
            current_name = match.group(1).replace(",", "").strip() # 去掉可能残留的逗号
            current_text = [line] # 把包含名字的这一行也放进去作为开头
            print(f"🔍 发现病人: {current_name}")
        else:
            # 如果当前有正在记录的病人，就追加文本
            if current_name:
                current_text.append(line)
    
    # 保存最后一个病人
    if current_name:
        patient_history_map[current_name] = "\n".join(current_text)
        
    return patient_history_map

def distribute_history_files():
    """
    将提取的文本保存为 history.txt 到对应的 saved_records 文件夹
    """
    # 1. 解析 Word
    print("正在解析病史文档...")
    history_data = parse_docx_to_dict(DOC_PATH)
    print(f"共提取到 {len(history_data)} 份病史。\n")

    # 2. 获取数据库里所有的病人 (我们需要知道 姓名 -> 文件夹路径 的关系)
    # 因为文件夹是用 ID 命名的 (P00xxxx)，所以必须查库才能对应上名字
    print("正在查询数据库匹配病人...")
    conn = DB_MANAGER.get_connection()
    cursor = conn.cursor()
    
    # 查出所有病人的 code, name, original_path
    cursor.execute("SELECT patient_code, patient_name, original_path FROM segmentation_records")
    records = cursor.fetchall()
    
    matched_count = 0
    
    for row in records:
        p_code = row[0].strip()
        p_name = row[1].strip()
        org_path = row[2] # 比如 D:\...\saved_records\P001\xxx.nii.gz
        
        # 获取该病人的存储目录 (D:\...\saved_records\P001)
        patient_folder = os.path.dirname(org_path)
        
        # 检查我们在 Word 里有没有这个人的病史
        # 模糊匹配：如果 Word 里的名字包含数据库里的名字，或者反过来
        target_history = None
        for doc_name, text in history_data.items():
            if p_name in doc_name or doc_name in p_name:
                target_history = text
                break
        
        if target_history:
            # 保存为 txt
            txt_path = os.path.join(patient_folder, "history.txt")
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(target_history)
                print(f"✅ [匹配成功] {p_name} -> {txt_path}")
                matched_count += 1
            except Exception as e:
                print(f"❌ 写入失败 {p_name}: {e}")
        else:
            print(f"⚠️ 未找到病史: {p_name}")

    print(f"\n🎉 处理完成！共生成 {matched_count} 个 history.txt 文件。")

if __name__ == "__main__":
    distribute_history_files()