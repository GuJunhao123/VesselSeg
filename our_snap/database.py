# our_snap_new/database.py
import os
import json
import shutil
import traceback
from datetime import datetime
from config import Config

class DatabaseManager:
    def __init__(self):
        self.log_file = Config.CASE_LOG_FILE
        self.save_dir = Config.SAVE_DIR
        self._init_log_file()

    def _init_log_file(self):
        if not os.path.exists(self.log_file):
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("")

    def get_connection(self):
        return None

    def save_record(self, p_code, p_name, current_img_path, mask_save_path, note=""):
        p_code = str(p_code).strip()
        p_name = str(p_name).strip()

        try:
            patient_dir = os.path.join(self.save_dir, p_code)
            if not os.path.exists(patient_dir):
                os.makedirs(patient_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            is_folder = os.path.isdir(current_img_path)
            target_org_path = ""

            if is_folder:
                new_folder_name = f"{timestamp}_dicom_series"
                target_org_path = os.path.join(patient_dir, new_folder_name)

                if os.path.exists(target_org_path):
                    shutil.rmtree(target_org_path)

                print(f"   正在复制 DICOM 文件夹...")
                shutil.copytree(current_img_path, target_org_path)

            else:
                src_dir, src_filename = os.path.split(current_img_path)
                filename_base, src_ext = os.path.splitext(src_filename)

                final_ext = ".nii.gz"

                if src_ext == "" or src_ext.lower() == ".dcm":
                    final_ext = ".dcm"
                elif src_ext.lower() == ".nii":
                    final_ext = ".nii"
                elif src_ext.lower() == ".gz":
                    final_ext = ".nii.gz"
                else:
                    final_ext = src_ext

                new_filename = f"{timestamp}_org{final_ext}"
                target_org_path = os.path.join(patient_dir, new_filename)

                print(f"   正在复制文件 (后缀识别为 {final_ext})...")
                shutil.copy2(current_img_path, target_org_path)

            new_mask_name = f"{timestamp}_mask.nii.gz"
            target_mask_path = os.path.join(patient_dir, new_mask_name)
            shutil.copy2(mask_save_path, target_mask_path)

            record = {
                "id": timestamp,
                "patient_code": p_code,
                "patient_name": p_name,
                "original_path": target_org_path,
                "mask_path": target_mask_path,
                "diagnosis_note": note,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"✅ [Success] 归档成功！病例编号: {p_code}, 姓名: {p_name}")
            return True

        except Exception as e:
            print(f"❌ [Error] 归档失败: {e}")
            traceback.print_exc()
            return False

    def get_patient_history(self, keyword):
        records = []

        if not os.path.exists(self.log_file):
            return records

        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if (keyword and
                            (keyword.lower() in record.get('patient_name', '').lower() or
                             keyword.lower() in record.get('patient_code', '').lower())):
                            records.append(record)
                        elif not keyword:
                            records.append(record)
                    except json.JSONDecodeError:
                        continue

            records.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            return records

        except Exception as e:
            print(f"❌ 查询历史记录失败: {e}")
            return []

    def get_all_records(self):
        return self.get_patient_history("")
