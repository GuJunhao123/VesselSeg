# our_snap_new/verify_fix.py
import os
import sys

# 强制将当前脚本所在的目录加入 Python 搜索路径，确保能 import database
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def verify_system():
    print(f"🔍 正在进行系统自检... (当前目录: {current_dir})\n")
    
    # === 1. 检查代码是否更新 ===
    target_file = os.path.join(current_dir, 'database.py')
    print(f"[1/3] 检查 database.py 代码逻辑... \n      目标文件: {target_file}")
    
    if not os.path.exists(target_file):
        print("  ❌ 严重错误：找不到 database.py 文件！")
        print("     请确认 verify_fix.py 和 database.py 都在同一个文件夹里。")
        return

    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键函数是否都在
        has_isdir = 'os.path.isdir' in content
        has_copytree = 'shutil.copytree' in content
        
        if has_isdir and has_copytree:
            print("  ✅ 代码已修复：检测到支持文件夹复制的逻辑 (copytree)。")
        else:
            print("  ❌ 代码未修复：你的 save_record 函数里似乎还没加上“判断文件夹”的代码。")
            print("     -> 请回到 database.py，把 save_record 函数替换为最新版本。")
    except Exception as e:
        print(f"  ❌ 无法读取文件: {e}")

    # === 2. 检查数据库连接 ===
    print("\n[2/3] 检查数据库连接...")
    try:
        from database import DatabaseManager
        db = DatabaseManager()
        conn = db.get_connection()
        
        if not conn:
            print("  ❌ 连接失败：请检查 config.py 配置。")
            return
        else:
            print("  ✅ 连接成功：已连上 SQL Server。")
            
            # === 3. 检查表是否存在 ===
            print("\n[3/3] 检查归档表 (segmentation_records)...")
            cursor = conn.cursor()
            try:
                # 尝试查询这张表
                cursor.execute("SELECT count(*) FROM segmentation_records")
                count = cursor.fetchone()[0]
                print(f"  ✅ 表结构正常！当前已有 {count} 条历史记录。")
            except Exception as e:
                print(f"  ⚠️ 表不存在或有错误: {e}")
                print("  🔧 正在尝试自动修复表结构...")
                try:
                    db.init_tables() # 调用你的建表函数
                    print("  ✅ 修复成功！表已自动创建。")
                except Exception as fix_err:
                    print(f"  ❌ 修复失败: {fix_err}")
            finally:
                conn.close()

    except ImportError:
        print("  ❌ 导入失败：无法加载 database 模块，请检查文件名是否正确。")
    except Exception as e:
        print(f"  ❌ 发生未知错误: {e}")

    print("\n------------------------------------------------")
    print("🎉 总结：如果上面全是绿色 ✅，请重启软件再次尝试保存！")

if __name__ == "__main__":
    verify_system()