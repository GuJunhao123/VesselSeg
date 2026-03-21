import shutil
print("正在检查 shutil...")
print(f"文件位置: {shutil.__file__}")

try:
    print(f"功能检查: {shutil.get_terminal_size}")
    print("✅ shutil 看起来没问题。")
except AttributeError:
    print("❌ 严重问题：你的系统 shutil.py 文件里丢失了 get_terminal_size 函数！")
    print("   -> 请去 D:\\python\\Lib\\shutil.py 看看是不是被你改坏了。")