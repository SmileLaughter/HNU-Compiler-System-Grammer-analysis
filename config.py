# config.py
"""
项目配置文件
用于控制程序的默认行为
"""

# 配置模式枚举
class ConfigMode:
    ALWAYS_YES = 1  # 默认行为为是（无须用户手动输入）
    ALWAYS_NO = 2   # 默认行为为否（无须用户手动输入）
    ASK_USER = 3    # 由用户输入决定

# ==========================================
# 用户配置区
# ==========================================

# 1. 是否启用确定性模式 (DFA生成)
DETERMINISTIC_MODE_CONFIG = ConfigMode.ASK_USER

# 2. 是否生成DFA图片
GENERATE_DFA_IMAGE_CONFIG = ConfigMode.ALWAYS_YES

# 3. 是否将DFA导出成JSON格式
EXPORT_DFA_JSON_CONFIG = ConfigMode.ALWAYS_NO
