"""配置中心：Key、模型、主服务地址、持久化路径。"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
GOALS_DIR = os.path.join(OUTPUT_DIR, "goals")
MEMORY_FILE = os.path.join(OUTPUT_DIR, "memory.json")

# 大模型（主 agent 大脑），与主服务共用同一套环境变量
LLM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("UPSTREAM_API_KEY", "sk-e76068d4dd5f4da9a8da51094ff09d91")
LLM_MODEL = os.getenv("UPSTREAM_MODEL", "deepseek-v4-flash")
LLM_TIMEOUT = float(os.getenv("HARNESS_LLM_TIMEOUT", "60"))

# 主业务服务（船检 classify SSE 服务）
MAIN_BASE = os.getenv("HARNESS_MAIN_BASE", "http://127.0.0.1:8000")

# harness 持久状态
STATE_FILE = os.getenv("HARNESS_STATE_FILE", "harness_state.json")

os.makedirs(GOALS_DIR, exist_ok=True)
