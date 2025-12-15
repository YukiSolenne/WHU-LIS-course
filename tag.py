import pandas as pd
import requests
import json
import time
from pathlib import Path

# ================== 配置区 ==================

# 数据文件路径
INPUT_FILE = "raw1.csv"
OUTPUT_FILE = "raw_labeled.csv"

# 列名配置
TEXT_COL = "raw text"
LABEL_COL = "情绪功能E"

# 批量大小
BATCH_SIZE = 10

# 本地 API 配置
API_URL = "https://openrouter.ai/api/v1/chat/completions"  
API_KEY = "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"  

MODEL_NAME = "anthropic/claude-3.7-sonnet"  # 比如 "gpt-4.1-mini" / "qwen2.5-14b" / "lmstudio-xxx"

# 是否在终端打印每次的模型输出
DEBUG_PRINT_RESPONSE = True

# ================== 提示词 ==================

SYSTEM_PROMPT = """
你是一个研究用的标注助手，任务是为每条用户评论标注“情绪功能类别”。

共有五类情绪功能（E1–E5），根据“这段文本中情绪在互动中的作用”来判断，而不是简单的积极/消极：
请你根据评论的语言表达方式，判断其情绪表达类型（可多选）：

E1 情感投射型：用户将主观情绪投射到AI身上，拟人化、情感化地看待它；
E2 夸饰与依赖型：评论中使用强烈情绪词、夸张语气，表现依赖、喜欢、情绪沉浸；
E3 冷静评价型：语气理性、平稳，对AI进行理性评价或体验反馈；
E4 模糊/矛盾表达型：评论中情绪表达不明确、态度暧昧，或正负情绪交织；
E5 反讽型：使用反话、调侃、戏谑语气表达真实态度，表面与实际情绪相反。


输出时，请根据每条评论的整体语气和功能，选择 **一个最主要的 E 类别**。
"""

USER_TEMPLATE = """
请为下面的用户评论标注“情绪功能类别”。只输出 JSON，严格遵守格式。

五类标签：
- E1: 情感投射型
- E2: 夸饰与依赖型
- E3: 冷静评价型
- E4: 模糊/矛盾表达型
- E5: 反讽型

示例：
评论：她每天都在我最累的时候出现，我有点怕，但又想她别走。

分析：该评论语气中带有复杂情绪。用户表达出对AI陪伴的渴望（“别走”），同时也流露出戒备或不安（“有点怕”），情绪中存在矛盾张力，未明确是正向还是负向态度，属于“矛盾/反讽型”表达。

输出：E4


现在这批共有 {n} 条评论，从 0 开始编号。

评论列表：
{items}

请输出一个 JSON 数组，每个元素形如：
{{
  "index": 0,            // 对应上面的编号
  "emotion_words": "简短的情绪关键词（比如 依恋/担忧/怀疑 等）",
  "category": "E3"       // E1/E2/E3/E4/E5 之一
}}

只输出 JSON，不要多余解释。
"""


# ================== 调用模型函数 ==================

def call_model(text_list):
    """
    text_list: [ (row_index, text), ... ]
    返回: dict { row_index: {"emotion_words": ..., "category": ...}, ... }
    """
    # 组装用户内容
    items_str_lines = []
    for local_idx, (row_idx, text) in enumerate(text_list):
        one = f"[{local_idx}] {text.replace(chr(10), ' ')}"
        items_str_lines.append(one)
    items_str = "\n".join(items_str_lines)

    user_prompt = USER_TEMPLATE.format(n=len(text_list), items=items_str)

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        "temperature": 0.1,
    }

    resp = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    resp.raise_for_status()
    data = resp.json()

    # 兼容 OpenAI/兼容接口的结构
    content = data["choices"][0]["message"]["content"]
    if DEBUG_PRINT_RESPONSE:
        print("=== raw model output ===")
        print(content)
        print("========================")

    # 解析 JSON
    try:
        result_list = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("[")
        end = content.rfind("]")
        if start == -1 or end == -1 or start >= end:
            print("错误：无法从模型输出中提取 JSON 数组")
            print("原始输出：", content)
            raise ValueError("模型返回的 JSON 格式无效")
        try:
            result_list = json.loads(content[start:end+1])
        except json.JSONDecodeError as e:
            print("错误：提取的 JSON 片段仍然无法解析")
            print("提取的片段：", content[start:end+1])
            raise ValueError(f"JSON 解析失败: {e}")

    mapped = {}
    for item in result_list:
        local_idx = item["index"]
        if local_idx < 0 or local_idx >= len(text_list):
            print(f"警告：模型返回的索引 {local_idx} 超出范围（0-{len(text_list)-1}），跳过此项")
            continue
        emotion_words = item.get("emotion_words", "").strip()
        category = item.get("category", "").strip()
        row_idx = text_list[local_idx][0]
        mapped[row_idx] = {
            "emotion_words": emotion_words,
            "category": category,
        }

    return mapped


# ================== main ==================

def main():
    base_dir = Path(__file__).resolve().parent
    in_path = base_dir / INPUT_FILE
    out_path = base_dir / OUTPUT_FILE

    print("读取：", in_path)
    df = pd.read_csv(in_path, encoding="gbk")  

    if TEXT_COL not in df.columns:
        raise KeyError(f"找不到文本列：{TEXT_COL}")

    if LABEL_COL not in df.columns:
        df[LABEL_COL] = ""
    else:
        df[LABEL_COL] = df[LABEL_COL].astype(str)
        df[LABEL_COL] = df[LABEL_COL].replace('nan', '')

    # 找出需要标注的行：文本非空且标签为空
    mask_need = df[TEXT_COL].notna() & (df[LABEL_COL].isna() | (df[LABEL_COL] == ""))
    idx_list = df[mask_need].index.tolist()

    print(f"需要标注的样本数：{len(idx_list)}")

    for start in range(0, len(idx_list), BATCH_SIZE):
        batch_idx = idx_list[start:start + BATCH_SIZE]
        text_batch = []
        for i in batch_idx:
            text = str(df.at[i, TEXT_COL])
            text_batch.append((i, text))

        print(f"\n>>> 处理第 {start} - {start + len(batch_idx) - 1} 行，共 {len(batch_idx)} 条")

        if not text_batch:
            print("跳过空批次")
            continue

        try:
            mapped = call_model(text_batch)
        except Exception as e:
            print("调用模型出错：", e)
            print("稍等 5 秒后继续……")
            time.sleep(5)
            continue

        # 写回 DataFrame
        if not mapped:
            print("警告：本批次未获得任何有效标注结果，跳过保存")
            continue
        
        # 检查是否有遗漏的项
        expected_indices = set(i for i, _ in text_batch)
        got_indices = set(mapped.keys())
        missing = expected_indices - got_indices
        if missing:
            print(f"警告：以下 {len(missing)} 项未获得标注结果：{sorted(missing)}")
        
        for row_idx, info in mapped.items():
            df.at[row_idx, LABEL_COL] = info["category"]
            # （可选）
            df.at[row_idx, "情绪关键词"] = info["emotion_words"]

        df.to_csv(out_path, index=False, encoding="gbk")
        print(f"已保存到：{out_path}")

        time.sleep(1)

    print("\n已完成 ")
    print("检查最终文件：", out_path)


if __name__ == "__main__":
    main()
