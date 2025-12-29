import json
import time
from pathlib import Path

import pandas as pd
import requests

# ================== 配置区 ==================

INPUT_FILE = "raw1.csv"
OUTPUT_FILE = "raw_labeled.csv"

# 列名配置
TEXT_COL = "raw text"
LABEL_COL = "DM_Dimension"
ISSUE_COL = "具体问题"
SENTIMENT_COL = "Sentiment"

# 批量大小
BATCH_SIZE = 10

# 本地 API 配置
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = "" # 填自己的api key
MODEL_NAME = "anthropic/claude-haiku-4.5" 
# MODEL_NAME = "omnidimen-4b-emotion"
# API_URL = "http://127.0.0.1:1234/v1/chat/completions"
# API_KEY = ""

# 是否在终端打印每次的模型输出
DEBUG_PRINT_RESPONSE = True

# ================== 提示词（改进版）==================

SYSTEM_PROMPT = """
你是一个信息资源与数据管理领域的“信息服务与信息用户”研究助理，负责对用户评论进行 DeLone & McLean (2003) 信息系统成功模型的定向内容分析编码。

任务：将每条评论归入 SQ / IQ / SEQ / NB / Other 中的一个类别，并给出一个可分析的“主题桶（issue）”与情感倾向（sentiment）。

总原则（非常重要）：
1) **优先归入 SQ/IQ/SEQ/NB，尽量减少 Other。** 只有完全无关（广告/乱码/纯表情无语义）才用 Other。
2) **单条评论只选一个 category。** 若涉及多个点，选择“评论中最核心、最主要的抱怨/表扬点”作为分类依据。
3) **issue 必须从下方预定义主题桶中选择一个（精确匹配字符串）。** 不允许自造 issue。
4) “情感”不要扩展为心理诊断；如需表达关系体验，用“关系强弱/关系连续性/边界与可控”等中性术语。

-------------------------
分类定义（D&M 2003）
-------------------------

A) SQ (System Quality - 系统质量)
关注：系统与产品作为信息系统的技术与运行质量。
包括但不限于：
- 稳定性：崩溃/闪退/卡顿/延迟/发热/耗电/无法登录/消息丢失/同步失败
- 可用性与维护：**停服、跑路、倒闭、不再更新、公司不行了**（视作系统持续可用性极差）
- 功能与界面：功能缺失、UI 设计、广告干扰、充值不到账/订单异常（属于系统功能/交易流程缺陷）
注意：
- 单纯说“贵”“不值”“性价比低”不算 SQ，归 NB（除非明确指向充值系统故障）

SQ 的 issue 主题桶（只能从以下选一项）：
- "SQ_Stability"        （崩溃/闪退/不可用/停服跑路）
- "SQ_Performance"      （卡顿/延迟/发热耗电）
- "SQ_Function_UI"       （功能缺失/界面/广告/体验流程）
- "SQ_Payment_System"    （充值不到账/订单异常/付费功能故障）
- "SQ_Data_Loss"         （数据/聊天记录/资产丢失、同步失败）

B) IQ (Information Quality - 信息质量)
关注：AI 输出内容作为“信息/建议/文本”的质量。
包括但不限于：
- 相关性与对题：答非所问、抓不住需求点
- 准确性与可信度：编造、幻觉、事实错误
- 上下文一致性：记不住设定、前后矛盾、理解偏差、OOC/崩人设
- 表达质量：复读、空泛、逻辑混乱、语言贫乏
- 可执行性：建议无法落地、缺少步骤、不可操作

IQ 的 issue 主题桶（只能从以下选一项）：
- "IQ_Relevance"         （答非所问/不相关）
- "IQ_Accuracy"          （错误/编造/幻觉）
- "IQ_Context_Consistency"（记忆问题/前后矛盾/OOC）
- "IQ_Clarity_Logic"     （逻辑不通/表达混乱/复读空泛）
- "IQ_Actionability"     （建议不可执行/缺少可操作步骤）

C) SEQ (Service Quality - 服务质量；以“关系性信息服务”口径表达)
关注：系统提供的信息服务过程与保障体验，而非文本本身。
强调“关系性服务质量”（弱化情绪词，使用中性表述），包括但不限于：
- 关系连续性/定位一致：互动是否稳定、角色是否一致、服务是否可预测
- 边界与尊重：越界、冒犯、强推、引导不当（强调服务规范与边界）
- 可控与透明：可解释、可纠错、可撤回、记忆管理可见性、权限与控制感
- 审核/限制体验：锁文、敏感词拦截、模板化限制影响服务获得
- 客服与运营：客服处理、申诉、官方运营活动与服务响应

SEQ 的 issue 主题桶（只能从以下选一项）：
- "SEQ_Relational_Continuity" （关系连续性/定位一致/可预测）
- "SEQ_Boundary_Respect"       （边界/尊重/越界/不当引导）
- "SEQ_Controllability_Transparency"（可控/透明/可解释/可纠错）
- "SEQ_Moderation_Limits"      （审核机制/敏感词/锁文限制）
- "SEQ_CustomerOps"            （客服/运营活动/响应保障）

D) NB (Net Benefits - 净效益/用户感受与后果)
关注：用户从使用中获得的收益、成本与后果（正负皆可），包括但不限于：
- 性价比/价格感受：太贵、不值、值回票价
- 时间与注意力成本：打发时间、电子榨菜、上头/耗时间
- 关系强弱与依赖后果：离不开、替代现实互动、影响作息（用中性“依赖/替代/使用后果”表述）
- 总体收益：帮助我、支持我、缓解压力、提高效率（不做心理诊断）

NB 的 issue 主题桶（只能从以下选一项）：
- "NB_Price_Value"        （贵/不值/性价比）
- "NB_Time_Attention"     （打发时间/耗时/上头）
- "NB_Dependence_Substitution"（依赖/替代/使用后果）
- "NB_Overall_Benefit"    （总体收益/帮助/支持/效率）
- "NB_Risk_Concern"       （风险担忧：隐私、安全、被操控等总体担忧）

E) Other
仅用于：乱码、广告、与产品/服务完全无关、无法判断语义。
Other 的 issue 主题桶固定为：
- "Other_Noise"

-------------------------
情感倾向（sentiment）
-------------------------
- Positive：总体表达认可/满意/收益为主
- Negative：总体表达不满/抱怨/损失为主
- Neutral：仅陈述事实或信息不足以判断（尽量少用 Neutral）

输出要求：
- category：SQ / IQ / SEQ / NB / Other
- issue：必须是上面定义的主题桶字符串之一
- sentiment：Positive / Negative / Neutral
"""

USER_TEMPLATE = """
请为下面的用户评论按 D&M(2003) 维度进行单标签分类，并给出主题桶与情感倾向。只输出 JSON 数组，不要解释。

可选 category：SQ、IQ、SEQ、NB、Other。
issue 必须从系统提示中给定的主题桶中选择，严格匹配字符串。
sentiment：Positive / Negative / Neutral（尽量避免 Neutral）。

现在这批共有 {n} 条评论，从 0 开始编号。

评论列表：
{items}

请输出一个 JSON 数组，每个元素形如：
{{
  "index": 0,
  "category": "IQ",
  "issue": "IQ_Context_Consistency",
  "sentiment": "Negative"
}}

只输出 JSON，不要额外解释。
"""



# ================== 调用模型 ==================

def call_model(text_list):
    """
    text_list: [ (row_index, text), ... ]
    返回: dict { row_index: {"category": ..., "issue": ..., "sentiment": ...}, ... }
    """
    # 组装用户内容
    items_str_lines = []
    for local_idx, (_, text) in enumerate(text_list):
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

    # LM Studio 也应返回 choices
    if "choices" not in data:
        print("错误：响应中缺少 choices 字段，原始响应：", data)
        raise KeyError("choices")

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
            raise ValueError("模型返回的JSON格式无效")
        try:
            result_list = json.loads(content[start:end + 1])
        except json.JSONDecodeError as e:
            print("错误：提取的 JSON 片段仍然无法解析")
            print("提取的片段：", content[start:end + 1])
            raise ValueError(f"JSON 解析失败: {e}")

    mapped = {}
    for item in result_list:
        local_idx = item["index"]
        if local_idx < 0 or local_idx >= len(text_list):
            print(f"警告：模型返回的索引 {local_idx} 超出范围（0-{len(text_list) - 1}），跳过此项")
            continue
        category = item.get("category", "").strip()
        issue = item.get("issue", "").strip()
        sentiment = item.get("sentiment", "").strip()
        row_idx = text_list[local_idx][0]
        mapped[row_idx] = {
            "category": category,
            "issue": issue,
            "sentiment": sentiment,
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
        df[LABEL_COL] = df[LABEL_COL].astype(str).replace("nan", "")
    if ISSUE_COL not in df.columns:
        df[ISSUE_COL] = ""
    else:
        df[ISSUE_COL] = df[ISSUE_COL].astype(str).replace("nan", "")
    if SENTIMENT_COL not in df.columns:
        df[SENTIMENT_COL] = ""
    else:
        df[SENTIMENT_COL] = df[SENTIMENT_COL].astype(str).replace("nan", "")

    # 文本非空且标签为空
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

        if not mapped:
            print("警告：本批次未获得任何有效标注结果，跳过保存")
            continue

        expected_indices = set(i for i, _ in text_batch)
        got_indices = set(mapped.keys())
        missing = expected_indices - got_indices
        if missing:
            print(f"警告：共有 {len(missing)} 项未获得标注结果：{sorted(missing)}")

        for row_idx, info in mapped.items():
            df.at[row_idx, LABEL_COL] = info["category"]
            df.at[row_idx, ISSUE_COL] = info["issue"]
            df.at[row_idx, SENTIMENT_COL] = info["sentiment"]

        df.to_csv(out_path, index=False, encoding="gbk")
        print(f"已保存到：{out_path}")

        time.sleep(1)

    print("\n已完成")
    print("检查最终文件：", out_path)


if __name__ == "__main__":
    main()
