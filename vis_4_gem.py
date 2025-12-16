import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import HeatMap
from pyecharts.globals import ThemeType
from collections import Counter

# ==========================================
# 1. 读取数据
# ==========================================
file_name = 'raw.csv' # 你的新文件名
try:
    df = pd.read_csv(file_name, encoding='utf-8')
except:
    try:
        df = pd.read_csv(file_name, encoding='gbk')
    except:
        print("❌ 读取失败，请检查文件路径")

# 打印列名确认
print("CSV 文件的列名：", list(df.columns))

# ==========================================
# 2. 映射配置
# ==========================================

# LIS行为映射
lis_map = {
    'I1': 'I1 需求表达', 'I2': 'I2 系统性能', 'I3': 'I3 交互策略',
    'I4': 'I4 信任质量', 'I5': 'I5 隐私安全', 'I6': 'I6 商业模式'
}

# Topic语境映射
topic_map = {
    'T1': 'T1 情感/依附', 'T2': 'T2 系统/信任', 'T3': 'T3 隐私/安全',
    'T4': 'T4 商业/操纵', 'T5': 'T5 主体重构'
}

# E标签清洗逻辑 (提取 E1, E2...)
def clean_emotion(text):
    text = str(text).strip()
    # 如果包含 E1, E2 等关键词
    if 'E1' in text: return 'E1 情感投射型'
    if 'E2' in text: return 'E2 夸饰与依赖型'
    if 'E3' in text: return 'E3 冷静评价型'
    if 'E4' in text: return 'E4 模糊/矛盾表达型'
    if 'E5' in text: return 'E5 反讽型'
    return None

# ==========================================
# 3. 数据处理
# ==========================================
topic_behavior_data = []  # 用于主题-行为共现热力图
behavior_emotion_data = []  # 用于行为-情感热力图

for _, row in df.iterrows():
    # 清洗 Topic（对应 CSV 中的 'Topic' 列）
    raw_topics = str(row['Topic']).replace('、', ',').replace('，', ',').split(',')
    topics = []
    for t in raw_topics:
        code = t.strip()[:2].upper()
        if code in topic_map: topics.append(topic_map[code])
            
    # 清洗 LIS Behavior（对应 CSV 中的 'Information' 列）
    raw_lis = str(row['Information']).replace('、', ',').replace('，', ',').split(',')
    lis_items = []
    for l in raw_lis:
        code = l.strip()[:2].upper()
        if code in lis_map: lis_items.append(lis_map[code])
            
    # 清洗 Emotion（对应 CSV 中的 'Emotion' 列）
    raw_emos = str(row['Emotion']).replace('、', ',').replace('，', ',').split(',')
    emos = []
    for e in raw_emos:
        clean_e = clean_emotion(e)
        if clean_e: emos.append(clean_e)
    
    # 构建主题-行为共现数据
    if topics and lis_items:
        for top in topics:
            for lis in lis_items:
                topic_behavior_data.append([top, lis])
    
    # 构建行为-情感共现数据
    if lis_items and emos:
        for lis in lis_items:
            for emo in emos:
                behavior_emotion_data.append([lis, emo])

# ==========================================
# 4. 绘图 (针对新标签优化)
# ==========================================

# 确保情感顺序
emotion_order = [
    'E1 情感投射型',
    'E3 冷静评价型',
    'E2 夸饰与依赖型',
    'E4 模糊/矛盾表达型',
    'E5 反讽型'
]

# --- 图1：主题-行为共现热力图 ---
topic_behavior_counts = Counter([(x[0], x[1]) for x in topic_behavior_data])
topic_list = list(topic_map.values())
behavior_list = list(lis_map.values())

# 构建热力图数据：[(x_index, y_index, value), ...]
tb_hm_data = []
for (topic, behavior), count in topic_behavior_counts.items():
    x_idx = topic_list.index(topic)
    y_idx = behavior_list.index(behavior)
    tb_hm_data.append([x_idx, y_idx, count])

topic_behavior_heatmap = (
    HeatMap(init_opts=opts.InitOpts(width="1200px", height="700px"))
    .add_xaxis(topic_list)
    .add_yaxis(
        "信息行为",
        behavior_list,
        tb_hm_data,
        label_opts=opts.LabelOpts(is_show=True, position="inside", font_size=10),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="图3：主题语境与信息行为共现热力图", 
            pos_top="3%",
            title_textstyle_opts=opts.TextStyleOpts(font_size=16, font_weight="bold")
        ),
        visualmap_opts=opts.VisualMapOpts(
            max_=max([v for v in topic_behavior_counts.values()]) if topic_behavior_counts else 10,
            pos_right="5%", 
            pos_top="15%"
        ),
        xaxis_opts=opts.AxisOpts(
            axislabel_opts=opts.LabelOpts(interval=0, font_size=11, margin=15, rotate=25),
            name_location="middle",
            name_gap=30
        ),
        yaxis_opts=opts.AxisOpts(
            axislabel_opts=opts.LabelOpts(font_size=11),
            name_location="middle",
            name_gap=50
        ),
    )
)
topic_behavior_heatmap.render("1_主题-行为共现热力图.html")

# --- 图2：行为-情感共现热力图 ---
behavior_emotion_counts = Counter([(x[0], x[1]) for x in behavior_emotion_data])

# 构建热力图数据
be_hm_data = []
for (behavior, emotion), count in behavior_emotion_counts.items():
    x_idx = behavior_list.index(behavior)
    y_idx = emotion_order.index(emotion)
    be_hm_data.append([x_idx, y_idx, count])

behavior_emotion_heatmap = (
    HeatMap(init_opts=opts.InitOpts(width="1000px", height="600px"))
    .add_xaxis(behavior_list)
    .add_yaxis(
        "情感功能",
        emotion_order,
        be_hm_data,
        label_opts=opts.LabelOpts(is_show=True, position="inside", font_size=10),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="图2：信息行为与情感功能共现热力图", 
            pos_top="3%",
            title_textstyle_opts=opts.TextStyleOpts(font_size=16, font_weight="bold")
        ),
        visualmap_opts=opts.VisualMapOpts(
            max_=max([v for v in behavior_emotion_counts.values()]) if behavior_emotion_counts else 15,
            pos_right="5%", 
            pos_top="15%"
        ),
        xaxis_opts=opts.AxisOpts(
            axislabel_opts=opts.LabelOpts(interval=0, font_size=11, margin=15),
            name_location="middle",
            name_gap=30
        ),
        yaxis_opts=opts.AxisOpts(
            axislabel_opts=opts.LabelOpts(font_size=11),
            name_location="middle",
            name_gap=50
        ),
    )
)
behavior_emotion_heatmap.render("2_行为-情感共现热力图.html")

print("✅ 两张图表已生成！请查看 html 文件。")