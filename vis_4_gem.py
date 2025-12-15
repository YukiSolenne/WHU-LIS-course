import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Sankey, Bar, HeatMap
from pyecharts.globals import ThemeType
from collections import Counter

# ==========================================
# 1. 读取数据
# ==========================================
file_name = 'raw.csv'
try:
    df = pd.read_csv(file_name, encoding='utf-8')
except:
    try:
        df = pd.read_csv(file_name, encoding='gbk')
    except:
        print("读取失败")

# 打印列名确认
print("CSV 文件的列名：", list(df.columns))

# ==========================================
# 2. 映射配置
# ==========================================

# 行为映射
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
    if 'E1' in text: return 'E1 情感投射型'
    if 'E2' in text: return 'E2 夸饰与依赖型'
    if 'E3' in text: return 'E3 冷静评价型'
    if 'E4' in text: return 'E4 模糊/矛盾表达型'
    if 'E5' in text: return 'E5 反讽型'
    return None

# ==========================================
# 3. 数据处理
# ==========================================
sankey_links = []
bar_data = []
heatmap_data_list = [] # 用于热力图

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
    
    # 构建数据
    if lis_items and emos:
        for lis in lis_items:
            for emo in emos:
                bar_data.append({"lis": lis, "emo": emo})
                # 为热力图准备数据 (Behavior vs Emotion)
                heatmap_data_list.append([lis, emo])
            if topics:
                for top in topics:
                    sankey_links.append((top, lis))
                    for emo in emos:
                        sankey_links.append((lis, emo))

# ==========================================
# 4. 绘图 (针对新标签优化)
# ==========================================

# 图1：堆叠柱状图
bar_df = pd.DataFrame(bar_data)
pivot_df = bar_df.groupby(['lis', 'emo']).size().unstack(fill_value=0)

desired_order = [
    'E1 情感投射型',
    'E3 冷静评价型',
    'E2 夸饰与依赖型',
    'E4 模糊/矛盾表达型',
    'E5 反讽型'
]
for col in desired_order:
    if col not in pivot_df.columns: pivot_df[col] = 0
pivot_df = pivot_df[desired_order]

bar = (
    Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="1200px", height="700px"))
    .add_xaxis(pivot_df.index.tolist())
    # 设置特定颜色映射：E1绿, E3粉, E2灰, E4蓝, E5红
    .add_yaxis("E1 情感投射型", pivot_df['E1 情感投射型'].tolist(), stack="stack1", color="#67C23A") # 绿
    .add_yaxis("E3 冷静评价型", pivot_df['E3 冷静评价型'].tolist(), stack="stack1", color="#E6A23C") # 橙
    .add_yaxis("E2 夸饰与依赖型", pivot_df['E2 夸饰与依赖型'].tolist(), stack="stack1", color="#909399") # 灰
    .add_yaxis("E4 模糊/矛盾表达型", pivot_df['E4 模糊/矛盾表达型'].tolist(), stack="stack1", color="#409EFF") # 蓝
    .add_yaxis("E5 反讽型", pivot_df['E5 反讽型'].tolist(), stack="stack1", color="#F56C6C") # 红
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="图1：不同信息行为下的情感功能分布", 
            subtitle="基于E1-E5功能性情感分类", 
            pos_top="1%",
            title_textstyle_opts=opts.TextStyleOpts(font_size=16, font_weight="bold"),
            subtitle_textstyle_opts=opts.TextStyleOpts(font_size=12)
        ),
        legend_opts=opts.LegendOpts(
            type_="scroll", 
            pos_top="10%", 
            item_gap=15,
            item_width=25,
            item_height=14
        ),
        xaxis_opts=opts.AxisOpts(
            axislabel_opts=opts.LabelOpts(rotate=25, font_size=11, interval=0, margin=15),
            name_location="middle",
            name_gap=35
        ),
        yaxis_opts=opts.AxisOpts(
            name="评论频次",
            name_location="middle",
            name_gap=50
        ),
    )
)
bar.render("1_功能情感堆叠图.html")

# 图2
link_counts = Counter(sankey_links)
nodes = [{"name": name} for name in set([x[0] for x in link_counts.keys()] + [x[1] for x in link_counts.keys()])]
links_dict = [{"source": s, "target": t, "value": v} for (s, t), v in link_counts.items()]

sankey = (
    Sankey(init_opts=opts.InitOpts(width="1600px", height="950px"))
    .add(
        "全链路逻辑",
        nodes,
        links_dict,
        linestyle_opt=opts.LineStyleOpts(opacity=0.3, curve=0.6, color="source"),
        label_opts=opts.LabelOpts(position="right", font_size=12, distance=8),
        node_gap=40,
        pos_top="12%",
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="图2：语境-行为-体验 完整因果链路图", 
            pos_top="3%",
            title_textstyle_opts=opts.TextStyleOpts(font_size=16, font_weight="bold")
        )
    )
)
sankey.render("2_全链路桑基图.html")

# 图3
heatmap_counts = Counter([(x[0], x[1]) for x in heatmap_data_list])
hm_data = [[list(lis_map.values()).index(k[0]), desired_order.index(k[1]), v] for k, v in heatmap_counts.items()]

heatmap = (
    HeatMap(init_opts=opts.InitOpts(width="1000px", height="600px"))
    .add_xaxis(list(lis_map.values()))
    .add_yaxis(
        "情感功能",
        desired_order,
        hm_data,
        label_opts=opts.LabelOpts(is_show=True, position="inside", font_size=10),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="图3：信息行为与情感功能共现热力图", 
            pos_top="3%",
            title_textstyle_opts=opts.TextStyleOpts(font_size=16, font_weight="bold")
        ),
        visualmap_opts=opts.VisualMapOpts(max_=15, pos_right="5%", pos_top="15%"),
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
heatmap.render("3_行为体验热力图.html")


print("请查看同文件夹下的三个 html 文件")
