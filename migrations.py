import matplotlib.pyplot as plt

# 表格数据（包括表头）
data = [
    ["consumption/ services", "Random", "Sandpiper", "Metis", "Mpc"],
    ["10service", 10, 9, 9, 9],
    ["15service", 15, 12, 12, 8],
    ["20service", 19, 14, 15, 10]
]

# 创建图形
fig, ax = plt.subplots(figsize=(9, 2.5))
ax.axis('tight')
ax.axis('off')

# 创建表格
table = ax.table(
    cellText=data,
    loc='center',
    cellLoc='center',
    edges='closed'
)

# 设置样式
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)

# 加粗表头行和表头列
for j in range(len(data[0])):  # 第一行（表头）
    table[(0, j)].set_facecolor('#d0e0ff')
    table[(0, j)].set_text_props(weight='bold')

for i in range(1, len(data)):  # 第一列（行标签）
    table[(i, 0)].set_facecolor('#f0f0f0')
    table[(i, 0)].set_text_props(weight='bold')

# 特别强调左上角单元格（可选）
table[(0, 0)].set_facecolor('#c0d8ff')

plt.tight_layout()
plt.savefig('consumption.png', dpi=300, bbox_inches='tight')
plt.show()