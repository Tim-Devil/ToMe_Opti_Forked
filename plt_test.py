import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'

r_values = [128, 256, 512, 768, 1024]
grouped_matching_memory = [338.38, 332.38, 332.38, 344.38, 356.42]
bipartite_matching_memory = [482.38, 476.38, 476.38, 488.38, 500.38]

savings_percentage = [(bipartite - grouped) / bipartite * 100 
                     for grouped, bipartite in zip(grouped_matching_memory, bipartite_matching_memory)]

plt.figure(figsize=(10, 6))

x = np.arange(len(r_values))
width = 0.35

plt.bar(x - width/2, grouped_matching_memory, width, label='Grouped Matching (chunk=2)')
plt.bar(x + width/2, bipartite_matching_memory, width, label='Bipartite Matching')

# 添加数据标签
for i, v in enumerate(grouped_matching_memory):
    plt.text(i - width/2, v + 5, f'{v}', ha='center')

for i, v in enumerate(bipartite_matching_memory):
    plt.text(i + width/2, v + 5, f'{v}', ha='center')
    
# 在图表顶部添加节约百分比标签
for i, pct in enumerate(savings_percentage):
    plt.text(i, max(bipartite_matching_memory) + 15, f'节约 {pct:.1f}%', 
             ha='center', fontweight='bold', color='green')

# 设置图表标题和标签
plt.title('内存占用对比: Grouped Matching vs Bipartite Matching', fontsize=15)
plt.xlabel('r 值', fontsize=12)
plt.ylabel('内存占用 (MB)', fontsize=12)

# 添加平均节约百分比信息
avg_saving = sum(savings_percentage) / len(savings_percentage)
plt.figtext(0.5, 0.01, f'平均节约显存: {avg_saving:.1f}%', ha='center', 
            fontsize=12, fontweight='bold', bbox=dict(facecolor='lightgreen', alpha=0.5))
plt.xticks(x, r_values)
plt.legend()

# 调整y轴范围，使对比更明显并为百分比标签留出空间
plt.ylim(0, 600)

# 添加网格线，使图表更易读
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存图表
plt.savefig('memory_comparison.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()