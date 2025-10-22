import jieba
from snownlp import SnowNLP
import thulac

# 初始化thulac分词器
thu = thulac.thulac(seg_only=True)  # 只进行分词，不进行词性标注

# 要处理的文件名
filename = "04guwen"
filepath = f"{filename}.txt"

# 读取文件内容
with open(filepath, 'r', encoding='utf-8') as file:
    text = file.read()

# 定义标点符号集合
punctuations = ['，', '。', '！', '？', '：', '；', '“', '”', '‘', '’', '（', '）', '《', '》', '、', '…', '—']

# 1. 使用jieba分词
jieba_seg = jieba.lcut(text)
# 过滤标点和空字符
jieba_filtered = [word for word in jieba_seg if word not in punctuations and word.strip()]
# 保存结果
with open(f'seg_xunzi/{filename}_jieba.txt', 'w', encoding='utf-8') as file:
    file.write(", ".join(jieba_filtered))

# 2. 使用snownlp分词
s = SnowNLP(text)
snownlp_seg = s.words
# 过滤标点和空字符
snownlp_filtered = [word for word in snownlp_seg if word not in punctuations and word.strip()]
# 保存结果
with open(f'seg_xunzi/{filename}_snownlp.txt', 'w', encoding='utf-8') as file:
    file.write(", ".join(snownlp_filtered))

# 3. 使用thulac分词
thulac_seg = thu.cut(text, text=True).split()  # 获取分词结果并转换为列表
# 过滤标点和空字符
thulac_filtered = [word for word in thulac_seg if word not in punctuations and word.strip()]
# 保存结果
with open(f'seg_xunzi/{filename}_thulac.txt', 'w', encoding='utf-8') as file:
    file.write(", ".join(thulac_filtered))

print("分词完成，结果已保存到seg_xunzi文件夹")