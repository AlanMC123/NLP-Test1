from snownlp import SnowNLP
import string

for filenum in range(1, 4):
    filename = ""
    if filenum == 1:
        filename = "01news"
    elif filenum == 2:
        filename = "02passage"
    else:
        filename = "03poem"
    
    # 读取文件
    filepath = f"{filename}.txt"
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # 初始化SnowNLP对象
    s = SnowNLP(text)
    
    # 分词
    segmented_text = s.words
    
    # 去除中文标点符号
    filtered_text = [word for word in segmented_text if word not in ['，', '。', '！', '？', '：', '；', '“', '”', '‘', '’', '（', '）', '《', '》', '、','…','—']]
    
    # 去除空格与换行符
    filtered_text = [word for word in filtered_text if word != '' and word != '\n']
    
    # 导出分词结果
    with open(f'seg_snownlp/{filename}_filtered.txt', 'w', encoding='utf-8') as file:
        for word in filtered_text:
            file.write(word + ", ")
    
    # 词性标注（SnowNLP的tag方法返回词性标注结果）
    tagged_words = s.tags
    
    # 导出标注结果
    with open(f'seg_snownlp/{filename}_tagged.txt', 'w', encoding='utf-8') as file:
        for word, flag in tagged_words:
            # 过滤掉空字符的标注结果
            if word.strip() not in ['', '\n', '\t']:
                file.write(f"{word} ({flag})\n")
