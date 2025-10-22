import jieba
import jieba.posseg as pseg

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

    # 使用jieba分词
    segmented_text = jieba.lcut(text)
    
    # 去除中文标点符号
    filtered_text = [word for word in segmented_text if word not in ['，', '。', '！', '？', '：', '；', '“', '”', '‘', '’', '（', '）', '《', '》', '、','…','—']]
    
    # 去除空格与换行符
    filtered_text = [word for word in filtered_text if word != '' and word != '\n']
    
    # 将分词结果导出到文件
    with open(f'seg_jieba/{filename}_filtered.txt', 'w', encoding='utf-8') as file:
        for word in filtered_text:
            file.write(word + ", ")

    # 词性标注
    words = pseg.cut(text)
    
    # 将标注结果导出到文件
    with open(f'seg_jieba/{filename}_tagged.txt', 'w', encoding='utf-8') as file:
        for word, flag in words:
            file.write(f"{word} ({flag})\n")

