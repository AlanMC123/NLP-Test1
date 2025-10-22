import thulac

# 初始化THULAC，seg_only=False表示同时进行分词和词性标注
thu = thulac.thulac(seg_only=False)

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
    
    # 使用THULAC进行分词和词性标注
    # 关键修正：text=False时返回(词语, 词性)元组列表，而非字符串
    tagged_result = thu.cut(text, text=False)  # 这里改为text=False
    
    # 分离分词结果和标注结果（确保每个元素都是二元组）
    segmented_text = []
    tagged_words = []
    for item in tagged_result:
        if len(item) == 2:  # 只处理包含词语和词性的有效结果
            word, flag = item
            segmented_text.append(word)
            tagged_words.append((word, flag))
    
    # 去除中文标点符号
    filtered_text = [word for word in segmented_text if word not in ['，', '。', '！', '？', '：', '；', '“', '”', '‘', '’', '（', '）', '《', '》', '、','…','—']]
    
    # 去除空格与换行符
    filtered_text = [word for word in filtered_text if word != '' and word != '\n']
    
    # 导出分词结果
    with open(f'seg_thulac/{filename}_filtered.txt', 'w', encoding='utf-8') as file:
        for word in filtered_text:
            file.write(word + ", ")
    
    # 导出标注结果（过滤停用词）
    with open(f'seg_thulac/{filename}_tagged.txt', 'w', encoding='utf-8') as file:
        for word, flag in tagged_words:
            if word in filtered_text:
                file.write(f"{word} ({flag})\n")
                
    