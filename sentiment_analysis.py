from textblob import TextBlob

def analyze_english_sentiment(text):
    """
    分析英文文本的情感极性和主观性
    
    参数:
        text (str): 需要分析的英文文本
        
    返回:
        dict: 包含极性、主观性和情感判断的字典
    """
    # 创建TextBlob对象
    blob = TextBlob(text)
    
    # 获取情感极性 (-1 到 1之间，-1表示负面，1表示正面)
    polarity = blob.sentiment.polarity
    
    # 获取主观性 (0 到 1之间，0表示客观，1表示主观)
    subjectivity = blob.sentiment.subjectivity
    
    # 根据极性判断情感倾向
    if polarity > 0.1:
        sentiment = "Positive (正面)"
    elif polarity < -0.1:
        sentiment = "Negative (负面)"
    else:
        sentiment = "Neutral (中性)"
    
    return {
        "Text (文本)": text,
        "Polarity (极性)": polarity,
        "Subjectivity (主观性)": subjectivity,
        "Sentiment (情感倾向)": sentiment
    }

if __name__ == "__main__":
    # 示例英文文本
    sample_texts = [
        "I love this product! It's amazing and exceeded all my expectations.", # 主观性高，极性正面
        "The weather today seems quite bad.", # 主观性高，极性负面
        "Her speech was logically clear, rich in depth, and delivered fluently, leaving a deep impression on the audience.", # 主观性低，极性正面
        "The air quality in this area has always been poor, mainly caused by industrial pollution and vehicle exhaust emissions." # 主观性低，极性负面
    ]
    
    # 分析每个示例文本
    for i, text in enumerate(sample_texts, 1):
        print(f"=== Text {i} Analysis ===")
        result = analyze_english_sentiment(text)
        for key, value in result.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print()  # 空行分隔
    
    # 允许用户输入自定义英文文本进行分析
    print("Enter text to analyze (press Enter to exit):")
    while True:
        user_text = input("> ")
        if not user_text:
            break
        result = analyze_english_sentiment(user_text)
        print("\nAnalysis Result:")
        for key, value in result.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print()
    