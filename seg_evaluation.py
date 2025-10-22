def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        return [line.split(', ') for line in lines]
    
def single_evaluation(gold_file, test_file):
    gold_sentences = read_file(gold_file)
    test_sentences = read_file(test_file)

    C = E = M = 0
    total_gold_spans = 0
    total_len = 0
    total_words = 0

    for gold, test in zip(gold_sentences, test_sentences):
        total_len += sum(len(w) for w in test)
        total_words += len(test)

        def to_spans(words):
            spans = []
            start = 0
            for w in words:
                end = start + len(w)
                spans.append((start, end, w))
                start = end
            return spans

        gold_spans = set((s, e) for s, e, _ in to_spans(gold))
        test_spans = set((s, e) for s, e, _ in to_spans(test))

        C += len(gold_spans & test_spans)
        E += len(test_spans - gold_spans)
        M += len(gold_spans - test_spans)
        
        total_gold_spans += len(gold_spans)
        
    avg_word_len = total_len / total_words if total_words > 0 else 0
    
    return C, E, M, avg_word_len

def evaluate(file_type, file_path, seg_method):
    gold_file = f"{prefix_path}seg_gold_answer/{file_type}_answer.txt"
    test_file = f"{prefix_path}{file_path}"
    C, E, M, avg_len = single_evaluation(gold_file, test_file)
    
    precision = C / (C + E) if (C + E) > 0 else 0
    recall = C / (C + M) if (C + M) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{seg_method}对{file_type}的分词结果评价: ")
    print(f"正确分词数 C: {C}")
    print(f"错误分词数 E: {E}")
    print(f"遗漏分词数 M: {M}")
    print(f"正确率 (Precision): {precision:.2%}")
    print(f"召回率 (Recall): {recall:.2%}")
    print(f"F1值: {f1:.2%}")
    print(f"平均词长: {avg_len:.2f}")

if __name__ == "__main__":
    prefix_path = ""
    
    # jieba
    evaluate("news", "seg_jieba/01news_filtered.txt", "jieba")
    evaluate("passage", "seg_jieba/02passage_filtered.txt", "jieba")
    evaluate("poem", "seg_jieba/03poem_filtered.txt", "jieba")
    
    # snownlp
    evaluate("news", "seg_snownlp/01news_filtered.txt", "snownlp")
    evaluate("passage", "seg_snownlp/02passage_filtered.txt", "snownlp")
    evaluate("poem", "seg_snownlp/03poem_filtered.txt", "snownlp")
    
    # thulac
    evaluate("news", "seg_thulac/01news_filtered.txt", "thulac")
    evaluate("passage", "seg_thulac/02passage_filtered.txt", "thulac")
    evaluate("poem", "seg_thulac/03poem_filtered.txt", "thulac")
    
    # guwen - xunzi
    xunzi = 1
    
    print("\n\nXunzi - ", xunzi)
    xunzi += 1
    evaluate("guwen", "seg_xunzi/04guwen-1.txt", "xunzi")
    
    print("\n\nXunzi - ", xunzi)
    xunzi += 1
    evaluate("guwen", "seg_xunzi/04guwen-2.txt", "xunzi")
    
    print("\n\nXunzi - ", xunzi)
    xunzi += 1
    evaluate("guwen", "seg_xunzi/04guwen-3.txt", "xunzi")
    
    print("\n\nXunzi - ", xunzi)
    xunzi += 1
    evaluate("guwen", "seg_xunzi/04guwen-4.txt", "xunzi")
    
    print("\n\nXunzi - ", xunzi)
    xunzi += 1
    evaluate("guwen", "seg_xunzi/04guwen-5.txt", "xunzi")
    print("\n")
    
    # guwen - others
    
    evaluate("guwen", "seg_xunzi/04guwen_jieba.txt", "jieba")
    evaluate("guwen", "seg_xunzi/04guwen_snownlp.txt", "snownlp")
    evaluate("guwen", "seg_xunzi/04guwen_thulac.txt", "thulac")