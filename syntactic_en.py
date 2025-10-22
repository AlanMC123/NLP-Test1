#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from stanfordcorenlp import StanfordCoreNLP
from graphviz import Digraph

# Stanford CoreNLP 路径（请修改为你解压后的目录）
CORENLP_DIR = r"F:/LanguageProcessing/corenlp_model/stanford-corenlp-4.5.7"
VISUALIZATION_DIR = "syntactic_en"

# 添加 Graphviz 路径（根据实际安装修改）
os.environ["PATH"] += os.pathsep + r"D:/SoftwareFiles/Graphviz/bin"

def load_corenlp():
    """加载 Stanford CoreNLP"""
    try:
        nlp = StanfordCoreNLP(CORENLP_DIR, lang='en')
        print("Stanford CoreNLP 加载成功！")
        return nlp
    except Exception as e:
        print(f"CoreNLP 加载失败: {e}")
        return None

def read_text_file(file_path):
    """读取英文文本"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在！")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    print(f"读取文件成功，共 {len(text)} 字符。")
    return text

def segment_sentences(nlp, text):
    """使用 CoreNLP 分句"""
    props = {
        'annotators': 'ssplit',
        'pipelineLanguage': 'en',
        'outputFormat': 'json'
    }
    ann = nlp.annotate(text, properties=props)
    ann_json = json.loads(ann)
    sentences = [ " ".join([token["word"] for token in s["tokens"]]) 
                  for s in ann_json["sentences"] ]
    print(f"分句完成，共 {len(sentences)} 个句子。")
    return sentences

def visualize_dependency_tree(words, arcs, sentence_idx, output_dir):
    """依存关系可视化"""
    os.makedirs(output_dir, exist_ok=True)
    dot = Digraph(name=f"sentence_{sentence_idx}", format='png')
    dot.attr(rankdir='TB', fontname='Arial')

    for i, word in enumerate(words):
        dot.node(str(i+1), f"{word}\n(ID:{i+1})", shape='box', style='filled', color='lightblue')

    dot.node('0', 'ROOT', shape='ellipse', style='filled', color='lightgreen')

    for head_idx, rel, dep_idx in arcs:
        dot.edge(str(head_idx), str(dep_idx), label=rel, fontname='Arial')

    output_path = os.path.join(output_dir, f"sentence_{sentence_idx}_dep_tree")
    try:
        dot.render(output_path, view=False, cleanup=True)
        print(f"依存关系图已保存至: {output_path}.png")
    except Exception as e:
        print(f"生成可视化图像失败: {e}")
        dot.save(output_path + '.dot')

def syntactic_analysis(nlp, sentences):
    """句法分析：分词、词性、依存"""
    results = []
    for i, sent in enumerate(sentences, 1):
        print(f"\n--- 句子 {i}: {sent} ---")
        words = nlp.word_tokenize(sent)
        pos_tags = nlp.pos_tag(sent)
        dependencies = nlp.dependency_parse(sent)

        dep_result = [(head, rel, dep) for (rel, head, dep) in dependencies]

        print(f"分词: {words}")
        print(f"词性: {pos_tags}")
        print(f"依存关系: {dep_result}")

        visualize_dependency_tree(words, dep_result, i, VISUALIZATION_DIR)

        results.append({
            'sentence': sent,
            'seg': words,
            'pos': pos_tags,
            'dep': dep_result
        })
    return results

def save_results(results, output_file):
    """保存结果"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(f"Sentence: {res['sentence']}\n")
            f.write(f"Tokens: {' '.join(res['seg'])}\n")
            f.write(f"POS: {' '.join([p for _, p in res['pos']])}\n")
            f.write(f"Dependency: {res['dep']}\n\n")
    print(f"结果已保存到 {output_file}")

def analyze(input_file):
    nlp = load_corenlp()
    if not nlp:
        return
    try:
        text = read_text_file(f"{input_file}")
    except FileNotFoundError as e:
        print(e)
        return
    sentences = segment_sentences(nlp, text)
    results = syntactic_analysis(nlp, sentences)
    save_results(results, f"syntactic_en/ana_{input_file}")

if __name__ == "__main__":
    analyze("syntactic_en.txt")
