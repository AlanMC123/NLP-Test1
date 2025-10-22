#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from ltp import LTP
from graphviz import Digraph

# 配置
MODEL_DIR = "F:/LanguageProcessing/ltp_models/tiny"
VISUALIZATION_DIR = "syntactic_zh"

# 添加 Graphviz 路径配置（根据你的实际安装路径修改）
os.environ["PATH"] += os.pathsep + r"D:/SoftwareFiles/Graphviz/bin"

def load_ltp_model():
    """加载LTP模型"""
    try:
        ltp = LTP(MODEL_DIR)
        print("LTP模型加载成功！")
        return ltp
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请检查PyTorch和LTP安装，或手动下载模型。")
        return None

def read_text_file(file_path):
    """读取TXT文件内容"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在！请确保文件在当前目录。")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    print(f"读取文件成功，共 {len(text)} 字符。")
    return text

def segment_sentences(ltp, text):
    """使用LTP进行分句"""
    # 适配LTP 4.x版本的分句方法
    outputs = ltp.pipeline([text], tasks=["cws"])
    words = outputs.cws[0]
    full_text = ''.join(words)
    import re
    pattern = r'([^。？！；,.?!;]*[。？！；,.?!;])'
    sentences = re.findall(pattern, full_text)
    sentences = [s.strip() for s in sentences if s.strip() and not re.fullmatch(r'[。？！；,.?!;]+', s.strip())]
    print(f"分句完成，共 {len(sentences)} 个句子。")
    return sentences

def visualize_dependency_tree(words, arcs, sentence_idx, output_dir):
    """使用Graphviz可视化依存关系树"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建 Digraph 对象时指定中文字体
    dot = Digraph(name=f"sentence_{sentence_idx}", format='png')
    
    # 设置图形属性，包括中文字体
    dot.attr(rankdir='TB')  # 从上到下的布局
    dot.attr('graph', fontname='SimHei')  # 图形字体
    dot.attr('node', fontname='SimHei')   # 节点字体
    dot.attr('edge', fontname='SimHei')   # 边字体
    
    # 添加节点 - 使用支持中文的字体
    for i, word in enumerate(words):
        dot.node(str(i+1), f"{word}\n(ID:{i+1})", 
                shape='box', 
                style='filled', 
                color='lightblue',
                fontname='SimHei')  # 明确指定节点字体
    
    # 添加根节点
    dot.node('0', 'ROOT', 
             shape='ellipse', 
             style='filled', 
             color='lightgreen',
             fontname='SimHei')
    
    # 添加边 - 确保关系标签也使用中文字体
    for arc in arcs:
        head_idx, rel, dep_idx = arc
        dot.edge(str(head_idx), str(dep_idx), label=rel, fontname='SimHei')
    
    output_path = os.path.join(output_dir, f"sentence_{sentence_idx}_dep_tree")
    
    try:
        dot.render(output_path, view=False, cleanup=True)
        print(f"依存关系图已保存至: {output_path}.png")
    except Exception as e:
        print(f"生成可视化图像失败: {e}")
        print("请确保已安装 Graphviz 软件并将其添加到系统 PATH")
        # 保存 dot 文件以便手动生成
        dot.save(output_path + '.dot')
        print(f"已保存 dot 文件: {output_path}.dot")
        print("可以使用命令手动生成: dot -Tpng {output_path}.dot -o {output_path}.png")

def syntactic_analysis(ltp, sentences):
    """进行句法分析：分词、词性标注、依存分析"""
    results = []
    for i, sent in enumerate(sentences, 1):
        print(f"\n--- 句子 {i}: {sent} ---")
        
        # 统一使用pipeline获取分析结果（适配LTP 4.x）
        outputs = ltp.pipeline([sent], tasks=["cws", "pos", "dep"])
        seg_result = outputs.cws[0]
        pos_result = outputs.pos[0]
        dep_heads = outputs.dep[0]['head']
        dep_labels = outputs.dep[0]['label']
        dep_result = [(dep_heads[j], dep_labels[j], j+1) for j in range(len(seg_result))]
        
        print(f"分词: {seg_result}")
        print(f"词性: {pos_result}")
        print(f"依存关系: {dep_result}")
        
        print("依存树（树状表示）:")
        print_dependency_tree(seg_result, dep_result)
        
        visualize_dependency_tree(seg_result, dep_result, i, VISUALIZATION_DIR)
        
        results.append({
            'sentence': sent,
            'seg': seg_result,
            'pos': pos_result,
            'dep': dep_result
        })
    
    return results

def print_dependency_tree(words, arcs):
    """打印依存树（仅接收words和arcs两个参数）"""
    # 从arcs中提取根节点（head=0的节点）
    root_idx = next((i for i, arc in enumerate(arcs) if arc[0] == 0), None)
    if root_idx is None:
        print("  无明确根节点")
        return
    
    def print_tree(node_idx, level=0):
        word = words[node_idx]
        head, rel, _ = arcs[node_idx]
        indent = "  " * level
        print(f"{indent}└── {word} (ID:{node_idx+1}, 依存:{rel}, 首依:{head})")
        
        # 递归处理子节点
        children = [i for i, arc in enumerate(arcs) if arc[0] == node_idx + 1]
        for child in sorted(children):
            print_tree(child, level + 1)
    
    print_tree(root_idx)
    print()

def save_results(results, output_file):
    """保存结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(f"句子: {res['sentence']}\n")
            f.write(f"分词: {' '.join(res['seg'])}\n")
            f.write(f"词性: {' '.join(res['pos'])}\n")
            f.write(f"依存: {res['dep']}\n\n")
    print(f"结果已保存到 {output_file}")

def analyze(input_file):
    """主函数"""
    ltp = load_ltp_model()
    if not ltp:
        return
    
    try:
        text = read_text_file(f"{input_file}")
    except FileNotFoundError as e:
        print(e)
        return
    
    sentences = segment_sentences(ltp, text)
    results = syntactic_analysis(ltp, sentences)
    save_results(results, f"syntactic_zh/ana_{input_file}")

if __name__ == "__main__":
    from ltp import __version__
    print(f"LTP版本: {__version__}")
    try:
        import graphviz
        print(f"Graphviz版本: {graphviz.__version__}")
    except ImportError:
        print("请先安装graphviz库: pip install graphviz")
        print("并安装Graphviz软件: https://graphviz.org/download/")
    
    # 检查 Graphviz 是否可用
    try:
        import graphviz
        graphviz.version()
        print("Graphviz 可执行文件找到，可以生成图像。")
    except graphviz.ExecutableNotFound:
        print("警告: 未找到 Graphviz 可执行文件，将只能生成文本结果和 dot 文件。")
        print("请安装 Graphviz 软件并将其添加到系统 PATH")
    
    analyze("syntactic_zh.txt")