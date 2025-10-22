import stanza
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def load_local_stanza_model(model_dir, lang='en'):
    """加载本地Stanza模型"""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    
    stanza_resource_dir = os.path.abspath(model_dir)
    os.environ['STANZA_RESOURCES_DIR'] = stanza_resource_dir
    
    nlp = stanza.Pipeline(
        lang=lang,
        processors='tokenize,pos,lemma,depparse',
        download_method=None,
        dir=stanza_resource_dir
    )
    
    return nlp

def analyze_sentence(nlp, sentence):
    """分析句子并返回依存关系数据"""
    doc = nlp(sentence)
    dependencies = []
    for sent in doc.sentences:
        for word in sent.words:
            dependencies.append({
                "id": word.id,
                "text": word.text,
                "pos": word.pos,
                "deprel": word.deprel,
                "head_id": word.head
            })
    return dependencies

def get_tree_structure(dependencies):
    """构建树形结构，确定每个节点的子节点"""
    tree = {}
    root = None
    
    # 初始化每个节点的子节点列表
    for item in dependencies:
        tree[item["id"]] = []
    
    # 构建树结构并找到根节点
    for item in dependencies:
        if item["head_id"] == 0:
            root = item["id"]
        else:
            tree[item["head_id"]].append((item["id"], item["deprel"]))
    
    return tree, root

def assign_tree_positions(tree, root, level_spacing=2.0, node_spacing=1.5):
    """为树节点分配位置，使用改进的树形布局算法防止重叠"""
    positions = {}
    
    # 计算每个节点的子树大小和深度
    subtree_sizes = {}
    depths = {}
    
    def calculate_subtree_size(node):
        if node not in subtree_sizes:
            if not tree[node]:  # 叶子节点
                subtree_sizes[node] = 1
                depths[node] = 0
            else:
                # 计算所有子节点的子树大小和深度
                child_sizes = []
                child_depths = []
                for child, _ in tree[node]:
                    calculate_subtree_size(child)
                    child_sizes.append(subtree_sizes[child])
                    child_depths.append(depths[child])
                
                subtree_sizes[node] = sum(child_sizes)
                depths[node] = max(child_depths) + 1 if child_depths else 0
        
        return subtree_sizes[node]
    
    # 计算根节点的子树大小
    calculate_subtree_size(root)
    
    # 分配位置
    def assign_positions(node, x_offset, y_level):
        # 计算当前节点的宽度（基于子树大小）
        width = subtree_sizes[node] * node_spacing
        
        if not tree[node]:  # 叶子节点
            positions[node] = (x_offset + width / 2, -y_level * level_spacing)
            return x_offset + width
        else:
            # 先处理所有子节点
            current_x = x_offset
            child_positions = []
            
            for child, _ in tree[node]:
                child_width = subtree_sizes[child] * node_spacing
                current_x = assign_positions(child, current_x, y_level + 1)
                child_positions.append(positions[child])
            
            # 当前节点的x位置是子节点位置的中心
            if child_positions:
                avg_x = sum(pos[0] for pos in child_positions) / len(child_positions)
            else:
                avg_x = x_offset + width / 2
                
            positions[node] = (avg_x, -y_level * level_spacing)
            return x_offset + width
    
    # 从根节点开始分配位置
    assign_positions(root, 0, 0)
    
    return positions

def visualize_dependencies(dependencies, sentence=None):
    """使用树形布局可视化依存关系"""
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点（词语）
    for item in dependencies:
        G.add_node(item["id"], label=f"{item['text']}\n({item['pos']})")
    
    # 添加边（依存关系）
    edges = []
    for item in dependencies:
        if item["head_id"] != 0:  # 0表示根节点
            G.add_edge(
                item["head_id"], 
                item["id"], 
                label=item["deprel"]
            )
            edges.append((item["head_id"], item["id"], item["deprel"]))
    
    # 获取树形结构
    tree, root = get_tree_structure(dependencies)
    if not root:
        print("无法确定根节点，使用默认布局")
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    else:
        # 使用改进的树形布局
        pos = assign_tree_positions(tree, root, level_spacing=2.5, node_spacing=2.0)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', 
                          edgecolors='black', linewidths=1.5)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, 
                          width=1.5, alpha=0.8, edge_color='gray')
    
    # 绘制节点标签
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10,
                           font_family='sans-serif', font_weight='bold')
    
    # 绘制边标签
    edge_labels = {(u, v): label for u, v, label in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8,
                                label_pos=0.3, font_family='sans-serif',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # 添加标题
    if sentence:
        plt.title(f"{sentence}", fontsize=14, pad=20)
    else:
        plt.title("语义依存关系树", fontsize=14, pad=20)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 替换为你的模型路径
    LOCAL_MODEL_DIR = "F:\\LanguageProcessing\\stanza_resources"
    
    try:
        nlp = load_local_stanza_model(LOCAL_MODEL_DIR, lang='en')
        test_sentences = [
            "Alice eats an apple with a fork.",
            "The quick brown fox jumps over the lazy dog.",
            "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence."
        ]
        
        for i, sentence in enumerate(test_sentences, 1):
            print(f"\n===== 分析第 {i} 个句子 =====")
            deps = analyze_sentence(nlp, sentence)
            visualize_dependencies(deps, sentence)
            
    except Exception as e:
        print(f"发生错误: {str(e)}")