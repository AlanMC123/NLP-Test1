from pyhanlp import HanLP
import re

def extract_semantic_roles(conll_sentence):
    """优化的语义角色提取函数，解决成分拼接和标签映射问题"""
    conll_str = str(conll_sentence)
    lines = [line.strip() for line in conll_str.split('\n') if line.strip() and not line.startswith('#')]
    
    # 存储词语信息：id、词、父节点id、依存关系、词性
    word_info = []
    for line in lines:
        parts = re.split(r'\s+', line)
        if len(parts) >= 8:
            word_info.append({
                'id': parts[0],
                'word': parts[1],
                'head_id': parts[6],
                'deprel': parts[7],
                'pos': parts[3]  # 词性信息
            })

    # 识别核心动词（词性为动词且有子节点）
    core_verbs = {}
    for info in word_info:
        if info['pos'].startswith('v'):  # 动词词性判断
            has_children = any(child['head_id'] == info['id'] for child in word_info)
            if has_children:
                core_verbs[info['id']] = info['word']

    semantic_roles = []
    for verb_id, verb in core_verbs.items():
        roles = {'verb': verb, 'arguments': []}
        
        for info in word_info:
            if info['head_id'] == verb_id:
                deprel = info['deprel']
                word = info['word']
                word_id = info['id']

                # 优化1：精准拼接介词短语（如"在厨房里"而非"正在里"）
                phrase_components = [word]
                # 递归查找当前词的所有子节点，构建完整短语
                def find_children(parent_id):
                    children = []
                    for child in word_info:
                        if child['head_id'] == parent_id and child['id'] != parent_id:
                            children.append(child['word'])
                            # 继续查找子节点的子节点
                            children.extend(find_children(child['id']))
                    return children
                
                # 对介词、副词等需要组合的词进行短语构建
                if info['pos'] in ['p', 'ad', 'c']:  # 介词、副词、连词
                    phrase_components.extend(find_children(word_id))
                full_phrase = ''.join(phrase_components)

                # 优化2：更精细的语义角色标签映射
                role_label = ""
                if deprel == '主谓关系':
                    role_label = 'A0 (施事：动作执行者)'
                elif deprel == '动宾关系':
                    role_label = 'A1 (受事：动作承受者)'
                elif deprel == '状中结构':
                    # 根据完整短语的起始词判断具体角色
                    if full_phrase.startswith('用'):
                        role_label = 'AM-INS (工具：使用的物品)'
                    elif full_phrase.startswith('在'):
                        role_label = 'AM-LOC (地点：动作发生处)'
                    elif full_phrase.startswith('因为'):
                        role_label = 'AM-CAU (原因：动作原因)'
                    elif full_phrase.startswith('所以'):
                        role_label = 'AM-RES (结果：动作结果)'
                    elif full_phrase.startswith('不'):
                        role_label = 'AM-NEG (否定：动作否定)'
                    elif full_phrase.startswith('怎么'):
                        role_label = 'AM-MNR (方式：动作方式)'
                    elif full_phrase.startswith('正在'):
                        role_label = 'AM-TMP (时态：进行时)'
                    else:
                        role_label = f'AM (附加成分：{full_phrase})'
                elif deprel == '右附加关系' and word in ['了', '过']:
                    role_label = 'AM-TMP (时态：完成时标记)'
                elif deprel == '标点符号':
                    role_label = 'PUNCT (标点：句子符号)'
                elif deprel == '并列关系':
                    role_label = 'COORD (并列：并列动作)'
                elif deprel == '动补结构':
                    role_label = 'AM-EXT (补充：动作补充说明)'
                else:
                    role_label = f'OTHER (其他：{deprel})'

                roles['arguments'].append((role_label, full_phrase))
        
        semantic_roles.append(roles)

    return semantic_roles

def hanlp_srl_analysis(sentence):
    """HanLP中文语义角色标注主函数"""
    print(f"句子：{sentence}")
    print("语义角色标注结果：")

    # 获取依存分析结果
    conll_result = HanLP.parseDependency(sentence)
    roles = extract_semantic_roles(conll_result)

    if not roles:
        print("  未识别到核心动词及语义角色\n")
        return

    # 输出标注结果
    for role_info in roles:
        print(f"\n核心动词：{role_info['verb']}")
        for arg_label, arg_word in role_info['arguments']:
            print(f"  {arg_label}：{arg_word}")
    print("\n" + "-"*60 + "\n")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="A restricted method in java.lang.System has been called")

    print("初始化HanLP中文语义角色标注器...\n")

    test_sentences = [
        "小明用筷子夹了一些面条。",
        "因为我不知道怎么做，所以我便停了下来。",
        "张三说他妈妈正在厨房里做一顿美味的晚餐。",
        "柴犬蹲坐了下来，四处张望，用鼻子嗅着什么。"
    ]

    for sent in test_sentences:
        hanlp_srl_analysis(sent)