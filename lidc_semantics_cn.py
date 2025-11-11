# lidc_semantics.py

# 癌样等级语义标签
malignancy_semantics = {
    1: "完全不可能：结节几乎肯定是良性的",
    2: "不太可能：可能是良性，但不能完全排除恶性",
    3: "不确定：影像特征模糊，无法判断良恶性",
    4: "有可能恶性：具有一些恶性特征，但尚不明确",
    5: "极可能恶性：高度怀疑为恶性肿瘤"
}

def explain_malignancy(score: float) -> str:
    """根据加权平均分值返回癌样等级语义标签"""
    if score < 1.5:
        return malignancy_semantics[1]
    elif score < 2.5:
        return malignancy_semantics[2]
    elif score < 3.5:
        return malignancy_semantics[3]
    elif score < 4.5:
        return malignancy_semantics[4]
    else:
        return malignancy_semantics[5]

# 属性中英文对照表
attribute_name_cn = {
    "Subtlety": "结节明显程度",
    "Internal Structure": "内部结构",
    "Calcification": "钙化程度",
    "Sphericity": "球形程度",
    "Margin": "边缘清晰度",
    "Lobulation": "分叶程度",
    "Spiculation": "毛刺程度",
    "Texture": "纹理复杂度"
}

def get_attribute_cn(name: str) -> str:
    """根据英文属性名返回中文名称"""
    return attribute_name_cn.get(name, name)

# 每个属性的语义标签字典
attribute_semantics = {
    "Subtlety": {
        1: "极不明显", 2: "不明显", 3: "中等明显", 4: "明显", 5: "非常明显"
    },
    "Internal Structure": {
        1: "空泡结构", 2: "混合结构", 3: "实性结构", 4: "均匀实性", 5: "高密度实性"
    },
    "Calcification": {
        1: "高度钙化", 2: "中度钙化", 3: "轻度钙化", 4: "微钙化", 5: "极微钙化", 6: "无钙化"
    },
    "Sphericity": {
        1: "极不球形", 2: "不球形", 3: "中等球形", 4: "接近球形", 5: "非常球形"
    },
    "Margin": {
        1: "极不清晰", 2: "不清晰", 3: "中等清晰", 4: "清晰", 5: "非常清晰"
    },
    "Lobulation": {
        1: "无分叶", 2: "轻度分叶", 3: "中度分叶", 4: "明显分叶", 5: "强烈分叶"
    },
    "Spiculation": {
        1: "无毛刺", 2: "轻度毛刺", 3: "中度毛刺", 4: "明显毛刺", 5: "强烈毛刺"
    },
    "Texture": {
        1: "均匀", 2: "轻度不均匀", 3: "中度不均匀", 4: "明显不均匀", 5: "极度不均匀"
    }
}

def explain_attribute(attr_name: str, prob: float) -> tuple:
    """根据属性名和概率返回分值与语义标签"""
    max_score = max(attribute_semantics[attr_name].keys())
    score = round(prob * (max_score - 1) + 1)
    score = max(1, min(score, max_score))  # 限制在合法范围
    label = attribute_semantics[attr_name][score]
    return score, label
