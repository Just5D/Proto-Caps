# Proto-Caps

## Implementation code of paper "Interpretable Medical Image Classification using Prototype Learning and Privileged Information", MICCAI 2023

### Quick start:
Set up [pylidc](https://pylidc.github.io/install.html), install requirements and run main.py with argument **--train=True**
After training is finished, run main.py with arguments **--test=True --model_path=<model_path> --epoch=<epoch_number_of_model>**

Find more parser arguments in main.py to adapt model architecture and algorithm.

If you have any questions, please contact luisa.gallee@uni-ulm.de.

## 🔍 模块增强说明

本 Fork 在原始 Proto-Caps 基础上新增了：

- ✅ `lidc_semantics.py`：封装 LIDC-IDRI 癌样等级与视觉属性的语义解释
- ✅ `main.py`：集成语义解释模块，输出更具医学可读性的推理结果

### 📦 使用方式

```python
from lidc_semantics import explain_malignancy, explain_attribute


python main.py --infer_patch d:\OuDev2025\ExtractedNodules\nodule_4\patch.nii.gz --model_path="2025-08-07_10-33-13_0.9810972798524665_220.pth"

🔍 视觉属性预测结果：

1. 结节明显程度（Subtlety）：明显（分值 = 4 / 概率 = 0.6381）
2. 内部结构（Internal Structure）：实性结构（分值 = 3 / 概率 = 0.5010）
3. 钙化程度（Calcification）：极微钙化（分值 = 5 / 概率 = 0.7276）
4. 球形程度（Sphericity）：中等球形（分值 = 3 / 概率 = 0.6234）
5. 边缘清晰度（Margin）：中等清晰（分值 = 3 / 概率 = 0.5739）
6. 分叶程度（Lobulation）：中度分叶（分值 = 3 / 概率 = 0.5507）
7. 毛刺程度（Spiculation）：中度毛刺（分值 = 3 / 概率 = 0.5267）
8. 纹理复杂度（Texture）：明显不均匀（分值 = 4 / 概率 = 0.7143）

🧪 癌样等级预测分数（加权平均）: 3.0009
→ 语义判断：不确定：影像特征模糊，无法判断良恶性

癌样等级分布（LIDC 原始语义）:
  等级 1: 0.149076 → 完全不可能：结节几乎肯定是良性的
  等级 2: 0.149076 → 不太可能：可能是良性，但不能完全排除恶性
  等级 3: 0.402795 → 不确定：影像特征模糊，无法判断良恶性
  等级 4: 0.149977 → 有可能恶性：具有一些恶性特征，但尚不明确
  等级 5: 0.149076 → 极可能恶性：高度怀疑为恶性肿瘤