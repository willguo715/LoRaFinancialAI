from modelscope import MsDataset
import json

# 加载数据（只取500条作为小型数据集）
dataset = MsDataset.load('Duxiaoman-DI/Finance_QA', split='train')

# 转换为Qwen格式
qwen_data = []
for i, item in enumerate(dataset):
    if i >= 500:  # 只取500条，保持小型
        break
    qwen_data.append({
        "messages": [
            {"role": "user", "content": item['question']},
            {"role": "assistant", "content": item['answer']}
        ]
    })

# 保存
with open('finance_qa_mini.json', 'w', encoding='utf-8') as f:
    json.dump(qwen_data, f, ensure_ascii=False, indent=2)

print(f"✅ 已成功保存 {len(qwen_data)} 条数据到 finance_qa_mini.json")