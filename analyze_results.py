import json
import os

results = {}
base_dir = "test/ouputs_pswf_paper_latest"

for exp_name in os.listdir(base_dir):
    metrics_file = os.path.join(base_dir, exp_name, "metrics.jsonl")
    if not os.path.exists(metrics_file):
        continue
    
    best_acc = 0
    best_epoch = 0
    with open(metrics_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            acc = data.get('val_acc', 0)
            if acc > best_acc:
                best_acc = acc
                best_epoch = data.get('epoch', 0)
    
    results[exp_name] = {'best_acc': best_acc, 'best_epoch': best_epoch}

# 按准确率排序
sorted_results = sorted(results.items(), key=lambda x: x[1]['best_acc'], reverse=True)

print("=" * 80)
print("实验结果汇总（按最佳验证准确率排序）")
print("=" * 80)
print()

# VIL 结果
print("【VIL (Vision LSTM) 结果】")
print("-" * 80)
vil_results = [(k, v) for k, v in sorted_results if 'vil' in k.lower()]
for name, data in vil_results:
    print(f"{name:50s} Epoch {data['best_epoch']:3d}: {data['best_acc']*100:.2f}%")
print()

# ViT 结果
print("【ViT (Vision Transformer) 结果】")
print("-" * 80)
vit_results = [(k, v) for k, v in sorted_results if 'vit' in k.lower()]
for name, data in vit_results:
    print(f"{name:50s} Epoch {data['best_epoch']:3d}: {data['best_acc']*100:.2f}%")
print()

# 对比分析
print("=" * 80)
print("【关键对比】")
print("=" * 80)

# VIL 对比
vil_baseline = next((v for k, v in vil_results if 'poolonly' in k.lower()), None)
vil_w3_add = next((v for k, v in vil_results if 'W3_add' in k and 'improved' not in k), None)
vil_w3_improved = next((v for k, v in vil_results if 'W3_improved' in k and 'warmup' not in k), None)
vil_w3_improved_warmup = next((v for k, v in vil_results if 'W3_improved_warmup' in k), None)

if vil_baseline and vil_w3_add:
    diff = (vil_w3_add['best_acc'] - vil_baseline['best_acc']) * 100
    print(f"VIL W3_add vs POOLONLY: {diff:+.2f}%")
if vil_baseline and vil_w3_improved:
    diff = (vil_w3_improved['best_acc'] - vil_baseline['best_acc']) * 100
    print(f"VIL W3_improved vs POOLONLY: {diff:+.2f}%")
if vil_baseline and vil_w3_improved_warmup:
    diff = (vil_w3_improved_warmup['best_acc'] - vil_baseline['best_acc']) * 100
    print(f"VIL W3_improved_warmup vs POOLONLY: {diff:+.2f}%")
