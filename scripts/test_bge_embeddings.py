"""
测试 BGE-M3 推文嵌入特征的分类效果
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report


class Classifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    features = torch.load('../Data and Process/pt_data/tweets_feature_64d.pt', map_location=device)
    labels = torch.load('../Data and Process/pt_data/label.pt', map_location=device).long()

    print(f"特征形状: {features.shape}")
    print(f"标签形状: {labels.shape}")

    # 数据集划分
    train_idx = list(range(8278))
    val_idx = list(range(8278, 9278))
    test_idx = list(range(9278, len(labels)))

    # 计算类别权重
    train_labels = labels[train_idx]
    num_class0 = (train_labels == 0).sum().item()
    num_class1 = (train_labels == 1).sum().item()
    weight = torch.tensor([1.0 / num_class0, 1.0 / num_class1], device=device)
    weight = weight / weight.sum() * 2

    print(f"\n训练集: {len(train_idx)}, 验证集: {len(val_idx)}, 测试集: {len(test_idx)}")
    print(f"类别0: {num_class0}, 类别1: {num_class1}")
    print(f"类别权重: {weight.tolist()}")

    # DataLoader
    batch_size = 64
    train_loader = DataLoader(TensorDataset(features[train_idx], labels[train_idx]),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(features[val_idx], labels[val_idx]),
                            batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(features[test_idx], labels[test_idx]),
                             batch_size=batch_size)

    # 模型
    model = Classifier(input_dim=features.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 训练
    print("\n开始训练...")
    best_val_f1 = 0
    patience, counter = 15, 0

    for epoch in range(100):
        # Train
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                val_preds.extend(model(x).argmax(1).cpu().tolist())
                val_labels.extend(y.cpu().tolist())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), '../Data and Process/pt_data/best_classifier.pt')
            counter = 0
            print(f"Epoch {epoch+1}: Val Acc={val_acc:.4f}, F1={val_f1:.4f} ✓")
        else:
            counter += 1
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Val Acc={val_acc:.4f}, F1={val_f1:.4f}")

        if counter >= patience:
            print(f"早停于 Epoch {epoch+1}")
            break

    # 测试
    print("\n" + "="*50)
    print("测试集评估")
    print("="*50)

    model.load_state_dict(torch.load('../Data and Process/pt_data/best_classifier.pt'))
    model.eval()

    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for x, y in test_loader:
            test_preds.extend(model(x).argmax(1).cpu().tolist())
            test_labels_list.extend(y.cpu().tolist())

    print(f"\nAccuracy: {accuracy_score(test_labels_list, test_preds):.4f}")
    print(f"F1 Score: {f1_score(test_labels_list, test_preds):.4f}")
    print("\n分类报告:")
    print(classification_report(test_labels_list, test_preds, target_names=['Human', 'Bot'], digits=4))


if __name__ == '__main__':
    main()

