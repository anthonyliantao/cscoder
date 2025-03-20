# cscoder

## 介绍

`cscoder` 是一个用于将非结构化职业名称匹配到中国标准职业分类（CSCO）代码的 Python 库。
它使用 SentenceTransformer 进行语义匹配，并支持多层级职业代码转换。

## 安装

pip 安装：

```
pip install cscoder==0.2.4
```

## 使用示例

```python
from cscoder import CSCOder
 
# 初始化 CSCOder
coder = CSCOder()
 
# 匹配单个职业名称
result = coder.find_best_match("软件工程师")
print(result)
 
# 批量匹配
job_list = ["数据分析师", "产品经理", "注册会计师"]
results = coder.find_best_matches(job_list)
print(results)
```

## 贡献

欢迎贡献代码！如果你有任何改进建议，请提交 Issue 或 Pull Request：

1. Fork 本仓库
2. 创建新分支 (`git checkout -b feature-branch`)
3. 提交修改 (`git commit -m "添加新功能"`)
4. 推送到你的 Fork (`git push origin feature-branch`)
5. 创建 Pull Request

## 许可证

MIT License
