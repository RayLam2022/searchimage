# CLIP图像搜索工具

这是一个基于CLIP模型和Faiss向量数据库的图像搜索工具，允许用户通过文本描述或参考图片快速搜索本地图像库中的相似图片。

## 功能特点

- **文本搜索**：使用自然语言描述查找图片（如"gold"）
- **以图搜图**：上传参考图片查找视觉相似的图片
- **高效的向量索引**：使用Faiss构建本地图片索引，支持快速搜索
- **直观的用户界面**：使用Tkinter实现的桌面应用

## 安装指南

本项目使用[uv](https://github.com/astral-sh/uv)作为包管理工具，它提供比pip更快的依赖安装。

- [UV](https://github.com/astral-sh/uv) 包管理工具
- Python 3.12+
- CUDA 12.4


### 使用方法

```bash
git clone https://github.com/yourusername/searchimage.git
cd searchimage
uv run main.py
```

### 注意事项
因为用的是英文模型,文本搜索时要用英文，又或者自行改代码里的模型为支持中文的模型