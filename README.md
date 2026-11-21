# 🎙️ Colab InfiniteTalk Lipsync (Wan2.1)

基于 **Wan2.1 (14B) + InfiniteTalk + LightX2V 蒸馏加速** 的视频对口型工作室 WebUI。针对 Google Colab（A100 / L4）深度定制，提供免安装一键运行 Notebook、防刷新状态回显与 Google Drive 模型极速备份方案。

---

## ✨ 核心特性

- **一键开箱即用**：提供标准 Jupyter Notebook，一行代码自动配置依赖、模型与穿透网络。
- **页面刷新自动回显**：前端与后端状态彻底解耦，无论何时刷新浏览器，已上传的图片、音频、参数设置以及生成的视频均自动恢复回显。
- **物理防重复锁**：点击生成后按钮自动锁定并禁用，后台任务严格单并发排队，杜绝误触多点导致 GPU 显存溢出（OOM）。
- **智能截取控制**：支持自由指定“只生成前 X 秒”进行快速预览，或一键勾选“整首生成”。
- **Google Drive 秒级直载**：支持将 29GB 模型一键持久化备份至个人 Google Drive，下次启动时无需重复漫长下载。
- **静音/前奏保护**：内置人声分离门限保护，针对前奏纯伴奏与静音片段，防止底噪被放大导致误张嘴。

---

## 🖥️ 硬件配置建议

| GPU 规格 | 显存 | 480P 10s 生成耗时 | 推荐使用场景 |
| :--- | :--- | :--- | :--- |
| **NVIDIA A100 (SXM4)** | 40GB / 80GB | **~2-3 分钟** | **首选推荐**：40 层模型全驻留显存，速度极快 |
| **NVIDIA L4** | 24GB | **~15-18 分钟** | 经济实惠：28 层驻留显存 + 12 层动态交换，内存防爆设计 |

---

## 🚀 快速开始 (Google Colab)

直接打开项目内的 [`Colab_InfiniteTalk_Web.ipynb`](Colab_InfiniteTalk_Web.ipynb) 并在 Colab 中运行：

### 1. 检查显卡
确保 Colab 运行模式选择为 **GPU** (推荐 A100 或 L4)：
```bash
!nvidia-smi
```

### 2. 克隆项目与安装依赖
```bash
!git clone https://github.com/ajaxsys/colab-lipsync.git /content/colab-lipsync
%cd /content/colab-lipsync
!pip install -q -r requirements.txt
!git clone https://github.com/MeiGen-AI/InfiniteTalk.git /content/InfiniteTalk || true
```

### 3. 准备模型权重
* **如果 Google Drive 已有备份**：直接挂载秒级载入；
* **首次运行**：运行下载脚本自动拉取所有必需权重（Wan2.1 fp8, InfiniteTalk, LoRA, wav2vec, Demucs）：
```bash
python download_models.py
```

### 4. 启动 WebUI 与穿透隧道
```bash
python app.py
```

---

## 📂 项目文件结构

```text
├── Colab_InfiniteTalk_Web.ipynb   # Colab 一键运行 Notebook
├── app.py                         # Gradio WebUI 前端交互服务 (含回显与锁定机制)
├── extracted_runner.py            # 核心推理引擎 (Wan2.1 + InfiniteTalk + 内存动态交换)
├── download_models.py             # 29GB 模型多线程下载与路径映射脚本
├── requirements.txt               # 项目依赖清单
└── README.md                      # 项目说明文档
```

---

## 📄 开源与致谢
- [Wan2.1 (Wan-AI)](https://github.com/Wan-Video/Wan2.1)
- [InfiniteTalk (MeiGen-AI)](https://github.com/MeiGen-AI/InfiniteTalk)
- [LightX2V](https://github.com/Kijai/WanVideo_comfy)
