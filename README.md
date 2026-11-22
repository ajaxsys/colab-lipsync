# 🎙️ Colab InfiniteTalk Lipsync (Wan2.1) & LongCat-Video-Avatar

基于 **Wan2.1 (14B) + InfiniteTalk** 与 **LongCat-Video-Avatar 1.5/1.6** 的高精度开源数字人工作室。针对 Google Colab（A100 / L4）深度定制，提供免安装一键运行 Notebook、**Google Drive 分段断点续跑引擎（防回收/防崩溃）**、状态持久回显与双旗舰模型横向对比方案。

---

## ✨ 核心特性

- **🛡️ Google Drive 毫秒级断点续跑引擎**：长歌曲/长视频生成期间，系统按分段（约 2.85 秒）自动将视频小切片与末尾关键帧张量原子化提交至 Google Drive。即使遭遇 Colab 偶发断联、算力抢占回收，**下次启动直接从断点毫秒级接续，绝不重算白跑**！
- **一键开箱即用**：提供标准 Jupyter Notebook，一行代码自动配置依赖、模型与穿透网络。
- **页面刷新自动回显**：前端与后端状态彻底解耦，无论何时刷新浏览器，已上传的图片、音频、参数设置以及生成的视频均自动恢复回显。
- **物理防重复锁**：点击生成后按钮自动锁定并禁用，后台任务严格单并发排队，杜绝误触多点导致 GPU 显存溢出（OOM）。
- **智能截取控制**：支持自由指定“只生成前 X 秒”进行快速预览，或一键勾选“整首生成”。
- **静音/前奏保护**：内置人声分离门限保护，针对前奏纯伴奏与静音片段，防止底噪被放大导致误张嘴。

---

## ⚔️ 双旗舰数字人模型架构与代际对比

本项目全面支持与评测两大开源数字人旗舰方案：

| 对比维度 | 方案 A：Wan2.1 + InfiniteTalk (本项目默认) | 方案 B：LongCat-Video-Avatar 1.5/1.6 (进阶对比) |
| :--- | :--- | :--- |
| **底座基座** | 阿里 **Wan2.1-I2V-14B** (2025 年 2 月) | 美团自研 **LongCat-Video (13.6B)** (2025 年末/2026) |
| **音频编码器** | **Chinese-Wav2Vec2** (Tencent Games 2021~2022) | **OpenAI Whisper-Large-v3** (2024~2025) |
| **推理步数与加速** | **4 步蒸馏** (LightX2V CFG-Distill) | **8 步蒸馏** (DMD2 原生蒸馏) |
| **视觉风格与优势** | **影视级写实真人**：皮肤肌理、眼神光真实感极强 | **高泛化与动作防畸变**：强化学习防手部变形，二次元/动物表现好 |
| **歌唱咬字表现** | 中文普通话咬字清晰，长转音平稳 | Whisper 加持下，对微弱气口、转音起伏与多语种更敏感 |
| **断点接续支持** | **✅ 已实装 Google Drive 分段断点引擎** | 依赖一次性批次生成 |
| **Notebook 入口** | [Colab_InfiniteTalk_Web.ipynb](Colab_InfiniteTalk_Web.ipynb) | [Colab_LongCat_Avatar.ipynb](Colab_LongCat_Avatar.ipynb) |

---

## 💡 硬件配置与最佳黄金“甜蜜点” (Sweet Spot)

经过大量实测与算力开销对比，我们总结出运行 InfiniteTalk 的最佳硬件配置策略：

| GPU 规格 | 显存 / 内存 | 单步推理速度 | 480P 10s 耗时 | 推荐指数 | 场景与性价比分析 |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **NVIDIA A100 (40GB Standard)** | **40GB HBM2e / 83GB RAM** | **⚡ 13.3 秒 / 步** | **~40 秒** | 🌟🌟🌟🌟🌟<br/>**(最佳黄金甜蜜点)** | **首选主力卡**：40 层模型全常驻显存，零层交换延迟；点数仅需 ~12 点/hr（比 80GB 省近一半）；卡槽充裕易申请。 |
| **NVIDIA L4** | 24GB GDDR6 / 52GB RAM | **~45 秒 / 步** | **~2-3 分钟** | 🌟🌟🌟🌟<br/>(高性价比备选) | **经济长效**：点数消耗更低，适合长时间挂机或日常调试；Layer Swapper 自动调优。 |
| **NVIDIA A100 (80GB High-RAM)** | 80GB / 85GB RAM | 13.3 秒 / 步 | ~40 秒 | 🌟🌟<br/>(不推荐争抢) | 资源严重溢出，每小时消耗高达 18~23 点，且常年排队报 `Service Unavailable`，**完全无必要申请**。 |
| **NVIDIA T4** | 16GB / 12.7GB RAM | OOM | 崩溃 | ❌ (不可用) | 显存与架构无法承载 14B 模型，请勿使用。 |

> [!TIP]
> ### 🎯 为什么 A100-40GB 是终极“黄金甜蜜点”？
> 1. **显存刚好完美吃满**：在 480P 原生高清模式下，Wan2.1 14B + InfiniteTalk + VAE 峰值显存仅占用 **26.9 GB**。40GB 显存刚好让所有 40/40 层全部驻留显卡，达到 312 TFLOPS 的满血算力，**速度和 80GB 版本完全没有任何区别**！
> 2. **巨省算力点数**：A100-40GB（Standard 档）每小时消耗约 **12.5 计算单元**，而 80GB High-RAM 档每小时高达 **18~23 计算单元**，点数直接节省 **40% ~ 50%**！
> 3. **秒级秒批，无需排队**：全网都在争抢 80GB High-RAM，导致经常遇到 `Service Unavailable`；而 40GB Standard 算力池卡槽充裕，秒申秒就绪。
> 4. **自带 83GB 巨量内存**：Colab 为 A100 Standard 默认标配了 **83GB 系统内存**，本身就已处于 High-RAM 规格，根本不需要额外勾选 `--high-mem`。

---

## 🚀 快速开始 (Google Colab)

### 选项 1：运行 Wan2.1 + InfiniteTalk (带断点续跑保护)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajaxsys/colab-lipsync/blob/main/Colab_InfiniteTalk_Web.ipynb)
直接点击上方徽章打开 [`Colab_InfiniteTalk_Web.ipynb`](Colab_InfiniteTalk_Web.ipynb)，建议在第 2 步后勾选挂载 Google Drive，自动激活防回收断点续跑。

### 选项 2：运行 LongCat-Video-Avatar 1.5/1.6 (横向效果对比)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajaxsys/colab-lipsync/blob/main/Colab_LongCat_Avatar.ipynb)
直接打开 [`Colab_LongCat_Avatar.ipynb`](Colab_LongCat_Avatar.ipynb)，内置完全相同的 30 秒《崇拜》高潮片段与写实人像素材，对比 Whisper 驱动下的对口型效果。

---

## 📂 项目文件结构

```text
├── Colab_InfiniteTalk_Web.ipynb   # Wan2.1 + InfiniteTalk 一键运行 Notebook (含断点接续)
├── Colab_LongCat_Avatar.ipynb     # LongCat-Video-Avatar 1.5 对比评测 Notebook
├── app.py                         # Gradio WebUI 前端交互服务 (含回显、锁定与断点管理)
├── extracted_runner.py            # 核心推理引擎 (Wan2.1 + InfiniteTalk + 断点持久化)
├── run_master_a100.py             # A100 全自动双版本 (9:16 + 16:9) 批处理管线
├── download_models.py             # 29GB 模型多线程下载与路径映射脚本
├── sample/                        # 测试音频、人物图像与生成 Demo 视频
├── requirements.txt               # 项目依赖清单
└── README.md                      # 项目说明文档
```

---

## 📄 开源与致谢
- [Wan2.1 (Wan-AI)](https://github.com/Wan-Video/Wan2.1)
- [InfiniteTalk (MeiGen-AI)](https://github.com/MeiGen-AI/InfiniteTalk)
- [LongCat-Video (meituan-longcat)](https://github.com/meituan-longcat/LongCat-Video)
- [LightX2V](https://github.com/Kijai/WanVideo_comfy)
