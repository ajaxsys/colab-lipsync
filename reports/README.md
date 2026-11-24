# 📊 数字人流水线演进与评测技术报告库 (Reports Archive)

本项目的所有代际版本测试报告、算力基准测试、显存调优记录以及关键截图均已归档于本目录并纳入 Git 版本控制。

---

## 📑 报告目录索引

| 报告文档 | 对应阶段 / 算法架构 | 硬件规格 | 核心特性与结论 |
| :--- | :--- | :--- | :--- |
| **[v1.0_baseline_report.md](v1.0_baseline_report.md)** | **Wan2.1 (14B) + InfiniteTalk (Chinese-Wav2Vec2) + HTDemucs** | NVIDIA A100-40GB | **基线版本**：实测 4 步 Flow-Matching，13.3s/步，单段 59s。验证了 14B 底座与 63s 长视频稳定性，暴露了伴奏唇动误触与 25fps 顿挫短板。 |
| **[v1.1_upgrade_report.md](v1.1_upgrade_report.md)** | **Wan2.1 (14B) + InfiniteTalk + BS-RoFormer + 60fps RIFE** | NVIDIA A100-40GB | **重大升级成片**：引入 2025 SOTA BS-RoFormer (SDR 12.97 dB)，彻底杜绝伴奏底噪误触发；补齐 60fps 运动补偿与片尾声画渐隐，整场仅耗 3.79 计算点。 |
| **[v1.6_longcat_benchmark.md](v1.6_longcat_benchmark.md)** | **LongCat-Video-Avatar 1.6 (美团 13.6B BF16 + Whisper-v3 + 8步蒸馏 LoRA)** | **NVIDIA A100-80GB (全精度)** / L4-24GB (对比) | **巅峰对比版**：美团 13.6B 原生 BF16 全精度底座 + Whisper-Large-v3。彻底根除 L4 上的 Confetti 彩噪与纯黑，全片 17 分钟产出 30.48 秒成片。 |
| **[v2.1_longcat_upgrade_report.md](v2.1_longcat_upgrade_report.md)** | **LongCat-Video-Avatar v2.1 (双轨静音门控 + 唇动张力收敛 + 双 LoRA 融合)** | **NVIDIA A100-40GB (BF16 全精度)** | **🔥 旗舰交付成片**：前奏能量静音门控（第0秒彻底闭唇）、`guidance=0.7` + `mix_alpha=0.65`（解决过度撕口，达到抒情歌唱美感标杆）、融合算子化 `F.rms_norm`，全片 30.48s 完美收官。 |

---

## 🖼️ 采样快照资源库 (`reports/assets/`)

- `frame_5s.jpg`, `frame_25s.jpg`：v1.0 阶段 480P/720P 渲染截图
- `snap_5s.jpg`, `snap_20s.jpg`, `snap_35s.jpg`：v1.0 43s 竖屏成片关键帧
- `snap_v2_5s.jpg`, `snap_v2_15s.jpg`, `snap_v2_25s.jpg`：v1.1 增强版咬字与光影细节
- `snap_v2_fade_28_8s.jpg`：v1.1 片尾声画双平滑淡出黑屏截图
- `crop_portrait_preview.jpg`：人物主体 9:16 构图裁切预处理基准
- `longcat_final_1s.png`, `longcat_final_5s.png`, `longcat_final_15s.png`, `longcat_final_25s.png`, `longcat_final_30s.png`：v1.6 原生 BF16 采样快照
- `v21_crop_0s.png`, `v21_crop_3_2s.png`, `v21_crop_10s.png`, `v21_crop_20s.png`, `v21_crop_30s.png`：v2.1 唇动调优特写（首帧自然闭唇、3.2s“你”抒情微张）
