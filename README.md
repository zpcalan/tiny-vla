# Windows 运行 run_qwen3_vl_4b.py 教程

## 前置要求

- Windows 10/11 64位
- NVIDIA GPU（显存建议 ≥ 6GB，4bit量化后约需 4-5GB）
- CUDA 驱动已安装（建议 CUDA 12.1+）

---

## 1. 安装 Miniconda

前往 https://docs.conda.io/en/latest/miniconda.html 下载 Windows 版安装包，安装时勾选 "Add to PATH"。

安装完成后打开 **Anaconda Prompt**。

---

## 2. 创建并激活 conda 环境

```bash
conda create -n tiny-vla python=3.11 -y
conda activate tiny-vla
```

---

## 3. 安装 PyTorch（CUDA 版本）

> 注意：requirements.txt 中的 torch==2.9.1 为最新版，pip 会自动拉取。若找不到，改用下面的稳定版命令。

**推荐方式（稳定版 torch + CUDA 12.1）：**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

如果你的 CUDA 是 12.4+，把 `cu121` 换成 `cu124`。

---

## 4. 安装其余依赖

```bash
pip install -r requirements.txt
```

---

## 5. 下载模型

从 HuggingFace 或 ModelScope 下载 `Qwen3-VL-4B-Instruct` 模型到本地。

**ModelScope（国内推荐）：**

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-VL-4B-Instruct', local_dir='E:/我的项目/tiny-vla/qwen3-vl-4b-instruct')"
```

**HuggingFace：**

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir E:/我的项目/tiny-vla/qwen3-vl-4b-instruct
```

---

## 6. 准备测试图片

将一张图片放到 `E:/我的项目/tiny-vla/random_shot.jpg`，或在运行时通过参数指定路径。

---

## 7. 运行脚本

```bash
python run_qwen3_vl_4b.py
```

使用自定义路径：

```bash
python run_qwen3_vl_4b.py --model-path E:/你的路径/qwen3-vl-4b-instruct --image-file-path E:/你的路径/test.jpg
```

---

## 常见问题

**Q: `bitsandbytes` 报错 `CUDA Setup failed`**
A: 确认 CUDA 驱动版本与 PyTorch 的 CUDA 版本匹配，运行 `nvidia-smi` 查看驱动支持的最高 CUDA 版本。

**Q: `OutOfMemoryError`**
A: 显存不足，脚本已启用 4bit 量化，若仍 OOM 可尝试关闭其他占用显存的程序。

**Q: `qwen_vl_utils` 找不到**
A: 执行 `pip install qwen-vl-utils` 重新安装。

**Q: `Qwen3VLForConditionalGeneration` 找不到**
A: transformers 版本过低，确认 `transformers>=4.57.3`。
