# 🧠 Transformer From Scratch (Attention Is All You Need)
### 📐 End-to-End Implementation of the Standard Encoder-Decoder Transformer in Pure PyTorch

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-Core-EE4C2C.svg?logo=pytorch&logoColor=white)](#)
[![NLP](https://img.shields.io/badge/NLP-Seq2Seq%20Transformer-orange.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🖼️ Demo & Minh Họa Kiến Trúc
<div align="center">
  <img src="https://raw.githubusercontent.com/baolam/baolam/main/assets/transformer_arch.png" onerror="this.src='https://raw.githubusercontent.com/tensorflow/tensor2tensor/master/docs/transformer.png'" alt="Transformer Architecture Diagram" width="85%"/>
  <p><i>Kiến trúc kinh điển Attention Is All You Need (Vaswani et al.) được tái hiện từ con số 0 trong PyTorch</i></p>
</div>

---

## 🏛️ Kiến Trúc Hệ Thống (3 Dòng)
- **1. Tầng Nhúng & Mã Hóa Vị Trí:** Xây dựng từ điển token, chuyển câu thành chuỗi ID và áp dụng Sinusoidal Positional Encoding kết hợp chia tỉ lệ $\sqrt{d_{model}}$.
- **2. Tầng Xử Lý & Mô Hình Attention Lõi:** Tái hiện trọn vẹn Scaled Dot-Product Attention, Multi-Head Attention song song hóa và cơ chế Feed-Forward với kết nối dư (Residual Connection) & Layer Normalization.
- **3. Tầng Chấp Hành & Bộ Giải Mã Seq2Seq:** Vòng lặp huấn luyện tối ưu với Noam Scheduler, Label Smoothing Loss và bộ sinh từ ngữ giải mã bằng Beam Search.

---

## 🚀 Cách Chạy (Quickstart)
```bash
# 1. Clone repository
git clone https://github.com/baolam/attention-is-all-you-need-build-the-transformer-from-scratch.git
cd attention-is-all-you-need-build-the-transformer-from-scratch

# 2. Cài đặt các gói phụ thuộc
pip install torch numpy

# 3. Chạy toàn bộ quy trình kiểm thử và huấn luyện từng bước
python scaffold.py
```

---

## 📊 Số Liệu & Benchmark (Key Metrics)
- **Mốc cài đặt hoàn thành:** **100% (9/9 milestones)** từ Tokenization, Scaled Embeddings, Positional Encodings tới Beam Search Decoding.
- **Độ phức tạp tính toán:** Khảo sát và tối ưu hóa ma trận Attention $O(n^2 \cdot d)$ thông qua phép nhân tensor vector hóa (Vectorized Batched MatMul).
- **Độ chính xác giải mã:** Kiểm chứng độ hội tụ và thuật toán Beam Search cho kết quả dịch máy chính xác trên tập ngữ liệu mẫu.
