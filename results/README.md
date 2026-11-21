# Kết quả Cổng G1 (NV 1.1) và chuẩn bị Q8_0 (NV 1.2)

Tổng hợp ngày 03/11/2026. Đây là số đo thật từ notebook `notebooks/flux_g1_clean.ipynb` (Q4_0) và `flux_g1_q8.ipynb` (Q8_0).

## 1. Môi trường và cách đo
- GPU: 2 × Tesla T4 (Kaggle). torch 2.10.0+cu128. fp16.
- FLUX.1-dev GGUF + PuLID v0.9.1; T5-xxl 4-bit. 20 steps, guidance 3.5. Seed lượt đo = 42, 43, 44.
- Mỗi cấu hình: 1 lượt warmup (4 steps) rồi **3 lượt đo**; bảng báo trung bình.
- **Layout `1GPU`**: transformer, T5, CLIP, VAE đều trên `cuda:0`.
- **Layout `split`**: T5 + CLIP trên `cuda:1`; transformer, VAE, PuLID cross-attention trên `cuda:0`.
- Thời gian = mã hóa prompt + denoise + giải mã VAE (không gồm tính ID embedding, vì tính sẵn một lần).
- VRAM = `torch.cuda.max_memory_allocated` mỗi GPU (chưa gồm khoảng 0,3 đến 0,5 GB của CUDA context).
- `ID sim` = độ giống mặt với ảnh tham chiếu (insightface antelopev2). **Chỉ đo trên một danh tính**; thước đo chính thức của đồ án (AdaFace/FaceNet) do Vy làm ở NV 2.3.

## 2. Bảng G1 — Q4_0 (`results/Q4_0/`)
| Layout | Kích thước | PuLID | Thời gian/ảnh (s) | VRAM GPU0 / GPU1 (GB) | ID sim |
|---|---|---|---|---|---|
| 1GPU | 720² | tắt | 62,42 | 14,33 / 1,01 | 0,099 |
| 1GPU | 720² | bật | 63,57 | 14,33 / 1,01 | 0,794 |
| 1GPU | 1024² | tắt, bật | **OOM** | n/a | n/a |
| split | 720² | tắt | 62,49 | 8,21 / 7,29 | 0,099 |
| split | 720² | bật | 63,60 | 8,21 / 7,29 | 0,794 |
| split | 1024² | tắt | 110,14 | 9,65 / 7,29 | 0,144 |
| split | 1024² | bật | 112,40 | 9,65 / 7,29 | 0,809 |

## 3. Bảng G1 — Q8_0 (`results/Q8_0/`)
Chỉ đo layout `split`: file Q8_0 (~12,7 GB) cùng T5 không vừa một T4.

| Layout | Kích thước | PuLID | Thời gian/ảnh (s) | VRAM GPU0 / GPU1 (GB) | ID sim |
|---|---|---|---|---|---|
| split | 720² | tắt | 52,79 | 13,96 / 7,29 | 0,087 |
| split | 720² | bật | 54,11 | 13,96 / 7,29 | 0,801 |
| split | 1024² | tắt | 102,42 | 14,29 / 7,29 | 0,177 |
| split | 1024² | bật | 104,84 | 14,29 / 7,29 | 0,820 |

Cả hai bản: mọi dòng PuLID bật có `fired_ok = True` (PuLID chạy đủ số lần cross-attention) và không có ảnh đen.

## 4. So sánh Q4_0 và Q8_0 (split, PuLID bật)
| Kích thước | Model | Thời gian (s) | VRAM GPU0 (GB) | ID sim |
|---|---|---|---|---|
| 720² | Q4_0 | 63,60 | 8,21 | 0,794 |
| 720² | Q8_0 | 54,11 | 13,96 | 0,801 |
| 1024² | Q4_0 | 112,40 | 9,65 | 0,809 |
| 1024² | Q8_0 | 104,84 | 14,29 | 0,820 |

- Q8_0 **nhanh hơn** khoảng 8 đến 9 giây/ảnh nhưng dùng thêm **4,6 đến 5,8 GB** VRAM trên GPU0.
- Bật PuLID tốn thêm khoảng 1,1 đến 1,3 giây (720²) và 2,3 đến 2,4 giây (1024²), VRAM không đổi.
- Độ giống mặt gần như bằng nhau trên một danh tính; chưa kết luận được về chất lượng (việc của G3).
- Bật PuLID đưa ID sim từ khoảng 0,09 đến 0,18 lên khoảng 0,79 đến 0,82, nên PuLID đang tác động rõ.
- Ở cùng seed, layout `1GPU` và `split` cho ảnh trông giống hệt: chia GPU không đổi kết quả.

## 5. Cấu hình chuẩn đề xuất (chờ Vy xác nhận)
**split, 720×720, 20 steps, guidance 3.5, fp16.**
- Đây là cấu hình duy nhất cả Q4_0 và Q8_0 cùng chạy ổn định, nên G3 và C1 có thể so sánh ở cùng kích thước.
- Q8_0 ở 1024² dùng 14,29 GB, sát giới hạn (cột `fits_limit` = False): **không dùng 1024² cho Q8_0**.
- Hai file `g1_chosen_*.yaml` tự sinh đang lệch nhau (Q4_0 đề xuất 1024², Q8_0 đề xuất 720²) vì mỗi bên chọn riêng; chọn 720² chung để so sánh công bằng.
- Chi phí GPU tham khảo (Q4_0, 720²): khoảng 64 giây/ảnh; 120 ảnh của G3 mất khoảng 2,1 giờ. Q8_0 nhanh hơn, nhưng VRAM sát trần nên cần theo dõi khi chạy dài.

## 6. Thử nhiều ngữ cảnh (phụ lục, 720², PuLID bật, 6 prompt × 2 seed)
Không tính `watercolor`. Có 1 ảnh `full_body` (seed 42) không phát hiện được mặt, ở cả hai bản.

| Model | ID sim TB (antelopev2) | ID sim TB (buffalo_l) | Thấp nhất |
|---|---|---|---|
| Q4_0 | 0,701 | 0,644 | 0,239 (`side_3q`, seed 123) |
| Q8_0 | 0,766 | 0,704 | 0,536 (`full_body`, seed 123) |

Chênh lệch trung bình chủ yếu do một ảnh Q4_0 (`side_3q`, seed 123) bị méo mắt. Mới là một mẫu nhỏ, cần G3 kiểm tra trước khi kết luận Q8_0 tốt hơn.

## 7. Giới hạn đã biết
- Số liệu trên **một danh tính tham chiếu**, 3 lượt mỗi cấu hình.
- VRAM đo bằng `max_memory_allocated`; thực tế sát trần T4 hơn một chút (reserved: Q8_0 720² = 14,16 GB, 1024² = 14,34 GB).
- Q8_0 chưa đo layout 1GPU (không khả thi).
- ID sim ở đây dùng bộ nhận diện giống bộ của PuLID, nên chỉ dùng để kiểm tra nhanh.

## 8. Tệp trong thư mục này
```
results/
  README.md
  Q4_0/  g1_table.csv  g1_runs.csv  g1_env.json  g1_chosen_Q4_0.yaml
  Q8_0/  g1_table.csv  g1_runs.csv  g1_env.json  g1_chosen_Q8_0.yaml
```
Ảnh mẫu không commit (nặng); có thể thêm 2 đến 3 ảnh JPEG vào `results/<QUANT>/images/` nếu cần minh họa.

## 9. Tái lập
Notebook `notebooks/flux_g1_clean.ipynb` (Q4_0) và `flux_g1_q8.ipynb` (Q8_0) trên Kaggle T4 x2. Chi tiết trong `README.md` gốc của repo.
