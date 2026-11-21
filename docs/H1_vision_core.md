# H1 — `vision_core.py` và hàm `generate(payload)` (NV 1.3)

Người giao: Hòa. Người nhận: Vy (dùng cho `run_benchmark.py` NV 2.7 và `schema.py` NV 5.1).

## 1. Hợp đồng

```python
from src.vision_core import VisionEngine, DummyEngine, generate
res = generate(payload, engine=engine)      # -> GenerationResult
```

### Payload (đầu vào)
| Trường | Kiểu | Mặc định | Ghi chú |
|---|---|---|---|
| `prompt` | str | bắt buộc | |
| `ref_image` | str hoặc null | null | Đường dẫn ảnh mặt. **Bắt buộc nếu `use_id=True`** |
| `use_id` | bool | true | false = FLUX thuần, không PuLID |
| `id_weight` | float 0–2 | 1.0 | Trọng số PuLID |
| `seed` | int | 42 | |
| `width`, `height` | int hoặc null | null | Chia hết cho 16, trong [256, 1536]. null → lấy `standard` trong `configs/base.yaml`, nếu trống thì 720 |
| `steps` | int 1–100 | 20 | |
| `guidance` | float 0–20 | 3.5 | |
| `lora_path`, `lora_scale` | str, float | null, 1.0 | **Chỗ giữ trước** cho LoRA; engine thật chưa hỗ trợ cho tới khi có kết quả G2 (NV 1.4) |
| `output_name` | str hoặc null | null | Tên file không đuôi (chữ, số, `.`, `_`, `-`). null → tự sinh |
| `tags` | dict | {} | `identity_id`, `prompt_id`, ... được chép nguyên vào metadata |

Trường lạ (gõ sai tên) hoặc giá trị ngoài khoảng → `pydantic.ValidationError` ngay.

### Kết quả (`GenerationResult`)
`status`, `image_path`, `image` (PIL, None nếu lỗi), `error`, `metadata`, `res.ok`, `res.to_dict()`.

| status | Nghĩa |
|---|---|
| `OK` | Ảnh sinh đúng, PuLID chạy đủ số lần |
| `BLACK` | Ảnh đen/NaN (thường do fp16); ảnh vẫn được lưu để xem |
| `OOM` | Hết VRAM; không ném lỗi, vòng lặp benchmark chạy tiếp được |
| `ERROR` | Lỗi khác, hoặc PuLID chạy sai số lần (`ca_fired` ≠ `ca_expected`) |

**Quy ước:** payload sai → raise; lỗi lúc chạy → trả status, không raise.

### Metadata (mỗi ảnh, ghi thêm 1 dòng vào `<out_dir>/metadata.jsonl`)
`contract_version`, `engine`, `timestamp`, `git_hash`, `quant`, `layout`, `name`, `width`, `height`, `payload` (đầy đủ, gồm `tags`), `t_id_s`, `t_encode_s`, `t_denoise_s`, `t_total_s`, `vram_gpu0_gb`, `vram_gpu1_gb`, `ca_fired`, `ca_expected`, `status`, `error`, `image_path`.
Lỗi (OOM/ERROR) cũng được ghi dòng, thiếu các cột thời gian/VRAM. File này chính là nguồn để gom metadata cho H5.

## 2. Dùng trên máy không có GPU (Vy)
```python
from src.vision_core import DummyEngine, generate
eng = DummyEngine({"paths": {"out_dir": "outputs"}})          # hoặc DummyEngine.from_yaml("configs/base.yaml")
r = generate({"prompt": "test", "ref_image": "ref.jpg", "seed": 1,
              "tags": {"identity_id": "id01", "prompt_id": "p1"}}, engine=eng)
```
`DummyEngine` có đúng giao diện và metadata như engine thật, trả ảnh xám có ghi chữ. Đủ để viết và kiểm thử `run_benchmark.py`, `schema.py`. Chạy `python -m pytest tests -q` để xem các kiểm thử.

## 3. Dùng trên Kaggle (có GPU)
```python
import sys; sys.path.insert(0, "/kaggle/working/<tên-repo>")
from src.vision_core import VisionEngine, generate
engine = VisionEngine.from_yaml("configs/base.yaml", quant="Q4_0", layout="split")   # nạp model: chậm, làm 1 lần
res = generate({"prompt": "...", "ref_image": "/kaggle/input/.../ref_face.jpg",
                "seed": 42, "width": 720, "height": 720}, engine=engine)
print(res.status, res.image_path, res.metadata["t_denoise_s"])
```
Yêu cầu: GPU T4 x2; Secret `HF_TOKEN` (hoặc thư mục tokenizer đã lưu); đã cài thư viện như mục 1 notebook G1; `pip install pydantic`.

Khóa tùy chọn trong `configs/base.yaml` (có mặc định nếu thiếu):
```yaml
paths:
  work_dir: /kaggle/working
  pulid_repo: /kaggle/working/PuLID
  tokenizers_dir: /kaggle/working/flux_tokenizers
  out_dir: /kaggle/working/outputs
devices:
  t5_load: cuda:0        # nếu .to() T5 4-bit lỗi: đổi thành cuda:1
```

## 4. Giới hạn đã biết
- Mỗi tiến trình chỉ nên có **một engine**, vì engine gọi `os.chdir(work_dir)` và gắn hook vào transformer.
- `layout` cố định theo engine, không đổi theo từng payload (di chuyển T5 mỗi lần sẽ chậm và dễ lỗi).
- LoRA: payload đã có trường, engine thật trả `ERROR` cho tới khi G2 (NV 1.4) xong.
- **Q8_0 chỉ chạy layout `split`**: engine tự nạp T5 và CLIP thẳng lên GPU text (`cuda:1`) khi `quant="Q8_0"`, vì Q8_0 (~12,7 GB) cùng T5 không vừa một T4. Chọn `layout="1GPU"` với Q8_0 sẽ báo lỗi ngay.
- VAE vẫn ở GPU0 (NV 1.5 sẽ xử lý).
- Chưa tính độ giống mặt trong `generate()`; chấm điểm chính thức dùng `face_eval.py` của Vy (H3).

## 5. Kiểm tra nghiệm thu H1 trên Kaggle (Hòa làm trước khi giao)
1. Nạp `VisionEngine(... quant="Q4_0", layout="split")` không lỗi.
2. `generate` 3 ảnh PuLID bật ở 720×720: `status == "OK"`, `ca_fired == 400`, ảnh có mặt đúng người.
3. So số đo với bảng G1: thời gian khoảng 63 giây/ảnh, VRAM đỉnh khoảng 8,2 GB (GPU0) và 7,3 GB (GPU1). Chênh nhiều thì kiểm tra lại.
4. `generate` với `use_id=False`: `ca_fired == 0`.
5. `metadata.jsonl` có đủ 4 dòng.
