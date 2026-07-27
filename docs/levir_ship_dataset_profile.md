# LEVIR-Ship dataset profile

Tài liệu này mô tả snapshot COCO được tạo bởi
`train_all_levir_baseline.py` với seed 42. Các nhận xét kiến trúc nên dựa
trên số liệu ở đây thay vì giả định rằng mọi tàu đều có contour hoặc
high-frequency edge rõ.

## Nguồn và protocol

- Một class: `ship`.
- Ảnh nguồn: PNG `512×512` trong `LevirShipData/All Images`.
- Annotation nguồn: YOLO text trong `LevirShipData/All Annotations`.
- Annotation dùng khi train: COCO JSON trong
  `mmdetection/data/levir_ship_coco/annotations`.
- Split theo source scene được trích từ tên file bằng biểu thức
  `^(.*)_(-?\d+)_(-?\d+)$`. Các crop của cùng một scene không xuất hiện ở
  nhiều split.
- Tiny object trong các thí nghiệm PAHR được định nghĩa bằng
  `sqrt(area_original) <= 16 px`.

## Thống kê split

| Split | Scenes | Images | Boxes | Positive images | Negative images | Tiny boxes |
|---|---:|---:|---:|---:|---:|---:|
| Train | 100 | 2,728 | 1,818 | 1,262 | 1,466 (53.7%) | 449 (24.7%) |
| Validation | 7 | 584 | 763 | 366 | 218 (37.3%) | 347 (45.5%) |
| Test | 7 | 584 | 638 | 345 | 239 (40.9%) | 105 (16.5%) |

Validation có tỷ lệ tiny box cao hơn đáng kể train và test. Vì vậy nên chọn
model bằng validation AP75 nhưng phải báo riêng test AP75 và tiny recall;
không suy diễn rằng mức cải thiện validation sẽ giữ nguyên trên test.

Số box chạm ít nhất một biên ảnh:

| Train | Validation | Test |
|---:|---:|---:|
| 104 | 28 | 31 |

## Phân bố kích thước

Các hàng dưới là percentile `[min, p10, p25, p50, p75, p90, max]`, tính bằng
pixel trên ảnh gốc `512×512`.

| Split | Width | Height | sqrt(area) |
|---|---|---|---|
| Train | 4.37, 12.23, 15.73, 20.53, 26.00, 31.00, 51.00 | 5.84, 13.00, 15.92, 19.00, 23.74, 28.00, 41.00 | 5.05, 13.23, 16.12, 19.97, 23.65, 27.89, 42.43 |
| Validation | 6.55, 11.00, 13.00, 16.00, 21.00, 27.80, 54.00 | 7.43, 12.00, 14.00, 17.00, 21.00, 27.00, 63.00 | 7.43, 11.96, 13.82, 16.73, 20.47, 25.97, 50.20 |
| Test | 6.12, 13.85, 18.00, 21.00, 25.75, 31.00, 49.00 | 6.37, 14.00, 16.98, 21.00, 25.00, 31.00, 45.00 | 7.08, 14.64, 17.44, 21.09, 24.92, 28.37, 43.15 |

Resize 768 làm các kích thước tuyến tính tăng `1.5×`. Đây là thay đổi quan
trọng với AP75: một box `6×6` bị dịch ngang 1 pixel chỉ còn IoU khoảng
`0.714`, trong khi box tương ứng `9×9` sau resize có IoU khoảng `0.800`.

## Appearance và local contrast

Local contrast được đo trên ảnh grayscale 8-bit:

1. Lấy mean intensity bên trong bbox.
2. Lấy mean intensity của vòng nền rộng 4 pixel quanh bbox, loại phần bbox.
3. Báo `abs(object_mean - background_mean)`.

Median absolute contrast:

| Train | Validation | Test |
|---:|---:|---:|
| 2.14 / 255 | 1.74 / 255 | 1.24 / 255 |

Percentile contrast `[min, p10, p25, p50, p75, p90, max]`:

- Train: `0.00, 0.44, 1.01, 2.14, 4.02, 7.41, 35.13`.
- Validation: `0.01, 0.54, 1.04, 1.74, 2.93, 4.75, 29.43`.
- Test: `0.01, 0.25, 0.53, 1.24, 3.01, 5.36, 19.54`.

Đây là measured fact về contrast trung bình, không phải phép đo edge hoặc
shape quality. Quan sát định tính contact sheet của 16 GT nhỏ nhất cho thấy
nhiều mục tiêu là blob sáng/mờ trên texture biển, nhưng mẫu này không đại
diện cho mọi tàu trong dataset. Không nên kết luận toàn dataset “không có
edge” nếu chưa đo gradient/contour trên toàn bộ phân bố.

## Hàm ý cho thí nghiệm

- Báo cả mAP, AP50, AP75 và tiny recall; AP75 đặc biệt nhạy với sai số tâm và
  kích thước 1–2 pixel.
- So sánh kiến trúc phải giữ cùng image size, split, seed và LR schedule.
- Validation thiên về tiny hơn test; chỉ test winner cuối cùng.
- Negative images chiếm tỷ lệ lớn, nên auxiliary heatmap loss phải xử lý
  empty-GT batch và imbalance hữu hạn.
- Border boxes cần được giữ trong target tests và diagnostics.
- Image-space/local-contrast hoặc subpixel measurement là giả thuyết phù hợp
  để thử; edge enhancement chỉ nên dùng khi có số đo chứng minh.

## Tái tạo số liệu

Đọc lần lượt:

```text
mmdetection/data/levir_ship_coco/annotations/train.json
mmdetection/data/levir_ship_coco/annotations/val.json
mmdetection/data/levir_ship_coco/annotations/test.json
```

Đếm scene bằng cùng `SCENE_RE` trong `train_all_levir_baseline.py`. Kích
thước box lấy từ COCO `bbox=[x,y,width,height]`; tiny dùng `sqrt(width*height)
<= 16`. Contrast dùng OpenCV grayscale và vòng nền 4 pixel như protocol trên.
Mọi lần regenerate split hoặc thay seed phải cập nhật lại tài liệu này.
