# Báo cáo các hướng cải thiện phát hiện tàu nhỏ trên LEVIR-Ship

## 1. Phạm vi và baseline chung

Báo cáo này tổng hợp code ở các branch `dbss`, `guided_alignment`, `haar`,
`hard_transport`, `rcfn_ltmr` và `open_close`. Các hướng đều cố gắng cải thiện
đặc trưng mức thấp của detector cho tàu nhỏ, chủ yếu tại P3 của FPN. Baseline
được dùng nhiều nhất là FCOS với ResNet-50-Caffe + FPN, một lớp `ship`, ảnh
được resize về 512×512 hoặc 768×768 tùy experiment; resolution được ghi riêng
cho từng run.

LEVIR-Ship được chuyển sang COCO và chia theo **scene** để tránh các crop của
cùng một ảnh nguồn rơi vào nhiều split. Metric trong các bảng dưới đây là kết
quả **test** COCO:

- **mAP**: AP trung bình trên IoU 0.50:0.95.
- **AP50**, **AP75**: AP tại IoU 0.50 và 0.75.
- `Δ` là chênh lệch tuyệt đối so với baseline trong chính nhóm thí nghiệm đó.

Không nên so trực tiếp các bảng dùng độ phân giải, detector, lịch train hoặc
checkpoint-selection khác nhau. Các số liệu được đọc từ JSON test công khai
trên Hugging Face, không lấy từ tên checkpoint hay metric validation.

### 1.1. Quy ước mức độ xác nhận

- **Đã test và xác minh được:** có `patched_config.py` cùng test JSON công
  khai, nên xác định được detector, resolution, epoch, seed và metric test.
- **Có config, chưa confirm kết quả:** code/config mô tả cách chạy nhưng không
  tìm thấy test artifact công khai khớp. Tên work directory hay checkpoint
  không được xem là bằng chứng run đã hoàn tất.
- **Chưa kiểm chứng đầy đủ:** có một test run nhưng chỉ một seed, một
  resolution, hoặc thiếu baseline cùng protocol. Metric là thật nhưng chưa đủ
  để kết luận method tổng quát.

### 1.2. Ma trận những gì đã test và chưa confirm

| Method / variant | Detector | Resolution | Schedule / seed | Trạng thái | Chưa confirm |
|---|---|---:|---|---|---|
| DBSS baseline, ridge, ridge+Haar, softmax | FCOS R50-FPN | 768 | 20 epoch, seed 42 | Có test JSON | 512; seed khác; detector khác |
| DBSS ridge γ=0.3/0.6/1.0 | FCOS R50-FPN | 768 | 20 epoch, seed 42 | Có test JSON | γ tối ưu ở resolution/seed khác |
| DBSS falsification controls | FCOS R50-FPN | 768 | 12 epoch, seed 42 | Có test JSON, protocol riêng | Schedule 20 epoch; nhiều seed |
| DGFE + API | Intended FCOS/LEVIR | 512 | Config 12 epoch, seed 42 | Có code/config, **không có LEVIR test JSON** | Mọi run LEVIR, 768 và multi-seed |
| PAHR cơ bản (`haar`) | FCOS R50-FPN | **512** | 40 epoch, seed 42 | Có test JSON | FCOS-512 baseline cùng protocol |
| PAHR shift | FCOS R50-FPN | 768 | 40 epoch, seed 42 | Có test JSON | 512; nhiều seed |
| Haar C2 wavelet fusion | FCOS R50-FPN | 768 | 40 epoch, seed 42 | Có test JSON | C2 ablation; nhiều seed |
| PAHR-P2 và P2 baseline | Faster R-CNN R50-FPN | 768 | 20 epoch, seed 42 | **Controlled pair** có test JSON | 512; nhiều seed |
| HIT probe/warmup/detached/joint | FCOS R50-FPN | Intended 512 | Config chính 12 epoch, seed 42; joint có seed 43 | Có config, **không có test artifact** | Chưa confirm stage nào chạy xong |
| RCFN-R2, LTMR-L1 | FCOS R50-FPN | 512 | 30 epoch, seed 42 | Có test JSON | Baseline cùng schedule; 768; nhiều seed |
| PG-RCFN: R2/Aux/H/CH/low-weight/floor | FCOS R50-FPN | 512 | 30 epoch, seed 42 | Controlled ablation cùng repo | 768; nhiều seed; detector khác |
| Morphology positive/negative/both/Conv3×3 | FCOS R50-FPN | 768 | 30 epoch, seed 42 | Có test output live, **không có artifact HF** | Bias/gamma/control confounded; nhiều seed |
| Positive top-hat vs raw-P3 matched control | FCOS R50-FPN | 768 | 30 epoch, seed 42 | **Controlled pair** có test result | Artifact HF; nhiều seed |
| LMSCE raw/morphology/ring/consensus + strength | FCOS R50-FPN | 768 | 30 epoch, seed 42 | Có best-validation artifact HF | Test split; nhiều seed |

Artifact hiện tại chủ yếu là **single-seed (42)**. Không method nào đã được
xác minh đầy đủ ở cả 512 và 768 với ít nhất ba seed. `guided_alignment` và
`hard_transport` mới dừng ở mức code/config đối với LEVIR-Ship; chưa có
metric test công khai để xác nhận run hoàn tất.

## 2. `dbss` — Dynamic Background Subspace Suppression

### Method và purpose

**Purpose.** DBSS nhắm vào trường hợp tín hiệu tàu nhỏ ở P3 bị lẫn với các mẫu
nền lặp lại như mặt biển, sóng và texture. Ý tưởng là ước lượng một không gian
con của nền riêng cho từng ảnh, tách thành phần P3 được giải thích bởi nền rồi
dùng residual để hiệu chỉnh đặc trưng.

**Luồng xử lý.**

1. FPN tạo P3; DBSS chỉ sửa P3, giữ P4–P7 và interface của FCOS.
2. P3 được embed, lấy các token nền hợp lệ và xây nhiều background basis động
   theo từng ảnh.
3. `ridge` chiếu token lên các basis bằng ridge regression; `softmax` dùng tổ
   hợp prototype theo attention/temperature.
4. Residual `token - background_projection` được ghép với P3 để dự đoán hướng
   hiệu chỉnh. Một hệ số bị chặn bởi `gamma_max` giới hạn độ dịch chuyển, giúp
   module khởi đầu gần identity.
5. `ridge_haar` bổ sung độ lớn ba dải chi tiết Haar như tín hiệu reliability
   khi xác định biên độ correction.

**Chi tiết phép tính.** Gọi feature P3 là
\(X\in\mathbb{R}^{C\times H\times W}\). DBSS dùng conv 1×1 và LayerNorm để
thu được embedding \(E\in\mathbb{R}^{d\times H\times W}\), với mặc định
\(d=64\). Trên từng ảnh:

1. Phần feature hợp lệ (không tính padding) được adaptive-average-pool thành
   lưới 8×8, tạo 64 candidate \(c_j\). Điểm đại diện của candidate là cosine
   similarity trung bình với mọi token đã chuẩn hóa:
   \[
   s_j=\frac{1}{HW}\sum_i
   \left\langle\frac{c_j}{\lVert c_j\rVert},
   \frac{e_i}{\lVert e_i\rVert}\right\rangle.
   \]
2. Lấy shortlist 24 candidate có \(s_j\) cao nhất. Basis đầu là candidate tốt
   nhất; các basis sau tối đa hóa
   \(s_j-\beta\max_{b\in B}\cos(c_j,b)\), đồng thời ưu tiên cosine không vượt
   ngưỡng 0.9. `legacy_forced_k` luôn bù đủ 8 basis; `variable_k` dừng khi
   không còn candidate đủ khác biệt.
3. Với ma trận token \(T\in\mathbb{R}^{N\times d}\) và basis
   \(B\in\mathbb{R}^{K\times d}\), ridge projection được code tính bằng:
   \[
   A=(BB^\top+\lambda I)^{-1}BT^\top,\qquad
   \widehat T_{\mathrm{bg}}=A^\top B,\qquad
   R=T-\widehat T_{\mathrm{bg}},
   \]
   với \(\lambda=10^{-3}\). Phép giải chạy FP32, retry với regularization
   \(10^{-2}\), rồi fallback `lstsq` nếu hệ vẫn suy biến.
4. Với softmax projection:
   \[
   w_{ij}=\operatorname{softmax}_j
   \left(\cos(t_i,b_j)/\tau\right),\quad
   \widehat t_i=\sum_j w_{ij}b_j,
   \]
   mặc định \(\tau=0.1\).
5. `direction([X;R])` sinh tensor hướng \(D\); `magnitude(R[,Haar])` sinh
   \(g=\sigma(\cdot)\). Correction cuối được chặn theo RMS của feature:
   \[
   \Delta X=\gamma_{\max}\,
   \operatorname{RMS}_{ch}(X)\,g\,
   \frac{D}{1+\lVert D\rVert_2},\qquad X'=X+\Delta X.
   \]
   Conv cuối của nhánh direction được zero-init, nên lúc bắt đầu \(X'=X\).

DBSS không có loss riêng: basis selection, projection và correction được học
gián tiếp qua detection loss. Basis selection có thao tác top-k rời rạc, còn
embedding/direction/magnitude trên các basis đã chọn vẫn nhận gradient.

Các variant/control có trong code:

- `ridge`, `softmax`, `ridge_haar`; sweep `gamma_max` 0.3/0.6/1.0.
- `learned_control`: residual học trực tiếp, không dựa trên background basis.
- `random_bases`, `shuffled_bases`: phá ý nghĩa của basis để kiểm định cơ chế.
- `topk_only`: chỉ giữ selection/top-k, bỏ projection có ý nghĩa.
- `variable_k_ridge`: thay đổi số basis theo ảnh.

### Kết quả

Nguồn: [fcos_test_dbss](https://huggingface.co/datasets/duyle2408/fcos_test_dbss).
Các run dưới đây cùng FCOS/LEVIR-Ship, **768×768, 20 epoch, seed 42**;
baseline là `0.256/0.714/0.084`.

| Variant | mAP | AP50 | AP75 | Δ mAP | Δ AP50 | Δ AP75 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.256 | 0.714 | 0.084 | — | — | — |
| Ridge | 0.247 | 0.705 | 0.082 | -0.009 | -0.009 | -0.002 |
| Ridge + Haar | 0.267 | 0.717 | 0.110 | +0.011 | +0.003 | +0.026 |
| Softmax | 0.266 | 0.722 | 0.085 | +0.010 | +0.008 | +0.001 |
| Ridge, γ=0.3 | 0.271 | 0.730 | 0.119 | +0.015 | +0.016 | +0.035 |
| **Ridge, γ=0.6** | **0.282** | **0.740** | **0.123** | **+0.026** | **+0.026** | **+0.039** |
| Ridge, γ=1.0 | 0.269 | 0.729 | 0.113 | +0.013 | +0.015 | +0.029 |

Nguồn falsification:
[fcos_dbss_falsification](https://huggingface.co/datasets/duyle2408/fcos_dbss_falsification).
Đây là protocol riêng ở **768×768, 12 epoch, seed 42** (baseline thấp hơn),
nên chỉ so trong bảng này.

| Variant | mAP | AP50 | AP75 | Δ mAP |
|---|---:|---:|---:|---:|
| Baseline | 0.216 | 0.713 | 0.033 | — |
| Learned control | 0.263 | 0.730 | 0.090 | +0.047 |
| Random bases | 0.267 | 0.718 | 0.081 | +0.051 |
| Shuffled bases | 0.271 | 0.726 | 0.102 | +0.055 |
| Softmax control | 0.264 | 0.741 | 0.078 | +0.048 |
| Top-k only | 0.255 | 0.743 | 0.072 | +0.039 |
| Variable-k ridge | 0.220 | 0.687 | 0.042 | +0.004 |

**Nhận xét.** Sweep chính cho thấy γ=0.6 tốt nhất và cải thiện mạnh nhất ở
AP75. Tuy nhiên, random/shuffled/learned controls cũng vượt baseline trong
falsification. Vì vậy kết quả hiện tại ủng hộ việc “can thiệp residual có kiểm
soát tại P3” hơn là chứng minh riêng background-subspace projection là nguyên
nhân tạo cải thiện. Cần nhiều seed và control cùng protocol trước khi khẳng
định cơ chế DBSS. DBSS hiện **chỉ được xác minh ở 768**, chưa confirm rằng
γ=0.6 hoặc thứ hạng các variant còn giữ nguyên ở 512.

## 3. `guided_alignment` — DGFE + Adversarial Perturbation Injection

### Method và purpose

Branch `guided_alignment` hiện trỏ đúng commit nền `35cf22b1` và không có
commit riêng phía sau commit này. Phần triển khai có thể xác minh trong lịch
sử/repository là `FeatureAugmentNeck` với hai module:

- **FeatureDGFE (image-guided feature enhancement).** Upsample đặc trưng P3 để
  tái tạo RGB; sai khác tuyệt đối giữa ảnh tái tạo và ảnh input tạo spatial
  gate. Spatial gate được kết hợp với channel gate (average/max pooling +
  MLP), rồi nhân residual vào P3 qua hệ số `alpha` học được và bị chặn.
- **Adversarial Perturbation Injection (API).** Chỉ hoạt động khi train. Module
  giữ gradient của feature, chuẩn hóa nó thành perturbation có norm `rho`, chạy
  pass bị perturb và dùng auxiliary foreground loss. Mục đích là buộc detector
  ổn định trước nhiễu theo hướng bất lợi, đồng thời hướng attention về vùng có
  khả năng chứa foreground.

**Chi tiết DGFE.** Với \(X=P3\), module upsample \(X\) qua các `UpBlock` và
tái tạo ảnh ba kênh \(\widehat I\in[0,1]\). Ảnh input được min-max normalize
theo từng ảnh, sau đó:

\[
d=\operatorname{mean}_{RGB}|\widehat I-I|,\quad
S=1+\sigma(k(d-t)).
\]

Trong code, threshold khởi tạo \(t=0.0156862\), sharpness \(k=10\). Channel
gate là
\[
C=\sigma(\operatorname{MLP}(\operatorname{AvgPool}X)
       +\operatorname{MLP}(\operatorname{MaxPool}X)).
\]
Output là
\[
X'=X\odot[1+\alpha(C\odot S-1)],
\]
trong đó \(\alpha=\alpha_{\max}\sigma(a)\), khởi tạo khoảng \(10^{-3}\).
Vì vậy DGFE ban đầu gần identity; vùng khó tái tạo và kênh quan trọng mới được
tăng dần. Code không thêm reconstruction loss riêng cho \(\widehat I\);
reconstructor và gate được tối ưu thông qua detection loss.

**Chi tiết API.** Ở clean pass, module giữ feature \(X\). Từ tổng detection
loss cộng BCE của auxiliary foreground map, code lấy
\(g=\nabla_X(L_{\mathrm{det}}+L_{\mathrm{aux}})\), rồi tạo perturbation:
\[
\delta=\rho\,\frac{g}{\lVert g\rVert_2+\epsilon}.
\]
Một adversarial pass thứ hai chạy với \(X+\delta\). Tổng objective thực tế gồm
clean losses, `api_weight` nhân các detection losses của adversarial pass và
`api_weight × BCE` của auxiliary head. Mặc định module khai báo
\(\rho=0.02\), `api_weight=0.25`; inference bỏ hoàn toàn API.

DGFE phục vụ **feature alignment theo ảnh**: vùng P3 khó tái tạo từ ngữ cảnh
được tăng trọng số. API phục vụ **robust alignment khi train**. Khi inference,
API trả nguyên feature; DGFE vẫn tham gia inference.

### Kết quả

Có artifact DGFE+API công khai tại
[varroa_mmdet_runs_fcos_dgfe_api](https://huggingface.co/datasets/duyle2408/varroa_mmdet_runs_fcos_dgfe_api),
nhưng đó là thí nghiệm Varroa, không phải LEVIR-Ship. Ba repo baseline
[seed 42](https://huggingface.co/datasets/duyle2408/levir_ship_mmdet_runs_seed42),
[seed 43](https://huggingface.co/datasets/duyle2408/levir_ship_mmdet_runs_seed43)
và
[seed 44](https://huggingface.co/datasets/duyle2408/levir_ship_mmdet_runs_seed44)
không chứa run DGFE+API tương ứng. Do đó **không có kết quả LEVIR-Ship công
khai có thể xác minh cho branch này** và không chuyển số Varroa sang bảng so
sánh LEVIR.

**Nhận xét.** Cơ chế gần identity và API train-only giúp giảm rủi ro phá
baseline, nhưng branch không có diff độc lập và thiếu run LEVIR khớp protocol;
hiệu quả trên bài toán tàu nhỏ chưa được chứng minh bằng artifact hiện có.

**Đã test/chưa confirm.** Code LEVIR đặt resize 512×512, 12 epoch, seed 42,
nhưng đây chỉ là cấu hình dự kiến. Không có test JSON để xác nhận run
DGFE-only, API-only hay DGFE+API đã hoàn tất trên LEVIR. Artifact Varroa ba
seed không xác nhận khả năng chuyển sang LEVIR và cũng không xác nhận 768.

## 4. `haar` — Position-Aware Haar Recomposition (PAHR)

### Method và purpose

**Purpose.** PAHR cố gắng giữ và tái tổ hợp high-frequency detail của tàu nhỏ
thay vì để downsampling/FPN làm mờ chúng. Correction chỉ nên xuất hiện gần tàu,
vì khuếch đại dải cao trên toàn ảnh cũng khuếch đại sóng và nhiễu nền.

**PAHR-FCOS.**

1. Thực hiện Haar transform cố định trên P3 thành low-frequency và ba detail
   bands.
2. Một phase head dự đoán position probability và hai offset.
3. Position map được giám sát bằng Gaussian focal loss quanh tâm tàu; offset
   được học bằng Smooth L1 tại các tâm hợp lệ.
4. Position/offset gate điều khiển detail mixer; residual của ba detail bands
   được inverse-Haar về không gian P3 và cộng lại qua correction gain.
5. C2 guidance có thể được pixel-unshuffle/projection để cung cấp chi tiết
   không gian có độ phân giải cao hơn.

**Chi tiết Haar và recomposition.** Với mỗi block 2×2 của P3 gồm
\((a,b;c,d)\), transform trực chuẩn trong code là:
\[
L=(a+b+c+d)/2,\quad H=(a-b+c-d)/2,
\]
\[
V=(a+b-c-d)/2,\quad D=(a-b-c+d)/2.
\]
Locator đọc \([L,H,V,D]\), và nếu bật guidance thì đọc thêm C2 đã projection
và `pixel_unshuffle(4)`. `pixel_shuffle(2)` đưa output locator về lưới P3:
kênh đầu là position logit \(z\), hai kênh sau là offset
\((o_x,o_y)=\sigma(\cdot)\). Gate là
\[
p=\sigma(z),\quad q=q_{\min}+(1-q_{\min})p^\eta.
\]
Detail mixer nhận các Haar bands cùng phase context
`pixel_unshuffle([q, q·ox, q·oy], 2)` và sinh ba residual
\((\Delta H,\Delta V,\Delta D)\). Low-frequency correction bị cố định bằng
0; inverse Haar chỉ tái tổ hợp ba detail residual:
\[
X'=X+\kappa\,q\odot
\operatorname{IHaar}(0,\Delta H,\Delta V,\Delta D).
\]
Điều này giúp PAHR không trực tiếp thay đổi thành phần low-frequency của P3.

**Target và loss.** Chỉ object có căn bậc hai diện tích trong ảnh gốc không
vượt `tiny_max_sqrt_area` mới tạo target. Tại stride P3:

- Position target là max của các Gaussian tâm object, với
  \(\sigma_x,\sigma_y\) lấy theo kích thước box và clamp trong [0.5, 1.0].
- Offset target tại cell chứa tâm là phần lẻ
  \((x_c-\lfloor x_c\rfloor,y_c-\lfloor y_c\rfloor)\).
- `loss_pos` dùng Gaussian focal loss; `loss_offset` dùng Smooth L1 trên các
  tâm hợp lệ. Vùng ignored và padding không tham gia loss.
- `haar_shift_768` còn dùng offset dự đoán để dịch bbox prediction P3 về phía
  fractional center; PAHR cơ bản chỉ dùng offset để điều khiển correction.

Các ablation chính:

- `haar`: PAHR cơ bản.
- `haar_shift_768`: dùng position/offset để dịch correction tại 768×768.
- `haar_c2_wavelet_fusion_768`: Haar trên C2 rồi fusion vào P3.
- `faster_rcnn_pahr_p2_768`: áp dụng PAHR ở nhánh P2 của Faster R-CNN, so với
  Faster R-CNN P2 cùng độ phân giải.

P4–P7 và prediction interface giữ nguyên. Position/offset losses chỉ tồn tại
khi train; correction/gate đã học vẫn chạy khi inference.

### Kết quả

Nguồn: [fcos_test_haar](https://huggingface.co/datasets/duyle2408/fcos_test_haar).

**Các run FCOS hiện có (không phải tất cả đều cùng protocol).**

| Variant | Resolution | Epoch/seed | mAP | AP50 | AP75 | So sánh hợp lệ? |
|---|---:|---|---:|---:|---:|---|
| FCOS baseline | 768 | 40 / 42 | 0.272 | 0.735 | 0.100 | Baseline cho run 768 |
| PAHR (`haar`) | **512** | 40 / 42 | 0.259 | 0.720 | 0.088 | **Không**: thiếu FCOS-512 cùng protocol |
| PAHR shift | 768 | 40 / 42 | 0.258 | 0.718 | 0.096 | Có, Δ=-0.014/-0.017/-0.004 |
| Haar C2 wavelet fusion | 768 | 40 / 42 | 0.227 | 0.697 | 0.052 | So với FCOS-768, nhưng đồng thời đổi neck |

Không được lấy `0.259 - 0.272` để kết luận PAHR cơ bản giảm mAP, vì hai run
khác resolution. Với cùng 768, PAHR shift thấp hơn FCOS `-0.014 mAP`, còn C2
fusion thấp hơn `-0.045 mAP`; riêng C2 fusion cũng thay đổi neck nên chưa tách
được phần giảm do Haar transform hay do cách fusion C2.

**Nhóm Faster R-CNN P2 ở 768×768.**

| Variant | mAP | AP50 | AP75 | Δ mAP | Δ AP50 | Δ AP75 |
|---|---:|---:|---:|---:|---:|---:|
| Faster R-CNN P2 | 0.229 | 0.645 | 0.078 | — | — | — |
| **Faster R-CNN PAHR-P2** | **0.273** | **0.701** | **0.103** | **+0.044** | **+0.056** | **+0.025** |

**Nhận xét.** PAHR cơ bản 512 **chưa thể kết luận hơn/kém baseline** vì thiếu
FCOS-512 cùng protocol. Ở 768, PAHR shift và C2 fusion không vượt FCOS; C2
fusion giảm mạnh AP75. Ngược lại PAHR-P2 tăng rõ cả ba metric so với Faster
R-CNN P2 cùng protocol 768. Kết quả P2 mới có seed 42; chưa confirm ở 512
hoặc seed khác.

## 5. `hard_transport` — Dual-Irreducibility Hard Information Transport

### Method và purpose

**Purpose.** HIT tìm các vị trí P3 khó dự đoán theo cả không gian lẫn kênh,
xem chúng là “irreducible information”, rồi vận chuyển residual hiếm này về
phía object thay vì chỉ tăng cường tại chỗ.

**Luồng xử lý.**

1. `MaskedCenterConv2d` tái tạo mỗi vị trí từ lân cận nhưng che trọng số tâm,
   tạo spatial residual.
2. Channel reconstructor tạo channel residual. Tích năng lượng của hai
   residual tạo hard map: một vị trí chỉ mạnh khi khó tái tạo theo cả hai trục.
3. Chỉ top-k vị trí được giữ bởi sparse gate.
4. Offset head dự đoán vector dịch bị chặn bởi `max_offset`; Gaussian splatting
   đưa residual nguồn tới destination và projection zero-init cộng update vào
   P3.
5. Reconstruction losses được tính chủ yếu trên background; Smooth L1 offset
   loss hướng source về vùng/tâm object.

**Chi tiết irreducibility và transport.** Cho \(X=P3\), spatial reconstructor
là depthwise 3×3 với trọng số tâm luôn bị mask bằng 0, nên
\(\widehat X_s\) chỉ dùng tám láng giềng. Channel reconstructor dùng
pooling/MLP để tạo \(\widehat X_c\). Hai residual là
\(R_s=X-\widehat X_s\) và \(R_c=X-\widehat X_c\). Hard score dùng harmonic
mean của năng lượng hai residual:
\[
e_s=\operatorname{mean}_{ch}|R_s|,\quad
e_c=\operatorname{mean}_{ch}|R_c|,\quad
h=\frac{2e_se_c}{e_s+e_c+\epsilon}.
\]
Vì harmonic mean nhỏ nếu một trong hai residual nhỏ, điểm chỉ được xem là
“hard” khi khó tái tạo theo cả không gian và kênh. Score được chia cho mean
theo ảnh, clip bởi `hard_clip`, rồi chỉ giữ top `source_topq` (config chính:
1% vị trí).

Residual nguồn là
\[
S=\operatorname{Fuse}([R_s;R_c])\odot h\odot M_{\mathrm{topq}}.
\]
Offset head đọc \([X;h]\) và sinh
\(\Delta=\tanh(\cdot)\,\Delta_{\max}\), với \(\Delta_{\max}=8\) cell. Mỗi
source được splat tới \(u+\Delta_u\) trên lân cận 3×3 bằng Gaussian chuẩn hóa:
\[
w_{uv}\propto
\exp\left(-\frac{\lVert v-(u+\Delta_u)\rVert^2}{2\sigma^2}\right).
\]
Các contribution tới cùng destination được `scatter_add`; conv 1×1 zero-init
chiếu feature đã transport trước khi cộng vào P3. Do zero-init, HIT cũng bắt
đầu đúng như baseline.

**Loss.**

- \(L_{\text{recon-s}}=\lVert\widehat X_s-X\rVert_1\) và
  \(L_{\text{recon-c}}=\lVert\widehat X_c-X\rVert_1\), mặc định chỉ tính trên
  background (box và margin quanh box bị loại).
- Với source top-q nằm trong/giáp box, offset target là vector từ source tới
  tâm box, clamp bởi `max_offset`; \(L_{\text{offset}}\) là Smooth L1.
- Config chính dùng trọng số 0.1 cho cả ba loss. `detach_offset_input=True`
  chặn offset loss tác động ngược vào \(X,h\); stage `joint` bỏ detach.

Các stage/config:

- `probe`: tắt transport và offset loss để đo hard-map/reconstruction.
- `warmup`: chạy ngắn một epoch.
- `sparse_detached`: bật transport nhưng detach input của offset head.
- `joint`: bỏ detach để học end-to-end; có seed 42 và 43.

HIT sửa P3 của FCOS; các mức FPN khác giữ nguyên. Spatial/channel reconstruction
và offset supervision là train-time, còn hard selection và transport chạy khi
inference nếu `transport_enabled=True`.

### Kết quả

Qua danh sách dataset công khai của tài khoản
[duyle2408](https://huggingface.co/duyle2408), không tìm thấy repo/artifact có
tên HIT hoặc hard transport và không có đường dẫn Hugging Face trong branch
đủ để ánh xạ tới một run cụ thể. Vì vậy **không tìm thấy kết quả công khai có
thể xác minh cho `hard_transport`**; báo cáo không suy đoán metric từ checkpoint
name hoặc log không rõ nguồn.

**Nhận xét.** HIT có falsification path hợp lý (`probe` → detached → joint)
và zero-init giúp bắt đầu như baseline. Tuy nhiên top-k hard selection,
Gaussian splat và offset supervision tạo nhiều điểm có thể gây bất ổn; chưa
thể kết luận hiệu quả nếu thiếu artifact test.

**Đã test/chưa confirm.** Config chính đặt 512×512, 12 epoch, seed 42; có
config `warmup` một epoch, `probe`, `joint` seed 42 và `joint` seed 43. Tuy
nhiên không có test JSON/checkpoint artifact công khai ánh xạ được tới các
stage này. Report chỉ xác nhận **đã chuẩn bị cấu hình**, không xác nhận probe,
warmup, detached hay joint đã train/test thành công. Resolution 768,
multi-seed hoàn chỉnh và so sánh baseline đều chưa được kiểm chứng.

## 6. `rcfn_ltmr` — RCFN-R2, PG-RCFN và LTMR-L1

### Method và purpose

Ba hướng trong branch giải quyết hai vấn đề khác nhau:

- **RCFN-R2 (Residual Contrast Feature Normalization).** Chuẩn hóa local
  background quanh P3, trích local contrast/enhancement và cộng residual qua
  hệ số `gamma`. Purpose là làm tàu nhỏ nổi hơn so với nền cục bộ mà không đổi
  FCOS head.
- **PG-RCFN (Position-Guided RCFN).** Học heatmap Gaussian tại tâm tàu để gate
  enhancement RCFN. `PG-Aux` chỉ thêm position supervision; `PG-H` gate bằng
  heatmap H; `PG-CH` nhân H với channel/contrast gate C. Variant
  `PG-CH-w01-floor` dùng `0.1 + 0.9(H×C)` để gate không xóa hoàn toàn R2.
- **LTMR-L1 (Local Tiny-object Margin Regularization).** Chỉ trong training,
  tìm hard negatives quanh positive FCOS assignment của object nhỏ và ép
  positive logit cao hơn hard negative ít nhất một margin. Inference head
  không đổi; purpose là sửa lỗi ranking/classification cục bộ thay vì sửa FPN.

**Chi tiết RCFN-R2.** Với mỗi cell P3 và từng kênh, code tính mean/variance
FP32 trên đúng tám cell vòng 3×3, loại cell tâm:
\[
\mu_{\mathrm{ring}}=(9\mu_{3\times3}-X)/8,\quad
\sigma^2_{\mathrm{ring}}=
(9E_{3\times3}[X^2]-X^2)/8-\mu_{\mathrm{ring}}^2.
\]
Local standardized deviation là
\[
Z=(X-\mu_{\mathrm{ring}})/\sqrt{\sigma^2_{\mathrm{ring}}+\epsilon}.
\]
Depthwise 3×3 + SiLU + pointwise 1×1 biến \(Z\) thành enhancement \(E\), và:
\[
X'=X+\gamma\odot G\odot E.
\]
\(\gamma\) là tham số theo kênh, zero-init; R2 thuần dùng \(G=1\).

**Chi tiết PG-RCFN.** Position head kết hợp projection của P3 với P4 đã
upsample để dự đoán \(H=\sigma(\cdot)\). Với mỗi tiny object
(\(\sqrt{\mathrm{area}_{original}}\le16\)), target là Gaussian:
\[
T(x,y)=\max_k\exp\left(
-\frac{(x-x_k)^2}{2\sigma_{x,k}^2}
-\frac{(y-y_k)^2}{2\sigma_{y,k}^2}\right),
\]
với sigma tỉ lệ kích thước box theo stride và tối thiểu 1 cell. Position loss
là BCE trên vùng hợp lệ, reweight điểm gần tâm bằng
\(1+(w_{pos}-1)T\), mặc định \(w_{pos}=4\).

- `PG-Aux`: học \(H\) bằng loss trên nhưng enhancement vẫn có \(G=1\); dùng
  để kiểm tra lợi ích của supervision mà không cho gate can thiệp feature.
- `PG-H`: \(G=H\).
- `PG-CH`: dự đoán thêm contrast reliability
  \(C=\sigma(\operatorname{Conv}(|Z|))\), rồi \(G=H\cdot C\).
- Bản floor dùng \(G'=0.1+0.9G\); bản `w01` giảm trọng số position loss từ
  1.0 xuống 0.1.

**Chi tiết LTMR-L1.** Code khôi phục đúng FCOS assignment trên P3. Với mỗi GT
tiny có positive assignment, gọi \(\bar s^+\) là mean class logit của các
positive cell. Trong vùng box mở rộng thêm `radius × stride` (mặc định radius
2), module lấy các background cell hợp lệ, loại cell trong mọi GT/ignored
box, rồi lấy mean top-5 logit làm hard negative \(\bar s^-\). Loss mỗi object:
\[
L_{\mathrm{LTMR}}=
\operatorname{softplus}\left(m-(\bar s^+-\bar s^-)\right),
\]
với margin \(m=1\), trọng số tổng 0.05. Loss chỉ sửa classification logits
khi train; bbox regression, centerness và graph inference không đổi.

### Kết quả RCFN-R2 và LTMR-L1

Nguồn baseline test:
[levir_ship_mmdet_runs_seed42](https://huggingface.co/datasets/duyle2408/levir_ship_mmdet_runs_seed42);
nguồn candidates:
[fcos_test_rcfn_ltmr](https://huggingface.co/datasets/duyle2408/fcos_test_rcfn_ltmr).
Sử dụng test JSON mới nhất của R2/L1 trong repo candidate. R2/L1 chạy
**512×512, 30 epoch, seed 42**; baseline tham chiếu đến từ repo khác và không
phải controlled 30-epoch pair, nên delta dưới đây chỉ là tham khảo.

| Variant | mAP | AP50 | AP75 | Δ mAP | Δ AP50 | Δ AP75 |
|---|---:|---:|---:|---:|---:|---:|
| FCOS baseline | 0.233 | 0.685 | 0.070 | — | — | — |
| **RCFN-R2** | **0.260** | **0.717** | **0.091** | **+0.027** | **+0.032** | **+0.021** |
| LTMR-L1 | 0.233 | 0.683 | 0.061 | 0.000 | -0.002 | -0.009 |

Trong phép tham chiếu chéo artifact, R2 cao hơn cả ba metric còn LTMR-L1 không
tăng mAP. Tuy nhiên khác schedule/checkpoint-selection khiến bảng này chưa đủ
để khẳng định causal gain. Cần train FCOS baseline 512 đúng 30 epoch, seed 42
cùng pipeline trước khi xác nhận `+0.027 mAP`.

### Kết quả PG-RCFN

Nguồn:
[fcos_test_pg_rcfn](https://huggingface.co/datasets/duyle2408/fcos_test_pg_rcfn).
Để tránh trộn checkpoint/protocol, delta dùng run R2 nằm trong cùng repo này.
Tất cả run trong bảng là **FCOS 512×512, 30 epoch, seed 42**.

| Variant | mAP | AP50 | AP75 | Δ mAP | Δ AP50 | Δ AP75 |
|---|---:|---:|---:|---:|---:|---:|
| R2 reference | 0.244 | 0.701 | 0.080 | — | — | — |
| PG-Aux | 0.254 | 0.705 | 0.090 | +0.010 | +0.004 | +0.010 |
| PG-Aux, weight 0.1 | 0.256 | 0.707 | 0.099 | +0.012 | +0.006 | +0.019 |
| PG-H | 0.248 | 0.698 | 0.084 | +0.004 | -0.003 | +0.004 |
| **PG-CH** | **0.264** | **0.731** | **0.107** | **+0.020** | **+0.030** | **+0.027** |
| PG-CH, weight 0.1 + floor | 0.252 | 0.697 | 0.089 | +0.008 | -0.004 | +0.009 |

**Nhận xét.** R2 có kiến trúc đơn giản và cho metric tham khảo tốt, nhưng gain
so với FCOS chưa controlled. Trong PG ablation cùng protocol, PG-CH tốt nhất,
cho thấy position và channel/contrast cue có tính bổ sung. Tuy nhiên
floor/low-weight variant không giữ được mức tăng, và hai R2 artifact cho metric
khác nhau; chỉ nên kết luận trong từng protocol, chưa xem đây là kết quả
đa-seed. PG-RCFN chưa được confirm ở 768, seed 43/44 hoặc detector khác FCOS.

## 7. `open_close` — Local Morphological P3 Enhancement

### Ý tưởng và implementation ban đầu

Tiny ship được giả thuyết là local extremum trên một số channel P3. Module chỉ
sửa output index 0 của FPN (`start_level=1`, tương ứng P3 stride 8), giữ nguyên
P4–P7. Dilation và erosion được tính channel-wise bằng max pooling:

\[
D_k(X)=\operatorname{MaxPool}_k(X),\qquad
E_k(X)=-\operatorname{MaxPool}_k(-X).
\]

Positive top-hat dùng
\[
T^+=\operatorname{ReLU}(X-D_k(E_k(X))),
\]
còn negative top-hat dùng
\[
T^-=\operatorname{ReLU}(E_k(D_k(X))-X).
\]

Lượt đầu thử `positive`, `negative`, `both=T^++T^-` và full Conv3×3 control.
Residual ban đầu có dạng `P3 + gamma·mixer(input)`, với learnable
`gamma=0`. Mixer morphology là Conv1×1 C→C; control là full Conv3×3 C→C.

### Kết quả lượt đầu và confound

Các số dưới đây được đọc từ test output của live Marimo session đã mất; không
có upload Hugging Face nên chỉ xem là kết quả tạm thời, không phải artifact
công khai có thể tái kiểm tra.

| Variant | Best val epoch | Test mAP | Test AP50 | Test AP75 | Test AP-small |
|---|---:|---:|---:|---:|---:|
| Positive | 26 | 0.261 | 0.719 | **0.101** | **0.259** |
| Negative | 16 | 0.254 | 0.711 | 0.069 | 0.253 |
| Both | 10 | 0.255 | 0.711 | 0.091 | 0.254 |
| Conv3×3 | 9 | **0.261** | 0.718 | 0.092 | 0.258 |

Lượt này chưa falsify sạch morphology vì:

- Mixer dùng `bias=True`; top-hat bằng zero vẫn có thể tạo channel offset.
- `gamma=0` chặn gradient mixer ở iteration đầu, trong khi gamma học từ random
  projection chưa được tối ưu.
- Conv3×3 control có 590,080 tham số, gần chín lần mixer morphology 65,792
  tham số; hai nhánh không capacity-matched.
- `both` cộng hai residual không dấu trước cùng mixer, làm mất identity peak và
  hole. Negative-only đã giảm mạnh AP75 nên không tiếp tục nhánh này.

### Kết quả falsification matched control

Commit `4ad2a17f` thu hẹp API còn đúng hai mode:

\[
\begin{aligned}
\text{positive: }&P3'=P3+\operatorname{ZeroConv}_{1\times1}(T^+),\\
\text{raw: }&P3'=P3+\operatorname{ZeroConv}_{1\times1}(P3).
\end{aligned}
\]

Hai mixer đều C→C, `bias=False`, weight zero-init và có đúng 65,536 tham số.
Output ban đầu bằng chính xác baseline nhưng mixer nhận gradient ngay backward
đầu tiên. Negative, both, gamma và full Conv3×3 control đã bị xóa. Protocol là
FCOS R50-Caffe FPN, 768×768, 30 epoch, seed 42; chạy tuần tự positive rồi raw,
chọn best validation checkpoint và chỉ sau đó đánh giá test.

Session Marimo đầu tiên bị mất trước khi hoàn tất. Run được khởi động lại trong
work directory `morphology_matched_sha4ad2a17f_retry` và test tự động bằng best
validation checkpoint. Không có upload Hugging Face; các số dưới đây được lưu
trong `test_results.json` của live session.

| Variant | Test mAP | Test AP50 | Test AP75 | Test AP-small |
|---|---:|---:|---:|---:|
| **Positive top-hat** | **0.261** | **0.716** | **0.084** | **0.260** |
| Raw-P3 matched control | 0.246 | 0.705 | 0.074 | 0.244 |
| **Δ Positive** | **+0.015** | **+0.011** | **+0.010** | **+0.016** |

Positive thắng raw ở cả bốn metric và vượt gate định trước ở test mAP lẫn
AP-small. Kết quả này ủng hộ positive local-extrema input hơn một residual
Conv1×1 có cùng capacity và initialization. Tuy nhiên đây vẫn là một seed,
không có artifact HF và chưa so với baseline thuần trong chính lượt retry;
chưa đủ để claim gain tổng quát hay thêm Gaussian/multi-scale.

### LMSCE: morphology–contrast consensus

Commit
`6e479b1024c830c9c01bb80f1889c3d1f2c4b37d`
thêm **Local Morphological-Statistical Consensus Enhancement (LMSCE)** độc
lập với module morphology cũ. LMSCE vẫn chỉ sửa P3 và giữ P4–P7 nguyên vẹn.
Mọi mode dùng chung transformation có cùng capacity:

\[
\operatorname{DWConv}_{3\times3}\rightarrow\operatorname{SiLU}
\rightarrow\operatorname{PWConv}_{1\times1}
\rightarrow\operatorname{ZeroConv}_{1\times1}.
\]

Morphological opening dùng replicate padding tường minh cho từng erosion và
dilation. Ring statistics loại cell tâm trực tiếp bằng grouped convolution
3×3 có trọng số tâm bằng 0. Evidence chạy FP32 với variance floor
\(10^{-4}\):

\[
\widetilde M=\frac{\operatorname{ReLU}(X-\operatorname{Open}_3(X))}
{\sqrt{\max(\sigma_r^2,10^{-4})+10^{-6}}},
\qquad
Z=\operatorname{ReLU}\frac{X-\mu_r}
{\sqrt{\max(\sigma_r^2,10^{-4})+10^{-6}}},
\]

\[
A=\frac{2\widetilde MZ}{\widetilde M+Z+10^{-6}}.
\]

Bốn input ablation là raw \(X\), morphology \(\widetilde M\), ring \(Z\) và
consensus \(A\). Protocol dùng FCOS R50-Caffe FPN, 768×768, 30 epoch, seed 42.
Nguồn artifact:
[lmsce-p3-levir-ablation](https://huggingface.co/datasets/duyle2408/lmsce-p3-levir-ablation).
Các số sau là **best validation**, chưa phải test metrics.

| Variant | Best val mAP | Best val AP50 | Best val AP75 | Best val AP-small |
|---|---:|---:|---:|---:|
| Raw-P3 | 0.284 | **0.775** | 0.115 | 0.284 |
| Morphology-only | 0.282 | 0.772 | 0.104 | 0.283 |
| Ring-only | 0.286 | 0.766 | 0.106 | 0.285 |
| **Consensus** | **0.289** | 0.756 | **0.120** | **0.289** |

Consensus vượt raw `+0.005 mAP` và vượt cue đơn tốt nhất (ring) `+0.003 mAP`,
nên qua gate screening:
\[
\text{Consensus}>\text{Raw},\qquad
\text{Consensus}\ge\max(\text{Morphology},\text{Ring}).
\]
Tuy nhiên chênh lệch giữa các cue nhỏ và mới có một seed; kết quả chưa chứng
minh morphology và ring cung cấp information độc lập.

Norm correction đo trên một validation batch tại checkpoint consensus chỉ là
\(\lVert\Delta P3\rVert/\lVert P3\rVert\approx2.08\times10^{-5}\). Vì vậy
strength sweep được **fresh-train** từ đầu; kết quả post-hoc scaling không
được dùng để kết luận.

| Fresh-trained variant | Best val mAP | Best val AP50 | Best val AP75 | Best val AP-small | Δ mAP vs consensus |
|---|---:|---:|---:|---:|---:|
| Residual scale \(\alpha=2\) | 0.281 | 0.754 | 0.113 | 0.282 | -0.008 |
| Residual scale \(\alpha=4\) | 0.284 | 0.743 | 0.127 | 0.283 | -0.005 |
| **ZeroConv LR×5** | **0.298** | **0.807** | 0.112 | **0.297** | **+0.009** |
| ZeroConv LR×10 | 0.294 | 0.767 | **0.128** | 0.294 | +0.005 |

Tăng residual scale trực tiếp không tăng mAP. LR×5 tốt nhất về mAP/AP-small,
còn LR×10 cân bằng hơn cho strict localization vì đồng thời vượt consensus
`+0.005 mAP` và `+0.008 AP75`. Hai candidate này cần được test trên cùng test
split và chạy multi-seed trước khi chọn winner; không so các số validation này
trực tiếp với bảng test của matched top-hat ở trên.

## 8. So sánh các approach

| Branch / method | Vùng can thiệp | Cơ chế chính | Supervision bổ sung | Có tác động inference? | Mục tiêu |
|---|---|---|---|---|---|
| `dbss` / DBSS | P3 | Dynamic background bases, ridge/softmax projection, bounded residual | Không bắt buộc; học qua detection loss | Có | Loại thành phần nền, làm nổi residual của tàu |
| `guided_alignment` / DGFE | P3 | Image reconstruction error + spatial/channel gate | Tái tạo gián tiếp qua detection flow | Có | Tăng vùng feature khó khớp với ảnh |
| `guided_alignment` / API | P3 | Gradient-normalized adversarial perturbation | Auxiliary foreground BCE | Không | Tăng robustness khi train |
| `haar` / PAHR | P3 hoặc P2 | Haar decomposition, gated detail correction, inverse Haar | Gaussian position + offset loss | Có | Giữ/tái tổ hợp chi tiết tần số cao của tàu nhỏ |
| `hard_transport` / HIT | P3 | Dual residual, sparse top-k, offset + Gaussian splat | Reconstruction + offset loss | Có | Chuyển thông tin khó tái tạo về phía object |
| `rcfn_ltmr` / RCFN-R2 | P3 | Local-background standardization + residual contrast | Detection loss | Có | Tăng tương phản tàu so với nền cục bộ |
| `rcfn_ltmr` / PG-RCFN | P3 | Gaussian position gate và channel/contrast gate | Gaussian focal position loss | Có | Chỉ áp enhancement gần vị trí tàu |
| `rcfn_ltmr` / LTMR-L1 | FCOS logits | Positive-vs-local-hard-negative margin | Local margin loss | Không | Cải thiện ranking của tàu nhỏ khi train |
| `open_close` / positive top-hat | P3 | Channel-wise opening + positive local-extrema residual | Detection loss | Có | So local positive anomaly với raw-P3 matched control |
| `open_close` / LMSCE | P3 | Consensus giữa positive top-hat và standardized ring contrast | Detection loss | Có | Chỉ enhance vị trí được hai local cue cùng xác nhận |

## 9. Kết luận

- Kết quả mạnh nhất trong sweep DBSS 768/seed-42 là ridge γ=0.6
  (`mAP=0.282`), nhưng chưa test 512/multi-seed và falsification chưa chứng
  minh lợi ích đến riêng từ background subspace.
- PAHR cơ bản mới có run 512 nhưng thiếu FCOS-512 baseline. Ở 768, PAHR shift
  và C2 fusion không vượt FCOS; PAHR-P2 tăng `+0.044 mAP` trên controlled
  Faster R-CNN P2 pair, mới ở seed 42.
- RCFN-R2 có chênh lệch tham khảo `+0.027 mAP`, chưa phải controlled baseline
  pair. PG-CH tốt hơn R2 `+0.020 mAP` trong controlled ablation 512/seed-42.
  LTMR-L1 chưa cho thấy lợi ích.
- Chưa có kết quả LEVIR-Ship công khai xác minh được cho `guided_alignment`
  và `hard_transport`; hiện chỉ xác nhận được code/config intended ở 512.
- Morphology lượt đầu chỉ ngang Conv3×3 ở test mAP và bị ba confound lớn. Trong
  matched control, positive thắng raw `+0.015 mAP` và `+0.016 AP-small`, đạt
  gate định trước; cần baseline thuần và nhiều seed trước khi claim tổng quát.
- LMSCE consensus đạt best validation `0.289 mAP`, vượt raw `+0.005`. Strength
  sweep cho LR×5 cao nhất (`0.298 mAP`), còn LR×10 cải thiện cân bằng mAP/AP75
  (`0.294/0.128`). Đây mới là screening validation seed 42; chưa có controlled
  test comparison hoặc multi-seed.
- Các bảng chủ yếu là single-run/single-seed. Bước xác nhận tối thiểu trước
  khi chọn approach là chạy lại baseline và candidate tốt nhất trên cùng
  protocol với ít nhất ba seed, rồi báo mean ± standard deviation.
