# Literature survey for YOLO tracking-mode edge perception characterization

**This survey identifies 39 papers across six research domains directly relevant to profiling YOLO tracking-mode perception on edge devices for urban crowd monitoring.** The literature collectively reveals a critical gap: no existing study systematically measures how progressive resolution reduction differentially degrades detection versus tracking metrics on edge hardware. Papers on tracking stability under imperfect detection provide strong theoretical and empirical support for the hypothesis that track continuity fails at higher resolutions than detection itself, while edge deployment studies establish the hardware performance baselines against which tracking-mode profiling should be compared.

---

## Domain 1: YOLO variant edge deployment studies (2020–2026)

This domain provides the hardware performance baselines—FPS, latency, power—against which tracking-mode overhead must be measured. Seven papers span the exact device classes targeted in the paper (Raspberry Pi 5, Jetson Nano, Jetson Orin Nano).

**Paper 1.1** — I. Lazarevich, M. Grimaldi, R. Kumar, S. Mitra, S. Khan, and S. Sah, "YOLOBench: Benchmarking Efficient Object Detectors on Embedded Systems," *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops*, pp. 1169–1178, 2023. arXiv:2307.13901.

YOLOBench is the **largest controlled cross-architecture YOLO benchmark for edge hardware**, profiling 550+ YOLO variants (YOLOv3 through YOLOv8 at 12 depth-width configurations × 11 input resolutions) across x86 CPU, ARM CPU (Raspberry Pi 4), GPU (Jetson Nano), and NPU (Khadas VIM3). Pareto-optimality analysis reveals that hardware choice significantly affects which YOLO variant is optimal—YOLOv6 dominates on VPU while YOLOv5 excels on ARM CPUs. The resolution sweep across 11 input sizes directly informs how detection accuracy varies with resolution on edge hardware. **Cite as:** the primary prior benchmark for controlled YOLO architecture comparison across edge platforms and resolutions; reference for methodology of Pareto-optimal latency-accuracy analysis.

**Paper 1.2** — R. C. C. de M. Santos, M. C. Silva, and R. A. Oliveira, "Real-time Object Detection Performance Analysis Using YOLOv7 on Edge Devices," *IEEE Latin America Transactions*, vol. 22, no. 10, 2024.

This paper benchmarks YOLOv7-tiny on three platforms directly relevant to the target study: Raspberry Pi 4B (**0.9 FPS**, CPU-only), Jetson Nano (**7.4 FPS** at 10W, 5.2 FPS at 5W), and Jetson Xavier NX (30 FPS across three power modes). The study uniquely profiles different NVIDIA power modes and reveals that maximum CPU clock frequency impacts FPS more than GPU clock on Xavier NX. **Cite as:** prior edge benchmark establishing baseline FPS expectations for YOLO-tiny models on Jetson Nano with power-mode profiling methodology.

**Paper 1.3** — H. Feng, G. Mu, S. Zhong, P. Zhang, and T. Yuan, "Benchmark Analysis of YOLO Performance on Edge Intelligence Devices," *Cryptography*, vol. 6, no. 2, p. 16, 2022. DOI: 10.3390/cryptography6020016.

Benchmarks YOLOv3/v3-tiny/v4/v4-tiny on Jetson Nano, Jetson Xavier NX, and Raspberry Pi 4B with Intel NCS2 (Movidius Myriad X VPU). Key finding: model format conversion for the NCS2 caused severe accuracy degradation for YOLOv3-tiny (0% mean confidence versus 57.9% on Jetson), highlighting that **cross-platform deployment introduces accuracy risks** beyond pure latency tradeoffs. **Cite as:** early systematic comparison of heterogeneous accelerator types (GPU vs. VPU) for YOLO inference, reference for model-conversion challenges across edge platforms.

**Paper 1.4** — L. Rey, A. M. Bernardos, A. D. Dobrzycki, D. Carramiñana, L. Bergesio, J. A. Besada, and J. R. Casar, "A Performance Analysis of You Only Look Once Models for Deployment on Constrained Computational Edge Devices in Drone Applications," *Electronics*, vol. 14, no. 3, p. 638, 2025. DOI: 10.3390/electronics14030638.

The most directly hardware-relevant study, profiling **YOLOv8n and YOLOv8s on Jetson Orin Nano, Jetson Orin NX, and Raspberry Pi 5** with TensorRT (FP32/FP16/INT8) and NCNN. YOLOv8n INT8 on Orin NX achieves **~66 FPS** at 0.179 J/inference versus RPi 5 at 1.498 J/inference for YOLOv8s FP32. Energy measurements show Orin Nano draws 7.4–8.7W. **Cite as:** primary reference for YOLOv8 quantization effects on the exact Jetson Orin-series and RPi 5 hardware used in the target paper; establishes detection-only power/latency baselines that tracking overhead will exceed.

**Paper 1.5** — D. K. Alqahtani, M. A. Cheema, and A. N. Toosi, "Benchmarking Deep Learning Models for Object Detection on Edge Computing Devices," *Service-Oriented Computing (ICSOC 2024)*, LNCS vol. 15404, pp. 148–163, Springer, 2025. DOI: 10.1007/978-981-96-0805-8_11.

Evaluates YOLOv8 (Nano/Small/Medium), EfficientDet Lite, and SSD across **seven device configurations**: RPi 3, 4, and 5 (with and without Coral USB TPU) and Jetson Orin Nano. Uniquely measures energy consumption alongside inference time and mAP, finding that lower-mAP models (SSD) are more energy-efficient while **YOLOv8 Medium consumes the most energy**. Adding Coral TPU to RPi dramatically improves SSD but less so for YOLO. **Cite as:** broadest device coverage benchmark with energy metrics; reference for hardware selection tradeoffs when choosing detection backbone for tracking pipelines.

**Paper 1.6** — D. G. Lema, R. Usamentiaga, and D. F. García, "Quantitative comparison and performance evaluation of deep learning-based object detection models on edge computing devices," *Integration*, vol. 95, p. 102127, 2024. DOI: 10.1016/j.vlsi.2023.102127.

Evaluates YOLOv3, YOLOv5, and YOLOX on Jetson Nano, Jetson AGX Xavier, and Google Coral Dev Board with actual power consumption via external instrumentation. One of the few works to include **cost-performance economic analysis** alongside hardware benchmarking. **Cite as:** prior benchmark with external power measurement methodology; relevant for justifying hardware selection for scalable urban monitoring deployments.

**Paper 1.7** — P. Kang and A. Somtham, "An Evaluation of Modern Accelerator-Based Edge Devices for Object Detection Applications," *Mathematics*, vol. 10, no. 22, p. 4299, 2022. DOI: 10.3390/math10224299.

Evaluates YOLOv4-Tiny and SSD on Jetson Nano, Xavier NX, and Coral Dev Board Mini, establishing a **three-dimensional evaluation framework (accuracy × latency × energy)**. Xavier NX achieves best energy-per-inference despite higher absolute power. **Cite as:** methodological reference for multi-dimensional edge evaluation frameworks that the target paper extends with tracking-specific metrics.

---

## Domain 2: Multi-object tracking on resource-constrained devices (2020–2026)

This domain covers the tracker algorithms and their edge deployment, providing both algorithmic baselines and edge-specific system designs.

**Paper 2.1** — Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, and X. Wang, "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," *Proceedings of the European Conference on Computer Vision (ECCV)*, pp. 1–21, Springer, 2022. DOI: 10.1007/978-3-031-20047-2_1. arXiv:2110.06864.

ByteTrack introduces the BYTE association method that tracks **every detection box including low-confidence ones** through two-stage IoU matching—high-confidence detections matched first, then low-confidence detections matched to remaining tracklets. Achieves **80.3 MOTA, 77.3 IDF1, 63.1 HOTA on MOT17** at 30 FPS on V100. Its purely motion-based association (no deep Re-ID) makes the tracker computationally negligible beyond the detector, which is critical for edge deployment where the detector is the throughput bottleneck. The paper demonstrates that recovering low-confidence detections improves IDF1 by 1–10 points across nine different tracker architectures. **Cite as:** the tracker backend in the target paper's YOLO track mode; foundational reference for the association algorithm; evidence that detection confidence thresholds suitable for detection evaluation cause systematic track fragmentation.

**Paper 2.2** — J. Cao, J. Pang, X. Weng, R. Khirodkar, and K. Kitani, "Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking," *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 9686–9696, 2023. arXiv:2203.14360.

OC-SORT addresses Kalman filter error accumulation during occlusion with three observation-centric innovations (ORU, OCM, OCR). Like ByteTrack, it requires **no deep appearance model—running at 700+ FPS on CPU** with off-the-shelf detections. Achieves state-of-the-art on MOT17, MOT20, KITTI, and DanceTrack. Critically provides **analytical proof** that 1-pixel position variance accumulated over 10 frames produces drift exceeding bounding box dimensions. **Cite as:** alternative lightweight tracker for edge deployment comparison; analytical evidence that small localization errors compound into association failures.

**Paper 2.3** — N. Aharon, R. Orfaig, and B.-Z. Bobrovsky, "BoT-SORT: Robust Associations Multi-Pedestrian Tracking," arXiv:2206.14651, 2022. (Highly cited, 1000+ citations.)

BoT-SORT extends ByteTrack with camera motion compensation, revised Kalman filter state vector, and IoU-ReID fusion. Achieves **80.5 MOTA, 80.2 IDF1, 65.0 HOTA on MOT17**—the highest across all primary metrics. However, the Re-ID component adds substantial computational cost, creating a **quality-versus-throughput tradeoff** particularly acute on edge devices. **Cite as:** accuracy ceiling achievable with appearance-enhanced trackers; contextualizes why ByteTrack (motion-only) may be preferred for edge deployment despite BoT-SORT's superior server-class metrics.

**Paper 2.4** — X. Li, C. Chen, Y. Lou, M. Abdallah, K. T. Kim, and S. Bagchi, "HopTrack: A Real-time Multi-Object Tracking System for Embedded Devices," arXiv:2411.00608, 2024.

HopTrack is specifically designed for embedded devices, targeting Jetson AGX Xavier. It introduces content-aware dynamic frame sampling and lightweight pixel-intensity features for association. Achieves **63.12% MOTA on MOT16 at 39.29 FPS** on AGX Xavier—outperforming ByteTrack-Embed by 2.15% MOTA while reducing energy consumption by 20.8% and memory by 8%. The paper documents that **SOTA trackers designed for server GPUs achieve less than 11 FPS on embedded devices**. **Cite as:** state-of-the-art edge-MOT system; reference for the performance gap between server-grade and edge-grade tracking and for energy/power/memory constraints.

**Paper 2.5** — J. Müller and A. Pigors, "Efficient Multi-Object Tracking on Edge Devices via Reconstruction-Based Channel Pruning," arXiv:2410.08769, 2024.

Proposes reconstruction-based channel pruning using Dependency Graphs for compressing JDE (Joint Detection and Embedding) MOT networks for deployment on **Jetson Orin Nano**. Achieves up to **70% parameter reduction** while maintaining tracking accuracy on MOT20. Simultaneously prunes both detection and re-identification components. **Cite as:** model compression strategy for MOT on the exact Orin Nano hardware; evidence that pruning is essential for bridging server-class MOT accuracy and edge computational budgets.

**Paper 2.6** — G. Di Fabrizio, L. Calisti, C. Contoli, N. Kania, and E. Lattanzi, "A Study on the Energy-Efficiency of the Object Tracking Algorithms in Edge Devices," *Proceedings of the IEEE/ACM 16th International Conference on Utility and Cloud Computing (UCC '23)*, pp. 1–6, 2023. DOI: 10.1145/3603166.3632541.

Provides systematic energy consumption analysis of detection+tracking pipelines on Google Coral AI and Jetson Nano. Key finding: tracking algorithm choice impacts total energy budget by **9% to nearly 300%** depending on configuration. The tracking phase energy cost—often overlooked—can be comparable to or exceed the detection phase for feature-rich trackers. **Cite as:** evidence that ByteTrack's lightweight association is energy-favorable; supports positioning in the IoT/edge domain where power budget is a first-class constraint.

**Paper 2.7** — (Di Fabrizio et al. research group), "A power-aware vision-based virtual sensor for real-time edge computing," *Journal of Real-Time Image Processing*, vol. 21, art. 103, Springer, 2024. DOI: 10.1007/s11554-024-01482-0.

Proposes Dynamic Inference Power Manager (DIPM) that adapts inference frame rate based on scene dynamicity on Jetson Nano. Achieves **~36% energy reduction** in low-dynamicity scenes with tracking accuracy degradation below 1.2%. Frames detection+tracking as a "virtual sensor" paradigm. **Cite as:** adaptive inference strategy for power-constrained edge deployments; directly relevant to YOLO tracking-mode deployment where scene dynamicity varies across urban monitoring scenarios.

---

## Domain 3: Resolution reduction effects on detection and tracking (2020–2026)

This domain is the most critical to the paper's central hypothesis. The literature establishes that resolution reduction degrades localization quality before it degrades detection recall, creating a mechanism for early tracking failure.

**Paper 3.1** — Y. Hao, H. Pei, Y. Lyu, Z. Yuan, J.-R. Rizzo, Y. Wang, and Y. Fang, "Understanding the Impact of Image Quality and Distance of Objects to Object Detection Performance," arXiv:2209.08237, 2022.

Systematically examines how spatial resolution, compression, and object distance affect detection accuracy using RA-YOLO (Resolution-Adaptive YOLOv5). **Small object AP drops far more sharply than large object AP** under resolution reduction, and an optimal resolution exists that balances accuracy and computational cost. **Cite as:** primary reference establishing resolution-dependent detection degradation with differential small/large object impact; supports the precondition that bounding box quality degrades before objects become undetectable.

**Paper 3.2** — L. Yang, Y. Han, X. Chen, S. Song, J. Dai, and G. Huang, "Resolution Adaptive Networks for Efficient Inference," *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2369–2378, 2020. arXiv:2003.07326.

RANet proposes adaptive resolution routing where "easy" inputs are processed at low resolution while "hard" inputs requiring spatial detail use high-resolution paths. The key insight is that **low-resolution representations suffice for detection (an "easy" task) but fine spatial detail is needed for distinguishing similar objects**—analogous to the detection-versus-association distinction in tracking. **Cite as:** theoretical support for why detection and association have different resolution requirements; directly motivates the hypothesis that detection degrades later than association under resolution reduction.

**Paper 3.3** — S. Jiang, Z. Lin, Y. Li, Y. Shu, and Y. Liu, "Flexible High-resolution Object Detection on Edge Devices with Tunable Latency," *Proceedings of the 27th ACM MobiCom*, pp. 1–14, 2022. DOI: 10.1145/3447993.3483274.

Remix demonstrates that naïve downsampling of 4K images to standard detector input sizes causes **catastrophic accuracy loss—up to 80% detection degradation**—because nearly all objects become "small" at reduced resolution. The paper shows that mAP improvements from stronger models plateau at higher resolutions, establishing that resolution is often the binding constraint rather than model capacity. **Cite as:** evidence that downsampling to practical edge resolutions causes massive detection degradation, especially for small objects; the cascading effect on tracking (identity switches from imprecise boxes, fragmentation from missed detections) occurs before overall recall drops.

**Paper 3.4** — X. Chen, N. Zeng, and L. Chen, "An effective method for small object detection in low-resolution images," *Engineering Applications of Artificial Intelligence*, vol. 130, 107028, 2024. DOI: 10.1016/j.engappai.2023.107028.

Quantifies that when resolution is reduced from 1333×800 to 608×608, **small object AP drops ~6.2 points more than large object AP** on COCO. Proposes YOLOFs with expanded receptive fields to mitigate this. **Cite as:** quantitative evidence for differential resolution sensitivity across object sizes—the mechanism underlying early tracking degradation, since small objects in tracking correspond to distant/occluded targets whose bounding box imprecision degrades IoU-based association.

**Paper 3.5** — A. Bayat and M. Pomplun, "Improving Performance of Object Detection Using the Multi-Resolution Faster-RCNN," arXiv:2301.09667, 2023.

Demonstrates that **standard detectors trained at one resolution suffer substantial mAP degradation at other resolutions**, with the performance gap widening at lower resolutions. Proposes multi-resolution training to mitigate this. **Cite as:** evidence that resolution mismatch between training and deployment causes detection degradation—a common edge scenario where resolution may be dynamically reduced.

**Paper 3.6** — J. Huang, V. Rathod, C. Sun, M. Zhu, A. Korattikara, A. Fathi, I. Fischer, Z. Wojna, Y. Song, S. Guadarrama, and K. Murphy, "Speed/Accuracy Trade-Offs for Modern Convolutional Object Detectors," *Proceedings of the IEEE CVPR*, pp. 3296–3305, 2017. DOI: 10.1109/CVPR.2017.351. (Foundational, 4000+ citations.)

The seminal study establishing that detection accuracy is strongly resolution-dependent. Critically shows that **mAP@0.75 drops faster than mAP@0.50** under resolution reduction—meaning localization precision degrades before detection recall. Since IoU-based tracking association depends on localization precision, this supports the hypothesis that track continuity breaks down at higher resolutions than detection recall. **Cite as:** foundational reference for resolution-accuracy relationships; the localization-precision finding directly supports the tracking degradation hypothesis. (Pre-2020 but essential.)

**Important gap identified:** No existing paper directly performs the experiment of progressively reducing resolution while measuring detection metrics and tracking metrics side-by-side. The evidence is inferential—each piece supports the hypothesis independently, but the complete causal chain from resolution reduction → localization degradation → IoU ambiguity → identity switches has not been demonstrated in a single controlled study. This represents the core novelty of the target paper.

---

## Domain 4: Pedestrian/crowd detection and tracking benchmarks (2020–2026)

These papers provide the evaluation framework, benchmark datasets, and metrics used in the target study.

**Paper 4.1** — P. Dendorfer, A. Ošep, A. Milan, K. Schindler, D. Cremers, I. Reid, S. Roth, and L. Leal-Taixé, "MOTChallenge: A Benchmark for Single-Camera Multiple Target Tracking," *International Journal of Computer Vision (IJCV)*, vol. 129, pp. 845–881, 2021. DOI: 10.1007/s11263-020-01393-0.

The definitive journal paper for the MOTChallenge benchmark suite covering MOT15/16/17. Analyzes **73+ trackers** with CLEAR MOT metrics (MOTA, MOTP) and IDF1. MOT17 extends MOT16 with three public detectors (DPM, Faster R-CNN, SDP) to test tracker robustness. Describes annotation protocol, visibility ratios, and sequence diversity. **Cite as:** primary citation for MOT17 benchmark protocol, evaluation metrics, and sequence characteristics.

**Paper 4.2** — J. Luiten, A. Ošep, P. Dendorfer, P. Torr, A. Geiger, and L. Leal-Taixé, "HOTA: A Higher Order Metric for Evaluating Multi-object Tracking," *International Journal of Computer Vision (IJCV)*, vol. 129, pp. 548–578, 2021. DOI: 10.1007/s11263-020-01375-2.

HOTA formally decomposes tracking performance into **Detection Accuracy (DetA), Association Accuracy (AssA), and Localization Accuracy (LocA)**. Demonstrates that MOTA correlates 0.96 with DetA but only 0.46 with AssA, while IDF1 correlates 0.97 with AssA but only 0.58 with DetA. The decomposition provides the exact mathematical vocabulary for the target paper's hypothesis: under resolution reduction, **AssA may degrade faster than DetA** because noisy bounding boxes still count as detections but produce ambiguous IoU overlaps. HOTA computed across multiple localization thresholds (α) makes explicit that localization accuracy affects association quality. **Cite as:** essential metric framework for separately measuring detection versus association degradation; the DetA/AssA decomposition is the analytical tool for demonstrating the paper's central claim.

**Paper 4.3** — P. Dendorfer, H. Rezatofighi, A. Milan, J. Shi, D. Cremers, I. Reid, S. Roth, K. Schindler, and L. Leal-Taixé, "MOT20: A Benchmark for Multi Object Tracking in Crowded Scenes," arXiv:2003.09003, 2020.

MOT20 targets extremely crowded scenes with density up to **246 pedestrians per frame** (average 170.9 in test set—approximately 4× denser than MOT17). Demonstrates that trackers performing well on MOT17 suffer substantial degradation in MOT20's dense scenarios. **Cite as:** contextualizes MOT17's moderate crowd density relative to extreme cases; motivates density-dependent performance analysis.

**Paper 4.4** — S. Shao, Z. Zhao, B. Li, T. Xiao, G. Yu, X. Zhang, and J. Sun, "CrowdHuman: A Benchmark for Detecting Human in a Crowd," arXiv:1805.00123, 2018. (1000+ citations.)

Contains **~470K human instances** across 24,370 images with ~22.6 persons per image, annotated with head, visible-region, and full-body bounding boxes. Models pre-trained on CrowdHuman achieve state-of-the-art on Caltech, CityPersons, and Brainwash. ByteTrack uses CrowdHuman for detector training. **Cite as:** pre-training dataset reference for the pedestrian detector; motivates robust crowd-scene detection as prerequisite for tracking quality.

**Paper 4.5** — G. Ciaparrone, F. Luque Sánchez, S. Tabik, L. Troiano, R. Tagliaferri, and F. Herrera, "Deep Learning in Video Multi-Object Tracking: A Survey," *Neurocomputing*, vol. 381, pp. 61–88, 2020. DOI: 10.1016/j.neucom.2019.11.023.

Comprehensive survey identifying four MOT pipeline stages (detection, feature extraction, affinity computation, association) with experimental comparison on MOTChallenge. Key finding: **detection quality is the primary bottleneck** for tracking performance. **Cite as:** survey establishing the tracking-by-detection paradigm and positioning the target work within the MOT taxonomy; reference for the claim that detection quality dominates tracking outcomes.

**Paper 4.6** — A. Milan, L. Leal-Taixé, I. Reid, S. Roth, and K. Schindler, "MOT16: A Benchmark for Multi-Object Tracking," arXiv:1603.00831, 2016.

Original MOT16 sequences and annotation protocol that MOT17 extends. Defines annotation standards including full-body bounding boxes, visibility ratios, pedestrian/distractor classes. **Cite as:** original source for MOT16/17 sequences; essential context about scene characteristics and camera configurations. (Pre-2020 but foundational.)

---

## Domain 5: Edge AI for urban surveillance and crowd monitoring (2020–2026)

These papers provide the application context, demonstrating why characterizing tracking-mode perception on edge devices matters for real-world crowd monitoring.

**Paper 5.1** — E. Badidi, K. Moumane, and F. E. Ghazi, "Opportunities, Applications, and Challenges of Edge-AI Enabled Video Analytics in Smart Cities: A Systematic Review," *IEEE Access*, vol. 11, pp. 80543–80572, 2023. DOI: 10.1109/ACCESS.2023.3300658.

PRISMA-methodology survey of 282 references covering edge-AI video analytics for smart cities. Classifies edge video analytics approaches and identifies key challenges including **limited computational resources, model optimization for inference under power constraints, and the gap between cloud-based training and edge deployment realities**. Notes Gartner prediction that 50% of inference will occur at the edge by 2025. **Cite as:** comprehensive survey positioning the work within edge-AI for smart cities; motivates why edge deployment characterization is critically important.

**Paper 5.2** — C. Neff, M. Mendieta, S. Mohan, M. Baharani, S. Rogers, and H. Tabkhi, "REVAMP²T: Real-Time Edge Video Analytics for Multicamera Privacy-Aware Pedestrian Tracking," *IEEE Internet of Things Journal*, vol. 7, no. 4, pp. 2591–2602, 2020. DOI: 10.1109/JIOT.2019.2955555.

Presents an integrated detection→re-identification→tracking pipeline on Jetson AGX Xavier for privacy-aware pedestrian tracking. Introduces the **Accuracy·Efficiency (Æ) metric** that jointly evaluates accuracy, throughput, and power efficiency—directly addressing the perception-analytics gap. Achieves up to 13-fold Æ improvement over prior art through system-level co-design. **Cite as:** the most directly relevant prior complete edge-deployed tracking pipeline; its Æ metric recognizes that detection accuracy alone is insufficient and must be evaluated jointly with throughput and power on actual hardware.

**Paper 5.3** — M. A. Ezzat, M. A. Abd El Ghany, S. Almotairi, and M. A.-M. Salem, "Horizontal Review on Video Surveillance for Smart Cities: Edge Devices, Applications, Datasets, and Future Trends," *Sensors*, vol. 21, no. 9, art. 3222, 2021. DOI: 10.3390/s21093222.

Bridges applications, algorithms, datasets, and edge hardware for smart city surveillance. Covers NVIDIA Jetson, Intel Movidius, and Google Coral platforms. Notes that **edge computing reduces latency from 150–200ms (cloud round-trip) to ~10ms** and identifies the gap between standalone algorithm benchmarks and integrated surveillance system requirements. **Cite as:** background on edge vision computing for smart cities; frames the challenge of translating benchmark-trained models to real-world urban monitoring.

**Paper 5.4** — Z. Chen, X. Xie, T. Qiu, and L. Yao, "Dense-stream YOLOv8n: A Lightweight Framework for Real-Time Crowd Monitoring in Smart Libraries," *Scientific Reports*, vol. 15, art. 11618, 2025. DOI: 10.1038/s41598-025-94659-x.

Optimizes YOLOv8n for crowd monitoring using DensityNet, pruning, and knowledge distillation. Achieves **mAP@0.5 of 0.99 with FPS increasing to 254** at only 4.0 GFLOP and 2.04M parameters. Explicitly addresses the gap between raw detection performance and practical crowd monitoring needs. **Cite as:** lightweight YOLO architecture for crowd analytics; demonstrates that model compression can maintain analytics-relevant accuracy while dramatically improving throughput.

**Paper 5.5** — S. Wang, Z. Pu, Q. Li, and Y. Wang, "Estimating Crowd Density with Edge Intelligence Based on Lightweight Convolutional Neural Networks," *Expert Systems with Applications*, vol. 206, art. 117823, 2022. DOI: 10.1016/j.eswa.2022.117823.

Proposes DICNN for crowd density map estimation on edge hardware, addressing the tension between computational demands and resource constraints. Uses dilated inception modules for scale-aware feature extraction with minimal parameters. **Cite as:** contrast between density-estimation and detection-based tracking approaches to crowd monitoring; highlights different accuracy/latency tradeoffs.

**Paper 5.6** — L. Sun, J. Sun, J. Zhang, et al., "Edge-Cloud Collaborative Video Analytics System for Crowd Gathering Event Detection in Metro Stations," *Tsinghua Science and Technology*, 2025. DOI: 10.26599/TST.2025.9010082.

Proposes EC-CGED for real-time metro station crowd event detection using edge-cloud collaboration. Demonstrates the **perception-analytics gap** concretely: camera-level detection (what edge nodes provide) must be fused with spatial context to produce station-level crowd analytics (what operators need). **Cite as:** deployed system showing the gap between raw detection/tracking outputs and actionable crowd management intelligence; motivates why characterizing tracking-mode perception quality matters for downstream analytics.

---

## Domain 6: Tracking stability and identity assignment under imperfect detection (2020–2026)

This domain provides the strongest theoretical and empirical support for the paper's central hypothesis. These papers collectively establish the mechanism: localization uncertainty → IoU ambiguity → association failure → identity switches, occurring before detection metrics indicate failure.

**Paper 6.1** — C. W. Lee and S. L. Waslander, "UncertaintyTrack: Exploiting Detection and Localization Uncertainty in Multi-Object Tracking," *2024 IEEE Intelligent Vehicles Symposium (IV)*, IEEE, 2024. DOI: 10.1109/IV55156.2024.10610458. arXiv:2402.12303.

Directly investigates how bounding-box localization uncertainty propagates into tracking failures. Shows that standard trackers **"blindly trust" detections with no sense of localization uncertainty**, causing erroneous associations. Four extensions (uncertainty-aware Kalman filter, confidence ellipse filtering, bounding box relaxation, entropy-based matching) **reduce identity switches by ~19%** on BDD100K. **Cite as:** the most direct evidence that localization uncertainty propagates through IoU-based association into identity switches; the 19% reduction from merely accounting for (not eliminating) uncertainty confirms that standard tracking is brittle to detection quality in ways detection metrics do not capture.

**Paper 6.2** — E. Solano-Carrillo, L. Porzi, S. R. Bulò, and T. Kontogianni, "UTrack: Multi-Object Tracking with Uncertain Detections," *ECCV 2024 Workshops*, LNCS vol. 15639, pp. 206–223, Springer, 2025. DOI: 10.1007/978-3-031-91585-7_14. arXiv:2408.17098.

**Formalizes the propagation of detection uncertainty through IoU computation**, deriving disambiguation methods for uncertain IoUs. Shows that IoU-based matching becomes fundamentally ambiguous when bounding box predictions carry uncertainty—two different track-detection assignments may have statistically indistinguishable IoU values, making Hungarian algorithm assignments unreliable. Demonstrates improvements on MOT17, MOT20, DanceTrack by properly accounting for uncertainty. **Cite as:** mathematical proof that IoU becomes ambiguous under localization uncertainty; explains exactly why tracking degrades before detection metrics indicate failure.

**Paper 6.3** — T. Mandel, M. Jimenez, E. Risley, H. Yee, J. Thomas, Y. Lu, M. Honda, and M. Maynord, "Detection confidence driven multi-object tracking to recover reliable tracks from unreliable detections," *Pattern Recognition*, vol. 135, 109107, 2023. DOI: 10.1016/S0031-3203(22)00587-8.

RCT is specifically designed for settings where detection quality is poor. Demonstrates that **standard MOT trackers exhibit dramatic performance collapse when detection quality degrades**, with identity switches increasing sharply—even when detection metrics still report adequate performance. Introduces the FISHTRAC benchmark for evaluating tracking under unreliable detection conditions. **Cite as:** direct experimental evidence that detection-only evaluation overestimates the viable operating range for tracking; shows that standard trackers fail when detection quality drops below a threshold invisible to detection metrics.

**Paper 6.4** — Y. Du, Z. Zhao, Y. Song, Y. Zhao, F. Su, T. Gong, and H. Meng, "StrongSORT: Make DeepSORT Great Again," *IEEE Transactions on Multimedia*, vol. 25, pp. 8291–8301, 2023. DOI: 10.1109/TMM.2023.3240881.

Systematically identifies that the vanilla Kalman filter in DeepSORT is **"vulnerable w.r.t. low-quality detections"** and that the feature bank is "sensitive to detection noise." Introduces NSA (Noise Scale Adaptive) Kalman filter and EMA feature updating. Ablation studies show that raw trajectory velocities **"jitter wildly due to detection noise"**—analogous to what happens at reduced resolutions where bounding box localization becomes noisier. **Cite as:** empirical evidence that Kalman filter and feature matching are primary failure modes under detection noise, with velocity estimation errors compounding into track fragmentation.

---

## Cross-domain synthesis and gap analysis

The literature surveyed reveals a clear and exploitable gap. The evidence chain supporting the paper's hypothesis assembles across three domains:

- **Resolution studies (Domain 3)** establish that resolution reduction degrades localization precision (mAP@0.75 drops before mAP@0.50) and disproportionately affects small objects, but no study measures the downstream tracking impact.
- **Tracking stability studies (Domain 6)** prove that localization uncertainty propagates through IoU-based association into identity switches, but these studies do not manipulate resolution as the source of uncertainty.
- **Edge deployment studies (Domains 1–2)** characterize detection-only throughput and power on the relevant hardware but do not profile tracking-mode operation or tracking-specific metrics.

**The target paper uniquely bridges all three**, performing the complete experiment: progressive resolution reduction → detection metric measurement → tracking metric measurement → differential degradation analysis, all on edge hardware with tracking-specific metrics. This positions it as the first controlled study demonstrating the full causal chain from resolution to tracking failure on resource-constrained devices.

Three additional observations for the paper's framing:

- **HOTA's DetA/AssA decomposition** (Paper 4.2) provides the ideal metric framework for demonstrating the hypothesis—showing AssA drops faster than DetA under resolution reduction would be the cleanest quantitative demonstration.
- **The "perception-analytics gap"** identified in Domain 5 papers (especially REVAMP²T's Æ metric) provides strong motivation: detection benchmarks that do not account for tracking-mode degradation give false confidence about edge system viability.
- **ByteTrack's two-stage association** (Paper 2.1) is both the tracker backend and part of the explanation for the phenomenon—its IoU-based first stage is exactly the mechanism through which localization degradation propagates into identity switches, as formalized by UTrack (Paper 6.2) and UncertaintyTrack (Paper 6.1).