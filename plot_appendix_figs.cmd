python plot_appendix_figs.py curves `
  --x time `
  --title "ImageNet-1K@192 / ViT / Validation Top-1 vs Time" `
  --ylabel "Validation Top-1 (%)" `
  --out figure/app_in1k_vit_curve.pdf `
  --threshold 58.26 `
  --inset `
  --inset-xmin 2.4 `
  --inset-xmax 4.6 `
  --inset-ymin 56.0 `
  --inset-ymax 59.5 `
  --ymin 0 `
  --ymax 63.5 `
  --series "ViT-Base=test\outputs_pswf_paper_20260225\imageNet\VIT\in1k192_vit_A3_ch32\in1k192_vit_A3_ch32_metrics.jsonl" `
  --series "PSF-Pool=test\outputs_pswf_paper_20260225\imageNet\VIT\in1k192_vit_W3_poolonly_ch32\in1k192_vit_W3_poolonly_ch32_metrics.jsonl" `
  --series "PSF-HeadWarmup-Mul=test\outputs_pswf_paper_20260225\imageNet\VIT\in1k192_vit_W3_improved_warmup_ch32_fuse_multiply\in1k192_vit_W3_improved_warmup_ch32_fuse_multiply_metrics.jsonl" `
  --series "PSF-HeadMod=test\outputs_pswf_paper_20260225\imageNet\VIT\in1k192_vit_W3_residual_ch32\in1k192_vit_W3_residual_ch32_metrics.jsonl"


python plot_appendix_figs.py timebar `
  --title "ImageNet-1K@192 / ViT / Time-to-Accuracy" `
  --ylabel "Time to 58.26% Top-1 (hours)" `
  --out figure/app_in1k_vit_time_at_t.pdf `
  --value "ViT-Base=4.4" `
  --value "PSF-Pool=2.7" `
  --value "PSF-HeadWarmup-Mul=3.1" `
  --value "PSF-HeadMod=3.2"



 python plot_appendix_figs.py curves `
  --x time `
  --title "ImageNet-1K@192 / VIL / Validation Top-1 vs Time" `
  --ylabel "Validation Top-1 (%)" `
  --out figure/app_in1k_vil_curve.pdf `
  --threshold 67.96 `
  --inset `
  --inset-xmin 9.8 `
  --inset-xmax 13.8 `
  --inset-ymin 67.0 `
  --inset-ymax 68.8 `
  --ymin 0 `
  --ymax 70.5 `
  --series "ViL-Base=test\outputs_pswf_paper_20260225\imageNet\VIL\in1k192_vil_A1_ch32_reg\in1k192_vil_A1_ch32_reg_metrics.jsonl" `
  --series "PSF-Pool=test\outputs_pswf_paper_20260225\imageNet\VIL\in1k192_vil_W3_poolonly_ch32_reg\in1k192_vil_W3_poolonly_ch32_reg_metrics.jsonl" `
  --series "PSF-HeadWarmup-Mul=test\outputs_pswf_paper_20260225\imageNet\VIL\in1k192_vil_W3_improved_warmup_ch32_reg_fuse_multiply\in1k192_vil_W3_improved_warmup_ch32_reg_fuse_multiply_metrics.jsonl" `
  --series "PSF-HeadMod=test\outputs_pswf_paper_20260225\imageNet\VIL\in1k192_vil_W3_residualonly_ch32_reg\in1k192_vil_W3_residualonly_ch32_reg_metrics.jsonl"



python plot_appendix_figs.py timebar `
  --title "ImageNet-1K@192 / ViL / Time-to-Accuracy" `
  --ylabel "Time to 67.96% Top-1 (hours)" `
  --out figure/app_in1k_vit_time_at_t.pdf `
  --value "ViL-Base=13.3" `
  --value "PSF-Pool=10.9" `
  --value "PSF-HeadMod=10.5" `
  --value "PSF-HeadWarmup-Mul=10.9"



python plot_appendix_figs.py curves `
  --x epoch `
  --title "Tiny-ImageNet / ViL / Validation Top-1 vs Epoch" `
  --ylabel "Validation Top-1 (%)" `
  --out figure/app_tiny_vil_reg_curve.pdf `
  --ymin 0 `
  --ymax 48 `
  --inset `
  --inset-xmin 180 `
  --inset-xmax 300 `
  --inset-ymin 45.0 `
  --inset-ymax 47.2 `
  --series "ViL-Base=test\outputs_pswf_paper_20260225\开正则\VIL\tiny_vil_A1_ch32_patch8_reg\tiny_vil_A1_ch32_patch8_reg_metrics.jsonl" `
  --series "PSF-Pool=test\outputs_pswf_paper_20260225\开正则\VIL\tiny_vil_W3_poolonly_ch32_patch8_reg\tiny_vil_W3_poolonly_ch32_patch8_reg_metrics.jsonl" `
  --series "PSF-HeadMod=test\outputs_pswf_paper_20260225\开正则\VIL\tiny_vil_W3_residualonly_ch32_patch8_reg\tiny_vil_W3_residualonly_ch32_patch8_reg_metrics.jsonl" `
  --series "PSF-Both=test\outputs_pswf_paper_20260225\开正则\VIL\tiny_vil_W3_add_ch32_patch8_reg\tiny_vil_W3_add_ch32_patch8_reg_metrics.jsonl" `
  --series "PSF-TokenWavelet=test\outputs_pswf_paper_20260225\开正则\VIL\tiny_vil_W3_tokenonly_ch32_patch8_reg\tiny_vil_W3_tokenonly_ch32_patch8_reg_metrics.jsonl" `
  --series "PSF-HeadWarmup-Add=test\outputs_pswf_paper_20260225\开正则\VIL\tiny_vil_W3_improved_warmup_ch32_patch8_reg\tiny_vil_W3_improved_warmup_ch32_patch8_reg_metrics.jsonl" `
  --series "PSF-HeadWarmup-Mul=test\outputs_pswf_paper_20260225\开正则\VIL\tiny_vil_W3_improved_warmup_ch32_patch8_reg_fuse_multiply\tiny_vil_W3_improved_warmup_ch32_patch8_reg_fuse_multiply_metrics.jsonl"



  python plot_appendix_figs.py curves `
  --x epoch `
  --title "Tiny-ImageNet / ViT / Validation Top-1 vs Epoch" `
  --ylabel "Validation Top-1 (%)" `
  --out figure/app_tiny_vit_reg_curve.pdf `
  --ymin 0 `
  --ymax 51 `
  --inset `
  --inset-xmin 180 `
  --inset-xmax 300 `
  --inset-ymin 48.0 `
  --inset-ymax 49.7 `
  --series "ViT-Base=test\outputs_pswf_paper_20260225\开正则\VIT\tiny_vit_A3_ch32_patch8_reg\tiny_vit_A3_ch32_patch8_reg_metrics.jsonl" `
  --series "PSF-Pool=test\outputs_pswf_paper_20260225\开正则\VIT\tiny_vit_W3_poolonly_ch32_patch8_reg\tiny_vit_W3_poolonly_ch32_patch8_reg_metrics.jsonl" `
  --series "PSF-HeadMod=test\outputs_pswf_paper_20260225\开正则\VIT\tiny_vit_W3_residual_ch32_patch8_reg\tiny_vit_W3_residual_ch32_patch8_reg_metrics.jsonl" `
  --series "PSF-Both=test\outputs_pswf_paper_20260225\开正则\VIT\tiny_vit_W3_add_ch32_patch8_reg\tiny_vit_W3_add_ch32_patch8_reg_metrics.jsonl" `
  --series "PSF-TokenWavelet=test\outputs_pswf_paper_20260225\开正则\VIT\tiny_vit_W3_tokenonly_ch32_patch8_reg\tiny_vit_W3_tokenonly_ch32_patch8_reg_metrics.jsonl" `
  --series "PSF-HeadWarmup-Add=test\outputs_pswf_paper_20260225\开正则\VIT\tiny_vit_W3_improved_warmup_ch32_patch8_reg\tiny_vit_W3_improved_warmup_ch32_patch8_reg_metrics.jsonl" `
  --series "PSF-HeadWarmup-Mul=test\outputs_pswf_paper_20260225\开正则\VIT\tiny_vit_W3_improved_warmup_ch32_patch8_reg_fuse_multiply\tiny_vit_W3_improved_warmup_ch32_patch8_reg_fuse_multiply_metrics.jsonl"




python plot_appendix_figs.py curves `
  --x epoch `
  --title "Tiny-ImageNet / ViL / Validation Top-1 vs Epoch" `
  --ylabel "Validation Top-1 (%)" `
  --out figure/app_tiny_vil_noreg_curve.pdf `
  --ymin 0 `
  --ymax 31 `
  --inset `
  --inset-xmin 150 `
  --inset-xmax 300 `
  --inset-ymin 25 `
  --inset-ymax 30.5 `
  --series "ViL-Base=test\outputs_pswf_paper_20260225\关正则\VIL\tiny_vil_A1_ch32_patch8_noreg\tiny_vil_A1_ch32_patch8_noreg_metrics.jsonl" `
  --series "PSF-Pool=test\outputs_pswf_paper_20260225\关正则\VIL\tiny_vil_W3_poolonly_ch32_patch8_noreg\tiny_vil_W3_poolonly_ch32_patch8_noreg_metrics.jsonl" `
  --series "PSF-HeadMod=test\outputs_pswf_paper_20260225\关正则\VIL\tiny_vil_W3_residualonly_ch32_patch8_noreg\tiny_vil_W3_residualonly_ch32_patch8_noreg_metrics.jsonl" `
  --series "PSF-Both=test\outputs_pswf_paper_20260225\关正则\VIL\tiny_vil_W3_add_ch32_patch8_noreg\tiny_vil_W3_add_ch32_patch8_noreg_metrics.jsonl" `
  --series "PSF-TokenWavelet=test\outputs_pswf_paper_20260225\关正则\VIL\tiny_vil_W3_tokenonly_ch32_patch8_noreg\tiny_vil_W3_tokenonly_ch32_patch8_noreg_metrics.jsonl" `
  --series "PSF-HeadWarmup-Add=test\outputs_pswf_paper_20260225\关正则\VIL\tiny_vil_W3_improved_warmup_ch32_patch8_noreg\tiny_vil_W3_improved_warmup_ch32_patch8_noreg_metrics.jsonl" `
  --series "PSF-HeadWarmup-Mul=test\outputs_pswf_paper_20260225\关正则\VIL\tiny_vil_W3_improved_warmup_ch32_patch8_noreg_fuse_multiply\tiny_vil_W3_improved_warmup_ch32_patch8_noreg_fuse_multiply_metrics.jsonl"



  python plot_appendix_figs.py curves `
  --x epoch `
  --title "Tiny-ImageNet / ViT / Validation Top-1 vs Epoch" `
  --ylabel "Validation Top-1 (%)" `
  --out figure/app_tiny_vit_noreg_curve.pdf `
  --ymin 0 `
  --ymax 37.5 `
  --inset `
  --inset-xmin 170 `
  --inset-xmax 300 `
  --inset-ymin 32.0 `
  --inset-ymax 36.0 `
  --series "ViT-Base=test\outputs_pswf_paper_20260225\关正则\VIT\tiny_vit_A3_ch32_patch8_noreg\tiny_vit_A3_ch32_patch8_noreg_metrics.jsonl" `
  --series "PSF-Pool=test\outputs_pswf_paper_20260225\关正则\VIT\tiny_vit_W3_poolonly_ch32_patch8_noreg\tiny_vit_W3_poolonly_ch32_patch8_noreg_metrics.jsonl" `
  --series "PSF-HeadMod=test\outputs_pswf_paper_20260225\关正则\VIT\tiny_vit_W3_residual_ch32_patch8_noreg\tiny_vit_W3_residual_ch32_patch8_noreg_metrics.jsonl" `
  --series "PSF-Both=test\outputs_pswf_paper_20260225\关正则\VIT\tiny_vit_W3_add_ch32_patch8_noreg\tiny_vit_W3_add_ch32_patch8_noreg_metrics.jsonl" `
  --series "PSF-TokenWavelet=test\outputs_pswf_paper_20260225\关正则\VIT\tiny_vit_W3_tokenonly_ch32_patch8_noreg\tiny_vit_W3_tokenonly_ch32_patch8_noreg_metrics.jsonl" `
  --series "PSF-HeadWarmup-Add=test\outputs_pswf_paper_20260225\关正则\VIT\tiny_vit_W3_improved_warmup_ch32_patch8_noreg\tiny_vit_W3_improved_warmup_ch32_patch8_noreg_metrics.jsonl" `
  --series "PSF-HeadWarmup-Mul=test\outputs_pswf_paper_20260225\关正则\VIT\tiny_vit_W3_improved_warmup_ch32_patch8_noreg_fuse_multiply\tiny_vit_W3_improved_warmup_ch32_patch8_noreg_fuse_multiply_metrics.jsonl"


python plot_appendix_figs.py bars `
  --title "Tiny-ImageNet-C / ViT / Grouped Robustness" `
  --out figure/app_tinyc_vit_grouped.pdf `
  --series "ViT-Base=test\outputs_pswf_paper_20260225\tiny-imagenet-c\VIT\eval_tinyc_vit_A3_ch32_patch8_reg\eval_tinyc_vit_A3_ch32_patch8_reg_imagenet_c.json" `
  --series "PSF-Pool=test\outputs_pswf_paper_20260225\tiny-imagenet-c\VIT\eval_tinyc_vit_W3_poolonly_ch32_patch8_reg\eval_tinyc_vit_W3_poolonly_ch32_patch8_reg_imagenet_c.json" `
  --series "PSF-TokenWavelet=test\outputs_pswf_paper_20260225\tiny-imagenet-c\VIT\eval_tinyc_vit_W3_tokenonly_ch32_patch8_reg\eval_tinyc_vit_W3_tokenonly_ch32_patch8_reg_imagenet_c.json" `
  --series "PSF-HeadMod=test\outputs_pswf_paper_20260225\tiny-imagenet-c\VIT\eval_tinyc_vit_W3_residual_ch32_patch8_reg\eval_tinyc_vit_W3_residual_ch32_patch8_reg_imagenet_c.json" `
  --series "PSF-Both=test\outputs_pswf_paper_20260225\tiny-imagenet-c\VIT\eval_tinyc_vit_W3_add_ch32_patch8_reg\eval_tinyc_vit_W3_add_ch32_patch8_reg_imagenet_c.json" `
  --series "PSF-HeadWarmup-Add=test\outputs_pswf_paper_20260225\tiny-imagenet-c\VIT\eval_tinyc_vit_W3_improved_warmup_ch32_patch8_reg\eval_tinyc_vit_W3_improved_warmup_ch32_patch8_reg_imagenet_c.json" `
  --series "PSF-HeadWarmup-Mul=test\outputs_pswf_paper_20260225\tiny-imagenet-c\VIT\eval_tinyc_vit_W3_improved_warmup_ch32_patch8_reg_fuse_multiply\eval_tinyc_vit_W3_improved_warmup_ch32_patch8_reg_fuse_multiply_imagenet_c.json"




python plot_appendix_figs.py bars `
  --title "Tiny-ImageNet-C / ViL / Grouped Robustness" `
  --out figure/app_tinyc_vil_grouped.pdf `
  --series "ViT-Base=test\outputs_pswf_paper_20260225\tiny-imagenet-c\VIL\eval_tinyc_vil_A1_ch32_patch8_reg\eval_tinyc_vil_A1_ch32_patch8_reg_imagenet_c.json" `
  --series "PSF-Pool=test\outputs_pswf_paper_20260225\tiny-imagenet-c\VIL\eval_tinyc_vil_W3_poolonly_ch32_patch8_reg\eval_tinyc_vil_W3_poolonly_ch32_patch8_reg_imagenet_c.json" `
  --series "PSF-TokenWavelet=test\outputs_pswf_paper_20260225\tiny-imagenet-c\VIL\eval_tinyc_vil_W3_tokenonly_ch32_patch8_reg\eval_tinyc_vil_W3_tokenonly_ch32_patch8_reg_imagenet_c.json" `
  --series "PSF-HeadMod=test\outputs_pswf_paper_20260225\tiny-imagenet-c\VIL\eval_tinyc_vil_W3_residualonly_ch32_patch8_reg\eval_tinyc_vil_W3_residualonly_ch32_patch8_reg_imagenet_c.json" `
  --series "PSF-Both=test\outputs_pswf_paper_20260225\tiny-imagenet-c\VIL\eval_tinyc_vil_W3_add_ch32_patch8_reg\eval_tinyc_vil_W3_add_ch32_patch8_reg_imagenet_c.json" `
  --series "PSF-HeadWarmup-Add=test\outputs_pswf_paper_20260225\tiny-imagenet-c\VIL\eval_tinyc_vil_W3_improved_warmup_ch32_patch8_reg\eval_tinyc_vil_W3_improved_warmup_ch32_patch8_reg_imagenet_c.json" `
  --series "PSF-HeadWarmup-Mul=test\outputs_pswf_paper_20260225\tiny-imagenet-c\VIL\eval_tinyc_vil_W3_improved_warmup_ch32_patch8_reg_fuse_multiply\eval_tinyc_vil_W3_improved_warmup_ch32_patch8_reg_fuse_multiply_imagenet_c.json"




   python plot_appendix_figs.py curves `
  --x time `
  --title "ImageNet-1K@192 / VIL / Validation Top-1 vs Time" `
  --ylabel "Validation Top-1 (%)" `
  --out figure/app_in1k_vil_curve_continue.pdf `
  --series "ViL-Base=test\outputs_pswf_paper_20260225\imageNet\continue\in1k192_vil_A1_ch32_reg_continue_metrics.jsonl" `
  --series "PSF-Pool=test\outputs_pswf_paper_20260225\imageNet\continue\in1k192_vil_W3_poolonly_ch32_reg_continue_metrics.jsonl" `
  --series "PSF-HeadWarmup-Mul=test\outputs_pswf_paper_20260225\imageNet\continue\in1k192_vil_W3_improved_warmup_ch32_reg_fuse_multiply_continue_metrics.jsonl" `
  --series "PSF-HeadMod=test\outputs_pswf_paper_20260225\imageNet\continue\in1k192_vil_W3_residualonly_ch32_reg_continue_metrics.jsonl"


python plot_appendix_figs.py curves `
  --x time `
  --title "ImageNet-1K@192 / ViT / Validation Top-1 vs Time" `
  --ylabel "Validation Top-1 (%)" `
  --out figure/app_in1k_vit_curve_continue.pdf `
  --series "ViT-Base=test\outputs_pswf_paper_20260225\imageNet\continue\in1k192_vit_A3_ch32_continue_metrics.jsonl" `
  --series "PSF-Pool=test\outputs_pswf_paper_20260225\imageNet\continue\in1k192_vit_W3_poolonly_ch32_continue_metrics.jsonl" `
  --series "PSF-HeadWarmup-Mul=test\outputs_pswf_paper_20260225\imageNet\continue\in1k192_vit_W3_improved_warmup_ch32_fuse_multiply_continue_metrics.jsonl" `
  --series "PSF-HeadMod=test\outputs_pswf_paper_20260225\imageNet\continue\in1k192_vit_W3_residual_ch32_continue_metrics.jsonl"
