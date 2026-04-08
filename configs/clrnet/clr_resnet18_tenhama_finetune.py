# CLRNet fine-tune config for Tenryu-Hamanako railway images
# Run on external GPU (Colab T4 or university cluster):
#
#   python main.py configs/clrnet/clr_resnet18_tenhama_finetune.py \
#       --gpus 0 --finetune_from pretrained/culane_r18.pth
#
# Prepare dataset first:
#   python src/labelme_to_culane.py \
#       --input test_tenhama_15pics/ --output culane_tenhama/

net = dict(type='Detector')

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

num_points = 72
max_lanes = 2          # left rail + right rail only
sample_y = range(639, 319, -20)

heads = dict(type='CLRHead',
             num_priors=192,
             refine_layers=3,
             fc_hidden_dim=64,
             sample_points=36)

iou_loss_weight = 2.
cls_loss_weight = 2.
xyt_loss_weight = 0.2
seg_loss_weight = 1.0

work_dirs = "work_dirs/clr/r18_tenhama_finetune"

neck = dict(type='FPN',
            in_channels=[128, 256, 512],
            out_channels=64,
            num_outs=3,
            attention=False)

test_parameters = dict(conf_threshold=0.3, nms_thres=30, nms_topk=max_lanes)

epochs = 50
batch_size = 4          # fits in 6 GB VRAM; use 8 on Colab T4 (16 GB)

optimizer = dict(type='AdamW', lr=1e-4)   # lower lr for fine-tuning
total_iter = (50 // batch_size) * epochs  # rough estimate; will be overridden
scheduler = dict(type='CosineAnnealingLR', T_max=total_iter)

eval_ep = 5
save_ep = 10

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])

ori_img_w = 640
ori_img_h = 640
img_w = 800
img_h = 320
cut_height = 0

train_process = [
    dict(
        type='GenerateLaneLine',
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.3),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.5),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.05, 0.05),
                                                        y=(-0.05, 0.05)),
                                 rotate=(-5, 5),
                                 scale=(0.9, 1.1)),
                 p=0.5),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],
    ),
    dict(type='ToTensor', keys=['img', 'lane_line', 'seg']),
]

val_process = [
    dict(type='GenerateLaneLine',
         transforms=[
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ],
         training=False),
    dict(type='ToTensor', keys=['img']),
]

dataset_path = 'culane_tenhama'   # relative to CLRNet/; created by labelme_to_culane.py
dataset_type = 'Tenhama'
dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='train',
        processes=train_process,
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        processes=val_process,
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        processes=val_process,
    ))

workers = 2
log_interval = 10
num_classes = 2 + 1    # left_rail + right_rail + background
ignore_label = 255
bg_weight = 0.4
lr_update_by_epoch = False
seed = 0
gpus = 1
view = False
finetune_from = 'pretrained/culane_r18.pth'
