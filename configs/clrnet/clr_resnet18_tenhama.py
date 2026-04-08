# CLRNet config for Tenryu-Hamanako front-facing images (640x640)
# Changes from clr_resnet18_culane.py:
#   - ori_img_w/h = 640
#   - cut_height  = 0
#   - sample_y    = range(639, 319, -20)
#   - max_lanes   = 4

net = dict(type='Detector')

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

num_points = 72
max_lanes = 4
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

work_dirs = "work_dirs/clr/r18_tenhama"

neck = dict(type='FPN',
            in_channels=[128, 256, 512],
            out_channels=64,
            num_outs=3,
            attention=False)

test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)

epochs = 15
batch_size = 4

optimizer = dict(type='AdamW', lr=0.6e-3)
total_iter = 1000
scheduler = dict(type='CosineAnnealingLR', T_max=total_iter)

eval_ep = 1
save_ep = 10

# Normalization: same as CULane (BGR, ImageNet-style)
img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])

ori_img_w = 640
ori_img_h = 640
img_w = 800
img_h = 320
cut_height = 0

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

dataset_path = '../../test_tenhama_15pics'
dataset_type = 'Tenhama'
dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        processes=val_process,
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

workers = 0
log_interval = 10
num_classes = 4 + 1
ignore_label = 255
bg_weight = 0.4
lr_update_by_epoch = False
seed = 0
gpus = 1
view = True
load_from = 'pretrained/culane_r18.pth'
