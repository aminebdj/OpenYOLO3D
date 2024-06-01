_backend_args = None
_multiscale_resize_transforms = [
    dict(
        transforms=[
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                320,
                320,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    320,
                    320,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                960,
                960,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    960,
                    960,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
]
affine_scale = 0.9
albu_train_transforms = [
    dict(p=0.01, type='Blur'),
    dict(p=0.01, type='MedianBlur'),
    dict(p=0.01, type='ToGray'),
    dict(p=0.01, type='CLAHE'),
]
backend_args = None
base_lr = 0.002
batch_shapes_cfg = None
close_mosaic_epochs = 2
coco_val_dataset = dict(
    _delete_=True,
    class_text_path='data/texts/lvis_v1_class_texts.json',
    dataset=dict(
        ann_file='lvis/lvis_v1_minival_inserted_image_name.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img=''),
        data_root='data/coco/',
        test_mode=True,
        type='YOLOv5LVISV1Dataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(scale=(
            640,
            640,
        ), type='YOLOv5KeepRatioResize'),
        dict(
            allow_scale_up=False,
            pad_val=dict(img=114),
            scale=(
                640,
                640,
            ),
            type='LetterResize'),
        dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
        dict(type='LoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'pad_param',
                'texts',
            ),
            type='mmdet.PackDetInputs'),
    ],
    type='MultiModalDataset')
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=98,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=True,
                pad_val=dict(img=114.0),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.09999999999999998,
                    1.9,
                ),
                type='YOLOv5RandomAffine'),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.01, type='Blur'),
                    dict(p=0.01, type='MedianBlur'),
                    dict(p=0.01, type='ToGray'),
                    dict(p=0.01, type='CLAHE'),
                ],
                type='mmdet.Albu'),
            dict(type='YOLOv5HSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                max_num_samples=80,
                num_neg_samples=(
                    1203,
                    1203,
                ),
                padding_to_max=True,
                padding_value='',
                type='RandomLoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'yolo_world',
    ])
data_root = 'data/coco/'
dataset_type = 'YOLOv5CocoDataset'
deepen_factor = 1.0
default_hooks = dict(
    checkpoint=dict(
        interval=2,
        max_keep_ckpts=2,
        rule='greater',
        save_best='auto',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(
        lr_factor=0.01,
        max_epochs=100,
        scheduler_type='linear',
        type='YOLOv5ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
flickr_train_dataset = dict(
    ann_file='annotations/final_flickr_separateGT_train.json',
    data_prefix=dict(img='full_images/'),
    data_root='data/flickr/',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            img_scale=(
                640,
                640,
            ),
            pad_val=114.0,
            pre_transform=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            type='MultiModalMosaic'),
        dict(
            border=(
                -320,
                -320,
            ),
            border_val=(
                114,
                114,
                114,
            ),
            max_aspect_ratio=100,
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            scaling_ratio_range=(
                0.09999999999999998,
                1.9,
            ),
            type='YOLOv5RandomAffine'),
        dict(
            bbox_params=dict(
                format='pascal_voc',
                label_fields=[
                    'gt_bboxes_labels',
                    'gt_ignore_flags',
                ],
                type='BboxParams'),
            keymap=dict(gt_bboxes='bboxes', img='image'),
            transforms=[
                dict(p=0.01, type='Blur'),
                dict(p=0.01, type='MedianBlur'),
                dict(p=0.01, type='ToGray'),
                dict(p=0.01, type='CLAHE'),
            ],
            type='mmdet.Albu'),
        dict(type='YOLOv5HSVRandomAug'),
        dict(prob=0.5, type='mmdet.RandomFlip'),
        dict(
            max_num_samples=80,
            num_neg_samples=(
                1203,
                1203,
            ),
            padding_to_max=True,
            padding_value='',
            type='RandomLoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'flip',
                'flip_direction',
                'texts',
            ),
            type='mmdet.PackDetInputs'),
    ],
    type='YOLOv5MixedGroundingDataset')
img_scale = (
    640,
    640,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
last_stage_out_channels = 512
last_transform = [
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
load_from = '/home/jean/Amine/OpenYolo3D/pretrained/checkpoints/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_weight = 7.5
loss_cls_weight = 0.5
loss_dfl_weight = 0.375
lr_factor = 0.01
max_aspect_ratio = 100
max_epochs = 100
max_keep_ckpts = 2
mg_train_dataset = dict(
    ann_file='annotations/final_mixed_train_no_coco.json',
    data_prefix=dict(img='gqa/images/'),
    data_root='data/mixed_grounding/',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            img_scale=(
                640,
                640,
            ),
            pad_val=114.0,
            pre_transform=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            type='MultiModalMosaic'),
        dict(
            border=(
                -320,
                -320,
            ),
            border_val=(
                114,
                114,
                114,
            ),
            max_aspect_ratio=100,
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            scaling_ratio_range=(
                0.09999999999999998,
                1.9,
            ),
            type='YOLOv5RandomAffine'),
        dict(
            bbox_params=dict(
                format='pascal_voc',
                label_fields=[
                    'gt_bboxes_labels',
                    'gt_ignore_flags',
                ],
                type='BboxParams'),
            keymap=dict(gt_bboxes='bboxes', img='image'),
            transforms=[
                dict(p=0.01, type='Blur'),
                dict(p=0.01, type='MedianBlur'),
                dict(p=0.01, type='ToGray'),
                dict(p=0.01, type='CLAHE'),
            ],
            type='mmdet.Albu'),
        dict(type='YOLOv5HSVRandomAug'),
        dict(prob=0.5, type='mmdet.RandomFlip'),
        dict(
            max_num_samples=80,
            num_neg_samples=(
                1203,
                1203,
            ),
            padding_to_max=True,
            padding_value='',
            type='RandomLoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'flip',
                'flip_direction',
                'texts',
            ),
            type='mmdet.PackDetInputs'),
    ],
    type='YOLOv5MixedGroundingDataset')
mixup_prob = 0.15
model = dict(
    backbone=dict(
        image_model=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            arch='P5',
            deepen_factor=1.0,
            last_stage_out_channels=512,
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            type='YOLOv8CSPDarknet',
            widen_factor=1.25),
        text_model=dict(
            frozen_modules=[
                'all',
            ],
            model_name='openai/clip-vit-base-patch32',
            type='HuggingCLIPLanguageBackbone'),
        type='MultiModalYOLOBackbone'),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            embed_dims=512,
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                256,
                512,
                512,
            ],
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=80,
            reg_max=16,
            type='YOLOWorldHeadModule',
            use_bn_head=True,
            widen_factor=1.25),
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=7.5,
            reduction='sum',
            return_iou=False,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=0.5,
            reduction='none',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=0.375,
            reduction='mean',
            type='mmdet.DistributionFocalLoss'),
        prior_generator=dict(
            offset=0.5, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        type='YOLOWorldHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOWDetDataPreprocessor'),
    mm_neck=True,
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        deepen_factor=1.0,
        embed_channels=[
            128,
            256,
            256,
        ],
        guide_channels=512,
        in_channels=[
            256,
            512,
            512,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        num_heads=[
            4,
            8,
            8,
        ],
        out_channels=[
            256,
            512,
            512,
        ],
        type='YOLOWorldPAFPN',
        widen_factor=1.25),
    num_test_classes=1203,
    num_train_classes=80,
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=80,
            topk=10,
            type='BatchTaskAlignedAssigner',
            use_ciou=True)),
    type='YOLOWorldDetector')
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.7, type='nms'),
    nms_pre=30000,
    score_thr=0.001)
mosaic_affine_transform = [
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='Mosaic'),
    dict(
        border=(
            -320,
            -320,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine'),
]
neck_embed_channels = [
    128,
    256,
    256,
]
neck_num_heads = [
    4,
    8,
    8,
]
norm_cfg = dict(eps=0.001, momentum=0.03, type='BN')
num_classes = 1203
num_det_layers = 3
num_training_classes = 80
obj365v1_train_dataset = dict(
    class_text_path='data/texts/obj365v1_class_texts.json',
    dataset=dict(
        ann_file='annotations/objects365_train.json',
        data_prefix=dict(img='train/'),
        data_root='data/objects365v1/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        type='YOLOv5Objects365V1Dataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            img_scale=(
                640,
                640,
            ),
            pad_val=114.0,
            pre_transform=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            type='MultiModalMosaic'),
        dict(
            border=(
                -320,
                -320,
            ),
            border_val=(
                114,
                114,
                114,
            ),
            max_aspect_ratio=100,
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            scaling_ratio_range=(
                0.09999999999999998,
                1.9,
            ),
            type='YOLOv5RandomAffine'),
        dict(
            bbox_params=dict(
                format='pascal_voc',
                label_fields=[
                    'gt_bboxes_labels',
                    'gt_ignore_flags',
                ],
                type='BboxParams'),
            keymap=dict(gt_bboxes='bboxes', img='image'),
            transforms=[
                dict(p=0.01, type='Blur'),
                dict(p=0.01, type='MedianBlur'),
                dict(p=0.01, type='ToGray'),
                dict(p=0.01, type='CLAHE'),
            ],
            type='mmdet.Albu'),
        dict(type='YOLOv5HSVRandomAug'),
        dict(prob=0.5, type='mmdet.RandomFlip'),
        dict(
            max_num_samples=80,
            num_neg_samples=(
                1203,
                1203,
            ),
            padding_to_max=True,
            padding_value='',
            type='RandomLoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'flip',
                'flip_direction',
                'texts',
            ),
            type='mmdet.PackDetInputs'),
    ],
    type='MultiModalDataset')
optim_wrapper = dict(
    clip_grad=dict(max_norm=10.0),
    constructor='YOLOWv5OptimizerConstructor',
    optimizer=dict(
        batch_size_per_gpu=16, lr=0.002, type='AdamW', weight_decay=0.025),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys=dict({
            'backbone.text_model': dict(lr_mult=0.01),
            'logit_scale': dict(weight_decay=0.0)
        }),
        norm_decay_mult=0.0),
    type='OptimWrapper')
param_scheduler = None
persistent_workers = True
pre_transform = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]
resume = False
save_epoch_intervals = 2
strides = [
    8,
    16,
    32,
]
tal_alpha = 0.5
tal_beta = 6.0
tal_topk = 10
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        class_text_path='data/texts/lvis_v1_class_texts.json',
        dataset=dict(
            ann_file='lvis/lvis_v1_minival_inserted_image_name.json',
            batch_shapes_cfg=None,
            data_prefix=dict(img=''),
            data_root='data/coco/',
            test_mode=True,
            type='YOLOv5LVISV1Dataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='MultiModalDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/coco/lvis/lvis_v1_minival_inserted_image_name.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.LVISMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(type='LoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
text_channels = 512
text_transform = [
    dict(
        max_num_samples=80,
        num_neg_samples=(
            1203,
            1203,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
train_ann_file = 'annotations/instances_train2017.json'
train_batch_size_per_gpu = 16
train_cfg = dict(
    dynamic_intervals=[
        (
            98,
            1,
        ),
    ],
    max_epochs=100,
    type='EpochBasedTrainLoop',
    val_interval=10)
train_data_prefix = 'train2017/'
train_dataloader = dict(
    batch_size=16,
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(
        datasets=[
            dict(
                class_text_path='data/texts/obj365v1_class_texts.json',
                dataset=dict(
                    ann_file='annotations/objects365_train.json',
                    data_prefix=dict(img='train/'),
                    data_root='data/objects365v1/',
                    filter_cfg=dict(filter_empty_gt=False, min_size=32),
                    type='YOLOv5Objects365V1Dataset'),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        img_scale=(
                            640,
                            640,
                        ),
                        pad_val=114.0,
                        pre_transform=[
                            dict(backend_args=None, type='LoadImageFromFile'),
                            dict(type='LoadAnnotations', with_bbox=True),
                        ],
                        type='MultiModalMosaic'),
                    dict(
                        border=(
                            -320,
                            -320,
                        ),
                        border_val=(
                            114,
                            114,
                            114,
                        ),
                        max_aspect_ratio=100,
                        max_rotate_degree=0.0,
                        max_shear_degree=0.0,
                        scaling_ratio_range=(
                            0.09999999999999998,
                            1.9,
                        ),
                        type='YOLOv5RandomAffine'),
                    dict(
                        bbox_params=dict(
                            format='pascal_voc',
                            label_fields=[
                                'gt_bboxes_labels',
                                'gt_ignore_flags',
                            ],
                            type='BboxParams'),
                        keymap=dict(gt_bboxes='bboxes', img='image'),
                        transforms=[
                            dict(p=0.01, type='Blur'),
                            dict(p=0.01, type='MedianBlur'),
                            dict(p=0.01, type='ToGray'),
                            dict(p=0.01, type='CLAHE'),
                        ],
                        type='mmdet.Albu'),
                    dict(type='YOLOv5HSVRandomAug'),
                    dict(prob=0.5, type='mmdet.RandomFlip'),
                    dict(
                        max_num_samples=80,
                        num_neg_samples=(
                            1203,
                            1203,
                        ),
                        padding_to_max=True,
                        padding_value='',
                        type='RandomLoadText'),
                    dict(
                        meta_keys=(
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'flip',
                            'flip_direction',
                            'texts',
                        ),
                        type='mmdet.PackDetInputs'),
                ],
                type='MultiModalDataset'),
            dict(
                ann_file='annotations/final_flickr_separateGT_train.json',
                data_prefix=dict(img='full_images/'),
                data_root='data/flickr/',
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        img_scale=(
                            640,
                            640,
                        ),
                        pad_val=114.0,
                        pre_transform=[
                            dict(backend_args=None, type='LoadImageFromFile'),
                            dict(type='LoadAnnotations', with_bbox=True),
                        ],
                        type='MultiModalMosaic'),
                    dict(
                        border=(
                            -320,
                            -320,
                        ),
                        border_val=(
                            114,
                            114,
                            114,
                        ),
                        max_aspect_ratio=100,
                        max_rotate_degree=0.0,
                        max_shear_degree=0.0,
                        scaling_ratio_range=(
                            0.09999999999999998,
                            1.9,
                        ),
                        type='YOLOv5RandomAffine'),
                    dict(
                        bbox_params=dict(
                            format='pascal_voc',
                            label_fields=[
                                'gt_bboxes_labels',
                                'gt_ignore_flags',
                            ],
                            type='BboxParams'),
                        keymap=dict(gt_bboxes='bboxes', img='image'),
                        transforms=[
                            dict(p=0.01, type='Blur'),
                            dict(p=0.01, type='MedianBlur'),
                            dict(p=0.01, type='ToGray'),
                            dict(p=0.01, type='CLAHE'),
                        ],
                        type='mmdet.Albu'),
                    dict(type='YOLOv5HSVRandomAug'),
                    dict(prob=0.5, type='mmdet.RandomFlip'),
                    dict(
                        max_num_samples=80,
                        num_neg_samples=(
                            1203,
                            1203,
                        ),
                        padding_to_max=True,
                        padding_value='',
                        type='RandomLoadText'),
                    dict(
                        meta_keys=(
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'flip',
                            'flip_direction',
                            'texts',
                        ),
                        type='mmdet.PackDetInputs'),
                ],
                type='YOLOv5MixedGroundingDataset'),
            dict(
                ann_file='annotations/final_mixed_train_no_coco.json',
                data_prefix=dict(img='gqa/images/'),
                data_root='data/mixed_grounding/',
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        img_scale=(
                            640,
                            640,
                        ),
                        pad_val=114.0,
                        pre_transform=[
                            dict(backend_args=None, type='LoadImageFromFile'),
                            dict(type='LoadAnnotations', with_bbox=True),
                        ],
                        type='MultiModalMosaic'),
                    dict(
                        border=(
                            -320,
                            -320,
                        ),
                        border_val=(
                            114,
                            114,
                            114,
                        ),
                        max_aspect_ratio=100,
                        max_rotate_degree=0.0,
                        max_shear_degree=0.0,
                        scaling_ratio_range=(
                            0.09999999999999998,
                            1.9,
                        ),
                        type='YOLOv5RandomAffine'),
                    dict(
                        bbox_params=dict(
                            format='pascal_voc',
                            label_fields=[
                                'gt_bboxes_labels',
                                'gt_ignore_flags',
                            ],
                            type='BboxParams'),
                        keymap=dict(gt_bboxes='bboxes', img='image'),
                        transforms=[
                            dict(p=0.01, type='Blur'),
                            dict(p=0.01, type='MedianBlur'),
                            dict(p=0.01, type='ToGray'),
                            dict(p=0.01, type='CLAHE'),
                        ],
                        type='mmdet.Albu'),
                    dict(type='YOLOv5HSVRandomAug'),
                    dict(prob=0.5, type='mmdet.RandomFlip'),
                    dict(
                        max_num_samples=80,
                        num_neg_samples=(
                            1203,
                            1203,
                        ),
                        padding_to_max=True,
                        padding_value='',
                        type='RandomLoadText'),
                    dict(
                        meta_keys=(
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'flip',
                            'flip_direction',
                            'texts',
                        ),
                        type='mmdet.PackDetInputs'),
                ],
                type='YOLOv5MixedGroundingDataset'),
        ],
        ignore_keys=[
            'classes',
            'palette',
        ],
        type='ConcatDataset'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_num_workers = 8
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='MultiModalMosaic'),
    dict(
        border=(
            -320,
            -320,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        max_num_samples=80,
        num_neg_samples=(
            1203,
            1203,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        max_num_samples=80,
        num_neg_samples=(
            1203,
            1203,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=300, nms=dict(iou_threshold=0.65, type='nms')),
    type='mmdet.DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    transforms=[
                        dict(scale=(
                            640,
                            640,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                640,
                                640,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            320,
                            320,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                320,
                                320,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            960,
                            960,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                960,
                                960,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
            ],
            [
                dict(prob=1.0, type='mmdet.RandomFlip'),
                dict(prob=0.0, type='mmdet.RandomFlip'),
            ],
            [
                dict(type='mmdet.LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'pad_param',
                        'flip',
                        'flip_direction',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_ann_file = 'annotations/instances_val2017.json'
val_batch_size_per_gpu = 1
val_cfg = dict(type='ValLoop')
val_data_prefix = 'val2017/'
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        class_text_path='data/texts/lvis_v1_class_texts.json',
        dataset=dict(
            ann_file='lvis/lvis_v1_minival_inserted_image_name.json',
            batch_shapes_cfg=None,
            data_prefix=dict(img=''),
            data_root='data/coco/',
            test_mode=True,
            type='YOLOv5LVISV1Dataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='MultiModalDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/coco/lvis/lvis_v1_minival_inserted_image_name.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.LVISMetric')
val_interval_stage2 = 1
val_num_workers = 2
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
weight_decay = 0.025
widen_factor = 1.25
work_dir = '/home/jean/Amine/OpenYolo3D/pretrained/configs/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival'
