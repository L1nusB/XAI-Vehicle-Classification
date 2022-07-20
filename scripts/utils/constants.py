from mmseg.models.Segmentors import CascadeEncoderDecoder

PALETTES={
    'Comp_Original_Ocrnet_Carparts_Noflip':
    [[102, 179,  92],
       [ 14, 106,  71],
       [188,  20, 102],
       [121, 210, 214],
       [ 74, 202,  87],
       [116,  99, 103],
       [151, 130, 149],
       [ 52,   1,  87],
       [235, 157,  37],
       [129, 191, 187],
       [ 20, 160, 203],
       [ 57,  21, 252],
       [235,  88,  48],
       [218,  58, 254],
       [169, 219, 187],
       [207,  14, 189],
       [189, 174, 189],
       [ 50, 107,  54]]
}

PIPELINES={
    'Resize224':
    {
        'pre':
        [
            {'type': 'LoadImageFromFile'},
            {'type': 'ResizeCls', 'size': (224,224)},
            {'type': 'Normalize',
                'mean': [123.675, 116.28, 103.53],
                'std': [58.395, 57.12, 57.375],
                'to_rgb': True},
            {'type': 'ImageToTensor', 'keys': ['img']},
            {'type': 'Collect', 'keys': ['img']}
        ],
        'post':
        [
            {'type': 'LoadImageFromFile'},
            {'type': 'Resize', 'size': (224,224)},
            {'type': 'Normalize',
                'mean': [123.675, 116.28, 103.53],
                'std': [58.395, 57.12, 57.375],
                'to_rgb': True},
            {'type': 'ImageToTensor', 'keys': ['img']},
            {'type': 'Collect', 'keys': ['img']}
        ],
    },
    'Resize224CenterCrop':
    {
        'pre':
        [
            {'type': 'LoadImageFromFile'},
            {'type': 'ResizeCls', 'size': (256, -1)},
            {'type': 'CenterCropCls', 'crop_size': 224},
            {'type': 'Normalize',
                'mean': [123.675, 116.28, 103.53],
                'std': [58.395, 57.12, 57.375],
                'to_rgb': True},
            {'type': 'ImageToTensor', 'keys': ['img']},
            {'type': 'Collect', 'keys': ['img']}
        ],
        'post':
        [
            {'type': 'LoadImageFromFile'},
            {'type': 'Resize', 'size': (256, -1)},
            {'type': 'CenterCrop', 'crop_size': 224},
            {'type': 'Normalize',
                'mean': [123.675, 116.28, 103.53],
                'std': [58.395, 57.12, 57.375],
                'to_rgb': True},
            {'type': 'ImageToTensor', 'keys': ['img']},
            {'type': 'Collect', 'keys': ['img']}
        ],
    }
}

PIPELINEMAPS={
    'cls':
    {
        'Resize':'ResizeCls',
        'CenterCrop':'CenterCropCls',
        'RandomCrop':'RandomCropCls'
    },
    'none':{},
}

PIPELINETRANSFORMS=[
    'Resize','CenterCrop','RandomCrop','ResizeCls','CenterCropCls','RandomCropCls'
]

TYPES=[
    'masks',
    'images',
]

MODELWRAPPERS={
    'CascadeEncoderDecoder':CascadeEncoderDecoder
    }