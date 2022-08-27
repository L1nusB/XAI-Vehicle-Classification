from mmseg.models.segmentors import CascadeEncoderDecoder

from mmcls.datasets.compcars import CompCars
from mmcls.datasets.compcarsWeb import CompCarsWeb

RESULTS_PATH='./results'
RESULTS_PATH_ANN='./results/annfiles'
RESULTS_PATH_DATACLASS='./results/dataClasses'

DATASETSDATAPREFIX={
    'CompCarsOriginal':'CompCars_sv_original',
    'CompCarsColor':'CompCars_sv_color',
    'CompCarsWeb':'CompCars_web_original',
    'CarPartsNoFlip':'carparts_noflip',
    'CarPartsNoFrontBack':'carparts_nofrontback',
    'CarPartsNoLeftRight':'carparts_noleftright',
}


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

DATASETWRAPPERS = {
    'CompCars': CompCars,
    'CompCarsWeb': CompCarsWeb
}
DATASETWRAPPERSBLURRED = {
    'CompCars': 'BlurredCompCars',
    'CompCarsWeb': 'BlurredCompCarsWeb'
}

EXCELCOLNAMESSTANDARD = {
    'summarizedSegmentedCAMActivations':'RawActivations',
    'summarizedPercSegmentedCAMActivations':'PercActivations',
    'totalActivation':'totalActivation'
}

EXCELCOLNAMESPROPORTIONAL = {
    'summarizedPercSegmentedCAMActivations':'PercActivations',
    'summarizedPercSegmentAreas':'PercSegmentAreas'
}

EXCELCOLNAMESNORMALIZED = {
    'summarizedPercSegmentedCAMActivations':'PercActivations',
    'summarizedPercSegmentAreas':'PercSegmentAreas',
    'relImportance':'RelativeCAMImportance',
    'rescaledSummarizedPercActivions':'PercActivationsRescaled'
}

EXCELCOLNAMESMEANSTDTOTAL = {
    'summarizedPercSegmentCAMActivations':'PercActivations',
    'stdPercSegmentCAMActivations':'PercActivationsStd',
    'summarizedSegmentCAMActivations':'RawActivations',
    'stdSegmentCAMActivations':'RawActivationsStd',
    'summarizedPercSegmentAreas':'PercSegmentAreas',
    'stdPercSegmentAreas':'PercSegmentAreasStd',
    'summarizedSegmentAreas':'RawSegmentAreas',
    'stdSegmentAreas':'RawSegmentAreasStd',
    'totalMean':'totalMean',
    'totalStd':'totalStd'
}

EXCELCOLNAMESMISSCLASSIFIED = {
    'summarizedPercCAMActivationsOriginal':'PercActivationsOriginal',
    'summarizedPercCAMActivationsCorrect':'PercActivationsCorrect',
    'summarizedPercCAMActivationsIncorrect':'PercActivationsIncorrect',
    'summarizedPercCAMActivationsCorrected':'PercActivationsCorrected',
    'summarizedPercCAMActivationsFixed':'PercActivationsFixed'
}