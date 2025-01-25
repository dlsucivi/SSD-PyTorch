from models.ssd import build_SSD
from models.sfdet_vgg import build_SFDetVGG
from models.sfdet_resnet import build_SFDetResNet
from models.sfdet_resnext import build_SFDetResNeXt
from models.sfdet_densenet import build_SFDetDenseNet


def get_model(config,
              anchors,
              output_txt):
    """
    returns the model
    """

    model = None

    if config['model'] == 'SFDet-VGG':
        model = build_SFDetVGG(mode=config['mode'],
                               new_size=config['new_size'],
                               anchors=anchors,
                               class_count=config['class_count'],
                               model_save_path=config['model_save_path'],
                               pretrained_model=config['coco_weights'],
                               output_txt=output_txt)

    elif config['model'] == 'SFDet-ResNet':
        model = build_SFDetResNet(mode=config['mode'],
                                  new_size=config['new_size'],
                                  resnet_model=config['resnet_model'],
                                  anchors=anchors,
                                  class_count=config['class_count'],
                                  model_save_path=config['model_save_path'],
                                  pretrained_model=config['coco_weights'],
                                  output_txt=output_txt)

    elif config['model'] == 'SFDet-DenseNet':
        model = build_SFDetDenseNet(mode=config['mode'],
                                    new_size=config['new_size'],
                                    densenet_model=config['densenet_model'],
                                    anchors=anchors,
                                    class_count=config['class_count'])

    elif config['model'] == 'SFDet-ResNeXt':
        model = build_SFDetResNeXt(mode=config['mode'],
                                   new_size=config['new_size'],
                                   resnext_model=config['resnext_model'],
                                   anchors=anchors,
                                   class_count=config['class_count'])

    elif config['model'] == 'SSD':
        model = build_SSD(mode=config['mode'],
                          new_size=config['new_size'],
                          anchors=anchors,
                          class_count=config['class_count'])

    return model
