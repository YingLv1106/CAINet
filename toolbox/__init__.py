from .metrics import averageMeter, runningScore
from .log import get_logger
from .loss import MscCrossEntropyLoss
from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB
from .ranger.ranger import Ranger
from .ranger.ranger913A import RangerVA
from .ranger.rangerqh import RangerQH


def get_dataset(cfg):
    assert cfg['dataset'] in [ 'irseg', 'pst900']


    if cfg['dataset'] == 'irseg':
        from .datasets.irseg import IRSeg
        return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='trainval'), IRSeg(cfg, mode='trainvaltest'), IRSeg(cfg, mode='test')
    if cfg['dataset'] == 'pst900':
        from .datasets.pst900 import PST900
        return PST900(cfg, mode='train'),  PST900(cfg, mode='trainval'), PST900(cfg, mode='test')


def get_model(cfg):

    if cfg['model_name'] == 'cainet':
        from toolbox.models.cainet import mobilenetGloRe3_CRRM_dule_arm_bou_att
        return mobilenetGloRe3_CRRM_dule_arm_bou_att(n_classes=cfg['n_classes'])