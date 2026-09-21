"""
Dataset loaders for TTA experiments
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from typing import Optional, Sequence, Tuple
from PIL import Image


IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def resolve_cifar_input_size(preprocess: str = 'cifar_default', input_size: Optional[int] = None) -> int:
    """Resolve the effective CIFAR input size from optional config overrides."""
    if input_size is not None:
        return int(input_size)
    return 224 if (preprocess or 'cifar_default').lower() == 'imagenet_224' else 32


def build_cifar_transform(preprocess: str = 'cifar_default', input_size: Optional[int] = None):
    """Build CIFAR preprocessing for native 32x32 or ImageNet-style 224 inputs."""
    preprocess = (preprocess or 'cifar_default').lower()
    size = resolve_cifar_input_size(preprocess, input_size)

    if preprocess == 'cifar_default':
        return transforms.Compose([transforms.ToTensor()])

    if preprocess == 'imagenet_224':
        return transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            IMAGENET_NORMALIZE,
        ])

    raise ValueError(
        f"Unknown CIFAR preprocess: {preprocess}. Expected 'cifar_default' or 'imagenet_224'."
    )


def _get_label_shift_dataset_spec(dataset_name: str) -> Tuple[int, int, int, int]:
    name = dataset_name.lower()
    if name == 'imagenetc':
        return 1000, 50000, 50, 100000
    if name == 'cifar10c':
        return 10, 10000, 1000, 100000
    if name == 'cifar100c':
        return 100, 10000, 100, 100000
    raise ValueError(f"Label shift is not supported for dataset: {dataset_name}")


def format_label_shift_ratio(imbalance_ratio: float) -> str:
    ratio = float(imbalance_ratio)
    if ratio.is_integer():
        return str(int(ratio))
    return f"{ratio:g}"


def build_label_shift_indices(
    dataset_name: str,
    imbalance_ratio: float,
    seed: Optional[int],
    total_samples: Optional[int] = None,
    shuffle_class_order: bool = True,
) -> np.ndarray:
    """Reproduce SAR/DeYO online imbalanced label-shift indices.

    The official protocol first samples a 100000-step label stream from a
    per-class dominant distribution, then maps each sampled label back to one of
    the original test examples from the matching class. The class-order shuffle
    depends on the experiment seed, while the actual label/index sampling uses a
    fixed NumPy seed (2022), matching the official scripts.
    """
    num_classes, _, num_each_class, default_total_samples = _get_label_shift_dataset_spec(dataset_name)
    total_samples = default_total_samples if total_samples is None else int(total_samples)
    if total_samples % num_classes != 0:
        raise ValueError(
            f"total_samples ({total_samples}) must be divisible by num_classes ({num_classes})"
        )

    imbalance_ratio = float(imbalance_ratio)
    if imbalance_ratio < 1.0:
        raise ValueError(f"imbalance_ratio must be >= 1, got {imbalance_ratio}")

    minor_class_prob = 1.0 / (imbalance_ratio + num_classes - 1)
    major_class_prob = minor_class_prob * imbalance_ratio
    q_for_all_classes = np.full((num_classes, num_classes), minor_class_prob, dtype=np.float64)
    np.fill_diagonal(q_for_all_classes, major_class_prob)

    class_order = list(range(num_classes))
    if shuffle_class_order:
        import random

        random.Random(2021 if seed is None else int(seed)).shuffle(class_order)
        q_for_all_classes = q_for_all_classes[class_order, :]

    num_for_repeat_each_q = total_samples // num_classes
    q_all = np.concatenate(
        [np.expand_dims(q_for_all_classes[i], axis=0) for i in range(num_classes) for _ in range(num_for_repeat_each_q)],
        axis=0,
    )

    rng = np.random.RandomState(2022)
    ys = np.squeeze(np.asarray([rng.choice(num_classes, 1, p=q) for q in q_all]))

    generated_indices = np.zeros(total_samples, dtype=np.int64)
    for class_idx in range(num_classes):
        mask = ys == class_idx
        num_i = int(mask.sum())
        if num_i == 0:
            continue
        sampled_indices = rng.randint(0, num_each_class, size=num_i)
        generated_indices[mask] = class_idx * num_each_class + sampled_indices

    return generated_indices


def get_label_shift_indices(
    dataset_name: str,
    imbalance_ratio: float,
    seed: Optional[int],
    cache_dir: Optional[str] = None,
    total_samples: Optional[int] = None,
) -> np.ndarray:
    """Load cached label-shift indices or build them with the official protocol."""
    _, _, _, default_total_samples = _get_label_shift_dataset_spec(dataset_name)
    total_samples = default_total_samples if total_samples is None else int(total_samples)

    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        ratio_tag = format_label_shift_ratio(imbalance_ratio)
        seed_tag = 2021 if seed is None else int(seed)
        cache_path = os.path.join(
            cache_dir,
            f"{dataset_name.lower()}_seed{seed_tag}_total_{total_samples}_ir_{ratio_tag}_class_order_shuffle_yes.npy",
        )
        if os.path.exists(cache_path):
            return np.load(cache_path)

    indices = build_label_shift_indices(
        dataset_name=dataset_name,
        imbalance_ratio=imbalance_ratio,
        seed=seed,
        total_samples=total_samples,
        shuffle_class_order=True,
    )

    if cache_path is not None:
        np.save(cache_path, indices)

    return indices


class CIFAR_C_Dataset(Dataset):
    """
    Dataset for CIFAR-10-C or CIFAR-100-C

    Args:
        data_root: Path to CIFAR-C directory
        corruption: Corruption type (e.g., 'brightness', 'gaussian_noise')
        severity: Corruption severity (1-5)
        transform: Image transforms
    """

    def __init__(
        self,
        data_root: str,
        corruption: str,
        severity: int = 5,
        transform=None
    ):
        self.transform = transform or build_cifar_transform()

        # Load data
        images_path = os.path.join(data_root, f'{corruption}.npy')
        labels_path = os.path.join(data_root, 'labels.npy')

        images = np.load(images_path)
        labels = np.load(labels_path)

        # Select severity level
        # Each .npy has 50000 images = 10000 per severity
        n_per_severity = 10000
        start_idx = (severity - 1) * n_per_severity
        end_idx = severity * n_per_severity

        self.images = images[start_idx:end_idx]
        self.labels = labels[start_idx:end_idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = int(self.labels[idx])

        # Convert to PIL for transforms
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label


# ImageNet synset to class index mapping
# This is the standard ILSVRC2012 ordering
IMAGENET_SYNSET_TO_IDX = None  # Will be lazily loaded

def get_imagenet_synset_mapping():
    """
    Get ImageNet synset to class index mapping.
    Uses the standard ILSVRC2012 ordering where synsets are sorted alphabetically.
    """
    global IMAGENET_SYNSET_TO_IDX
    if IMAGENET_SYNSET_TO_IDX is None:
        # Standard ImageNet synsets in alphabetical order (same as torchvision)
        # This matches the order used by PyTorch's pretrained models
        synsets = [
            'n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475',
            'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878',
            'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544',
            'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084',
            'n01601694', 'n01608432', 'n01614925', 'n01616318', 'n01622779',
            'n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777',
            'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541',
            'n01667114', 'n01667778', 'n01669191', 'n01675722', 'n01677366',
            'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811',
            'n01692333', 'n01693334', 'n01694178', 'n01695060', 'n01697457',
            'n01698640', 'n01704323', 'n01728572', 'n01728920', 'n01729322',
            'n01729977', 'n01734418', 'n01735189', 'n01737021', 'n01739381',
            'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939',
            'n01751748', 'n01753488', 'n01755581', 'n01756291', 'n01768244',
            'n01770081', 'n01770393', 'n01773157', 'n01773549', 'n01773797',
            'n01774384', 'n01774750', 'n01775062', 'n01776313', 'n01784675',
            'n01795545', 'n01796340', 'n01797886', 'n01798484', 'n01806143',
            'n01806567', 'n01807496', 'n01817953', 'n01818515', 'n01819313',
            'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805',
            'n01843065', 'n01843383', 'n01847000', 'n01855032', 'n01855672',
            'n01860187', 'n01871265', 'n01872401', 'n01873310', 'n01877812',
            'n01882714', 'n01883070', 'n01910747', 'n01914609', 'n01917289',
            'n01924916', 'n01930112', 'n01943899', 'n01944390', 'n01945685',
            'n01950731', 'n01955084', 'n01968897', 'n01978287', 'n01978455',
            'n01980166', 'n01981276', 'n01983481', 'n01984695', 'n01985128',
            'n01986214', 'n01990800', 'n02002556', 'n02002724', 'n02006656',
            'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02012849',
            'n02013706', 'n02017213', 'n02018207', 'n02018795', 'n02025239',
            'n02027492', 'n02028035', 'n02033041', 'n02037110', 'n02051845',
            'n02056570', 'n02058221', 'n02066245', 'n02071294', 'n02074367',
            'n02077923', 'n02085620', 'n02085782', 'n02085936', 'n02086079',
            'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394',
            'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02088632',
            'n02089078', 'n02089867', 'n02089973', 'n02090379', 'n02090622',
            'n02090721', 'n02091032', 'n02091134', 'n02091244', 'n02091467',
            'n02091635', 'n02091831', 'n02092002', 'n02092339', 'n02093256',
            'n02093428', 'n02093647', 'n02093754', 'n02093859', 'n02093991',
            'n02094114', 'n02094258', 'n02094433', 'n02095314', 'n02095570',
            'n02095889', 'n02096051', 'n02096177', 'n02096294', 'n02096437',
            'n02096585', 'n02097047', 'n02097130', 'n02097209', 'n02097298',
            'n02097474', 'n02097658', 'n02098105', 'n02098286', 'n02098413',
            'n02099267', 'n02099429', 'n02099601', 'n02099712', 'n02099849',
            'n02100236', 'n02100583', 'n02100735', 'n02100877', 'n02101006',
            'n02101388', 'n02101556', 'n02102040', 'n02102177', 'n02102318',
            'n02102480', 'n02102973', 'n02104029', 'n02104365', 'n02105056',
            'n02105162', 'n02105251', 'n02105412', 'n02105505', 'n02105641',
            'n02105855', 'n02106030', 'n02106166', 'n02106382', 'n02106550',
            'n02106662', 'n02107142', 'n02107312', 'n02107574', 'n02107683',
            'n02107908', 'n02108000', 'n02108089', 'n02108422', 'n02108551',
            'n02108915', 'n02109047', 'n02109525', 'n02109961', 'n02110063',
            'n02110185', 'n02110341', 'n02110627', 'n02110806', 'n02110958',
            'n02111129', 'n02111277', 'n02111500', 'n02111889', 'n02112018',
            'n02112137', 'n02112350', 'n02112706', 'n02113023', 'n02113186',
            'n02113624', 'n02113712', 'n02113799', 'n02113978', 'n02114367',
            'n02114548', 'n02114712', 'n02114855', 'n02115641', 'n02115913',
            'n02116738', 'n02117135', 'n02119022', 'n02119789', 'n02120079',
            'n02120505', 'n02123045', 'n02123159', 'n02123394', 'n02123597',
            'n02124075', 'n02125311', 'n02127052', 'n02128385', 'n02128757',
            'n02128925', 'n02129165', 'n02129604', 'n02130308', 'n02132136',
            'n02133161', 'n02134084', 'n02134418', 'n02137549', 'n02138441',
            'n02165105', 'n02165456', 'n02167151', 'n02168699', 'n02169497',
            'n02172182', 'n02174001', 'n02177972', 'n02190166', 'n02206856',
            'n02219486', 'n02226429', 'n02229544', 'n02231487', 'n02233338',
            'n02236044', 'n02256656', 'n02259212', 'n02264363', 'n02268443',
            'n02268853', 'n02276258', 'n02277742', 'n02279972', 'n02280649',
            'n02281406', 'n02281787', 'n02317335', 'n02319095', 'n02321529',
            'n02325366', 'n02326432', 'n02328150', 'n02342885', 'n02346627',
            'n02356798', 'n02361337', 'n02363005', 'n02364673', 'n02389026',
            'n02391049', 'n02395406', 'n02396427', 'n02397096', 'n02398521',
            'n02403003', 'n02408429', 'n02410509', 'n02412080', 'n02415577',
            'n02417914', 'n02422106', 'n02422699', 'n02423022', 'n02437312',
            'n02437616', 'n02441942', 'n02442845', 'n02443114', 'n02443484',
            'n02444819', 'n02445715', 'n02447366', 'n02454379', 'n02457408',
            'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02483708',
            'n02484975', 'n02486261', 'n02486410', 'n02487347', 'n02488291',
            'n02488702', 'n02489166', 'n02490219', 'n02492035', 'n02492660',
            'n02493509', 'n02493793', 'n02494079', 'n02497673', 'n02500267',
            'n02504013', 'n02504458', 'n02509815', 'n02510455', 'n02514041',
            'n02526121', 'n02536864', 'n02606052', 'n02607072', 'n02640242',
            'n02641379', 'n02643566', 'n02655020', 'n02666196', 'n02667093',
            'n02669723', 'n02672831', 'n02676566', 'n02687172', 'n02690373',
            'n02692877', 'n02699494', 'n02701002', 'n02704792', 'n02708093',
            'n02727426', 'n02730930', 'n02747177', 'n02749479', 'n02769748',
            'n02776631', 'n02777292', 'n02782093', 'n02783161', 'n02786058',
            'n02787622', 'n02788148', 'n02790996', 'n02791124', 'n02791270',
            'n02793495', 'n02794156', 'n02795169', 'n02797295', 'n02799071',
            'n02802426', 'n02804414', 'n02804610', 'n02807133', 'n02808304',
            'n02808440', 'n02814533', 'n02814860', 'n02815834', 'n02817516',
            'n02823428', 'n02823750', 'n02825657', 'n02834397', 'n02835271',
            'n02837789', 'n02840245', 'n02841315', 'n02843684', 'n02859443',
            'n02860847', 'n02865351', 'n02869837', 'n02870880', 'n02871525',
            'n02877765', 'n02879718', 'n02883205', 'n02892201', 'n02892767',
            'n02894605', 'n02895154', 'n02906734', 'n02909870', 'n02910353',
            'n02916936', 'n02917067', 'n02927161', 'n02930766', 'n02939185',
            'n02948072', 'n02950826', 'n02951358', 'n02951585', 'n02963159',
            'n02965783', 'n02966193', 'n02966687', 'n02971356', 'n02974003',
            'n02977058', 'n02978881', 'n02979186', 'n02980441', 'n02981792',
            'n02988304', 'n02992211', 'n02992529', 'n02999410', 'n03000134',
            'n03000247', 'n03000684', 'n03014705', 'n03016953', 'n03017168',
            'n03018349', 'n03026506', 'n03028079', 'n03032252', 'n03041632',
            'n03042490', 'n03045698', 'n03047690', 'n03062245', 'n03063599',
            'n03063689', 'n03065424', 'n03075370', 'n03085013', 'n03089624',
            'n03095699', 'n03100240', 'n03109150', 'n03110669', 'n03124043',
            'n03124170', 'n03125729', 'n03126707', 'n03127747', 'n03127925',
            'n03131574', 'n03133878', 'n03134739', 'n03141823', 'n03146219',
            'n03160309', 'n03179701', 'n03180011', 'n03187595', 'n03188531',
            'n03196217', 'n03197337', 'n03201208', 'n03207743', 'n03207941',
            'n03208938', 'n03216828', 'n03218198', 'n03220513', 'n03223299',
            'n03240683', 'n03249569', 'n03250847', 'n03255030', 'n03259280',
            'n03271574', 'n03272010', 'n03272562', 'n03290653', 'n03291819',
            'n03297495', 'n03314780', 'n03325584', 'n03337140', 'n03344393',
            'n03345487', 'n03347037', 'n03355925', 'n03372029', 'n03376595',
            'n03379051', 'n03384352', 'n03388043', 'n03388183', 'n03388549',
            'n03393912', 'n03394916', 'n03400231', 'n03404251', 'n03417042',
            'n03424325', 'n03425413', 'n03443371', 'n03444034', 'n03445777',
            'n03445924', 'n03447447', 'n03447721', 'n03450230', 'n03452741',
            'n03457902', 'n03459775', 'n03461385', 'n03467068', 'n03476684',
            'n03476991', 'n03478589', 'n03481172', 'n03482405', 'n03483316',
            'n03485407', 'n03485794', 'n03492542', 'n03494278', 'n03495258',
            'n03496892', 'n03498962', 'n03527444', 'n03529860', 'n03530642',
            'n03532672', 'n03534580', 'n03535780', 'n03538406', 'n03544143',
            'n03584254', 'n03584829', 'n03590841', 'n03594734', 'n03594945',
            'n03595614', 'n03598930', 'n03599486', 'n03602883', 'n03617480',
            'n03623198', 'n03627232', 'n03630383', 'n03633091', 'n03637318',
            'n03642806', 'n03649909', 'n03657121', 'n03658185', 'n03661043',
            'n03662601', 'n03666591', 'n03670208', 'n03673027', 'n03676483',
            'n03680355', 'n03690938', 'n03691459', 'n03692522', 'n03697007',
            'n03706229', 'n03709823', 'n03710193', 'n03710637', 'n03710721',
            'n03717622', 'n03720891', 'n03721384', 'n03724870', 'n03729826',
            'n03733131', 'n03733281', 'n03733805', 'n03742115', 'n03743016',
            'n03759954', 'n03761084', 'n03763968', 'n03764736', 'n03769881',
            'n03770439', 'n03770679', 'n03773504', 'n03775071', 'n03775546',
            'n03776460', 'n03777568', 'n03777754', 'n03781244', 'n03782006',
            'n03785016', 'n03786901', 'n03787032', 'n03788195', 'n03788365',
            'n03791053', 'n03792782', 'n03792972', 'n03793489', 'n03794056',
            'n03796401', 'n03803284', 'n03804744', 'n03814639', 'n03814906',
            'n03825788', 'n03832673', 'n03837869', 'n03838899', 'n03840681',
            'n03841143', 'n03843555', 'n03854065', 'n03857828', 'n03866082',
            'n03868242', 'n03868863', 'n03871628', 'n03873416', 'n03874293',
            'n03874599', 'n03876231', 'n03877472', 'n03877845', 'n03884397',
            'n03887697', 'n03888257', 'n03888605', 'n03891251', 'n03891332',
            'n03895866', 'n03899768', 'n03902125', 'n03903868', 'n03908618',
            'n03908714', 'n03916031', 'n03920288', 'n03924679', 'n03929660',
            'n03929855', 'n03930313', 'n03930630', 'n03933933', 'n03935335',
            'n03937543', 'n03938244', 'n03942813', 'n03944341', 'n03947888',
            'n03950228', 'n03954731', 'n03956157', 'n03958227', 'n03961711',
            'n03967562', 'n03970156', 'n03976467', 'n03976657', 'n03977966',
            'n03980874', 'n03982430', 'n03983396', 'n03991062', 'n03992509',
            'n03995372', 'n03998194', 'n04004767', 'n04005630', 'n04008634',
            'n04009552', 'n04019541', 'n04023962', 'n04026417', 'n04033901',
            'n04033995', 'n04037443', 'n04039381', 'n04040759', 'n04041544',
            'n04044716', 'n04049303', 'n04065272', 'n04067472', 'n04069434',
            'n04070727', 'n04074963', 'n04081281', 'n04086273', 'n04090263',
            'n04099969', 'n04111531', 'n04116512', 'n04118538', 'n04118776',
            'n04120489', 'n04125021', 'n04127249', 'n04131690', 'n04133789',
            'n04136333', 'n04141076', 'n04141327', 'n04141975', 'n04146614',
            'n04147183', 'n04149813', 'n04152593', 'n04153751', 'n04154565',
            'n04162706', 'n04179913', 'n04192698', 'n04200800', 'n04201297',
            'n04204238', 'n04204347', 'n04208210', 'n04209133', 'n04209239',
            'n04228054', 'n04229816', 'n04235860', 'n04238763', 'n04239074',
            'n04243546', 'n04251144', 'n04252077', 'n04252225', 'n04254120',
            'n04254680', 'n04254777', 'n04258138', 'n04259630', 'n04263257',
            'n04264628', 'n04265275', 'n04266014', 'n04270147', 'n04273569',
            'n04275548', 'n04277352', 'n04285008', 'n04286575', 'n04296562',
            'n04310018', 'n04311004', 'n04311174', 'n04317175', 'n04325704',
            'n04326547', 'n04328186', 'n04330267', 'n04332243', 'n04335435',
            'n04336792', 'n04344873', 'n04346328', 'n04347754', 'n04350905',
            'n04355338', 'n04355933', 'n04356056', 'n04357314', 'n04366367',
            'n04367480', 'n04370456', 'n04371430', 'n04371774', 'n04372370',
            'n04376876', 'n04380533', 'n04389033', 'n04392985', 'n04398044',
            'n04399382', 'n04404412', 'n04409515', 'n04417672', 'n04418357',
            'n04423845', 'n04428191', 'n04429376', 'n04435653', 'n04442312',
            'n04443257', 'n04447861', 'n04456115', 'n04458633', 'n04461696',
            'n04462240', 'n04465501', 'n04467665', 'n04476259', 'n04479046',
            'n04482393', 'n04483307', 'n04485082', 'n04486054', 'n04487081',
            'n04487394', 'n04493381', 'n04501370', 'n04505470', 'n04507155',
            'n04509417', 'n04515003', 'n04517823', 'n04522168', 'n04523525',
            'n04525038', 'n04525305', 'n04532106', 'n04532670', 'n04536866',
            'n04540053', 'n04542943', 'n04548280', 'n04548362', 'n04550184',
            'n04552348', 'n04553703', 'n04554684', 'n04557648', 'n04560804',
            'n04562935', 'n04579145', 'n04579432', 'n04584207', 'n04589890',
            'n04590129', 'n04591157', 'n04591713', 'n04592741', 'n04596742',
            'n04597913', 'n04599235', 'n04604644', 'n04606251', 'n04612504',
            'n04613696', 'n06359193', 'n06596364', 'n06785654', 'n06794110',
            'n06874185', 'n07248320', 'n07565083', 'n07579787', 'n07583066',
            'n07584110', 'n07590611', 'n07613480', 'n07614500', 'n07615774',
            'n07684084', 'n07693725', 'n07695742', 'n07697313', 'n07697537',
            'n07711569', 'n07714571', 'n07714990', 'n07715103', 'n07716358',
            'n07716906', 'n07717410', 'n07717556', 'n07718472', 'n07718747',
            'n07720875', 'n07730033', 'n07734744', 'n07742313', 'n07745940',
            'n07747607', 'n07749582', 'n07753113', 'n07753275', 'n07753592',
            'n07754684', 'n07760859', 'n07768694', 'n07802026', 'n07831146',
            'n07836838', 'n07860988', 'n07871810', 'n07873807', 'n07875152',
            'n07880968', 'n07892512', 'n07920052', 'n07930864', 'n07932039',
            'n09193705', 'n09229709', 'n09246464', 'n09256479', 'n09288635',
            'n09332890', 'n09399592', 'n09421951', 'n09428293', 'n09468604',
            'n09472597', 'n09835506', 'n10148035', 'n10565667', 'n11879895',
            'n11939491', 'n12057211', 'n12144580', 'n12267677', 'n12620546',
            'n12768682', 'n12985857', 'n12998815', 'n13037406', 'n13040303',
            'n13044778', 'n13052670', 'n13054560', 'n13133613', 'n15075141'
        ]
        IMAGENET_SYNSET_TO_IDX = {synset: idx for idx, synset in enumerate(synsets)}
    return IMAGENET_SYNSET_TO_IDX


class ImageNetC_Dataset(Dataset):
    """
    Dataset for ImageNet-C

    Directory structure expected:
    imagenet-c/
        brightness/
            1/, 2/, 3/, 4/, 5/
        contrast/
            ...
    """

    def __init__(
        self,
        data_root: str,
        corruption: str,
        severity: int = 5,
        transform=None
    ):
        self.transform = transform or transforms.Compose([
            # NOTE: ImageNet-C images are already 224x224, do NOT Resize(256)!
            # Resize(256) + CenterCrop(224) introduces interpolation artifacts and
            # discards edge information. Only CenterCrop(224) is needed (following
            # official SAR/EATA/Tent code).
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Path to corruption directory
        corruption_dir = os.path.join(data_root, corruption, str(severity))

        self.samples = []

        # Walk through class directories
        for class_name in sorted(os.listdir(corruption_dir)):
            class_dir = os.path.join(corruption_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            # Get class index from ILSVRC2012 format
            class_idx = self._get_class_idx(class_name)

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

    def _get_class_idx(self, class_name: str) -> int:
        """
        Convert synset name to class index
        For ImageNet-C, class folders are typically numbered 0-999
        or named with synsets (e.g., n01440764)
        """
        # If it's a number, use directly
        if class_name.isdigit():
            return int(class_name)

        # Use synset to index mapping
        synset_mapping = get_imagenet_synset_mapping()
        if class_name in synset_mapping:
            return synset_mapping[class_name]

        # Fallback: try to extract synset from folder name
        raise ValueError(f"Unknown ImageNet class: {class_name}. "
                        f"Expected synset like 'n01440764' or numeric index.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_corruption_loader(
    dataset_name: str,
    corruption: str,
    severity: int,
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = False,
    seed: int = None,
    persistent_workers: bool = False,
    preprocess: str = 'cifar_default',
    input_size: Optional[int] = None,
    subset_indices: Optional[Sequence[int]] = None,
) -> DataLoader:
    """
    Get dataloader for a specific corruption

    Args:
        dataset_name: 'cifar10c', 'cifar100c', or 'imagenetc'
        corruption: Corruption type
        severity: 1-5
        data_root: Path to data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data (default: False for TTA)
        seed: Random seed for reproducible shuffling (only used if shuffle=True)

    Returns:
        DataLoader for the corruption
    """
    if 'cifar' in dataset_name.lower():
        dataset = CIFAR_C_Dataset(
            data_root=data_root,
            corruption=corruption,
            severity=severity,
            transform=build_cifar_transform(preprocess=preprocess, input_size=input_size),
        )
    elif 'imagenet' in dataset_name.lower():
        dataset = ImageNetC_Dataset(
            data_root=data_root,
            corruption=corruption,
            severity=severity
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if subset_indices is not None:
        dataset = Subset(dataset, [int(index) for index in subset_indices])

    # For reproducible shuffling when shuffle=True
    generator = None
    if shuffle and seed is not None:
        import torch
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=bool(persistent_workers) if num_workers > 0 else False,
        pin_memory=True,
        generator=generator if shuffle else None
    )


def get_dataloader(
    dataset_name: str,
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    train: bool = False
) -> DataLoader:
    """
    Get dataloader for clean dataset (for source accuracy evaluation)
    """
    # This is a simplified version
    # In practice, use torchvision.datasets or your data pipeline
    raise NotImplementedError("Use get_corruption_loader for TTA experiments")


def get_cifar_train_loader(
    dataset_name,
    data_root='./data',
    batch_size=64,
    num_workers=4,
    preprocess: str = 'cifar_default',
    input_size: Optional[int] = None,
):
    """
    Get CIFAR-10/100 training dataloader for Fisher computation.

    Uses the same configurable preprocessing as CIFAR-C evaluation so Fisher
    estimation matches the active runtime path.

    Args:
        dataset_name: 'cifar10c' or 'cifar100c'
        data_root: Root directory containing CIFAR data
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        DataLoader for CIFAR training set
    """
    from torchvision import datasets

    transform = build_cifar_transform(preprocess=preprocess, input_size=input_size)

    if 'cifar100' in dataset_name.lower():
        dataset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform)
    else:
        dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )


def get_imagenet_val_loader(val_root, batch_size=64, num_workers=4):
    """
    Get ImageNet validation dataloader for Fisher computation

    Args:
        val_root: Path to ImageNet validation directory (e.g., './data/imagenet/val')
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        DataLoader for ImageNet validation set
    """
    from torchvision import transforms, datasets

    # Standard ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        IMAGENET_NORMALIZE,
    ])

    dataset = datasets.ImageFolder(val_root, transform=transform)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )


# =============================================================================
# Natural Distribution Shift Datasets
# ImageNet-R, ImageNet-A, ImageNet-V2, ImageNet-Sketch
# =============================================================================

# Standard ImageNet transform (for natural shift datasets)
IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    IMAGENET_NORMALIZE,
])


class ImageNetSubsetDataset(Dataset):
    """
    Dataset for ImageNet subset variants (ImageNet-R, ImageNet-A).
    These datasets contain only ~200 of the 1000 ImageNet classes.

    Maps synset folder names to original ImageNet class indices.
    Only outputs predictions for the subset classes present in the dataset.

    Directory structure:
        data_root/
            n01443537/
                img1.jpg
                img2.jpg
            n01484850/
                ...
    """

    def __init__(self, data_root: str, transform=None):
        self.transform = transform or IMAGENET_TRANSFORM
        self.data_root = data_root

        # Build class mapping: synset -> ImageNet class index
        synset_mapping = get_imagenet_synset_mapping()

        self.samples = []
        self.class_indices = set()  # Track which ImageNet classes are present

        for entry in sorted(os.listdir(data_root)):
            class_dir = os.path.join(data_root, entry)
            if not os.path.isdir(class_dir):
                continue
            if entry not in synset_mapping:
                continue

            class_idx = synset_mapping[entry]
            self.class_indices.add(class_idx)

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png', '.JPEG')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

        self.class_indices = sorted(self.class_indices)
        self.num_classes = len(self.class_indices)
        print(f"  Loaded {len(self.samples)} images from {self.num_classes} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class ImageNetV2Dataset(Dataset):
    """
    Dataset for ImageNet-V2 (MatchedFrequency variant).

    Uses numeric folder names (0-999) matching ImageNet class indices.

    Directory structure:
        data_root/
            0/
                img1.jpeg
            1/
                img2.jpeg
            ...
            999/
    """

    def __init__(self, data_root: str, transform=None):
        self.transform = transform or IMAGENET_TRANSFORM
        self.data_root = data_root

        self.samples = []

        for entry in sorted(os.listdir(data_root), key=lambda x: int(x) if x.isdigit() else -1):
            class_dir = os.path.join(data_root, entry)
            if not os.path.isdir(class_dir):
                continue
            if not entry.isdigit():
                continue

            class_idx = int(entry)

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png', '.JPEG')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

        print(f"  Loaded {len(self.samples)} images from {len(set(l for _, l in self.samples))} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class ImageNetSketchDataset(Dataset):
    """
    Dataset for ImageNet-Sketch.

    Uses synset folder names, covers all 1000 ImageNet classes.

    Directory structure:
        data_root/
            n01440764/
                sketch1.JPEG
            n01443537/
                ...
    """

    def __init__(self, data_root: str, transform=None):
        self.transform = transform or IMAGENET_TRANSFORM
        self.data_root = data_root

        synset_mapping = get_imagenet_synset_mapping()

        self.samples = []

        for entry in sorted(os.listdir(data_root)):
            class_dir = os.path.join(data_root, entry)
            if not os.path.isdir(class_dir):
                continue
            if entry not in synset_mapping:
                continue

            class_idx = synset_mapping[entry]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png', '.JPEG')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

        print(f"  Loaded {len(self.samples)} images from {len(set(l for _, l in self.samples))} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_natural_shift_loader(
    dataset_name: str,
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = False,
    seed: int = None
) -> DataLoader:
    """
    Get dataloader for natural distribution shift datasets.

    Args:
        dataset_name: 'imagenet_r', 'imagenet_a', 'imagenet_v2', 'imagenet_sketch'
        data_root: Path to dataset root directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        seed: Random seed for reproducible shuffling

    Returns:
        DataLoader for the dataset
    """
    name = dataset_name.lower().replace('-', '_')

    if name in ('imagenet_r', 'imagenetr'):
        dataset = ImageNetSubsetDataset(data_root=data_root)
    elif name in ('imagenet_a', 'imagineta'):
        dataset = ImageNetSubsetDataset(data_root=data_root)
    elif name in ('imagenet_v2', 'imagenetv2'):
        dataset = ImageNetV2Dataset(data_root=data_root)
    elif name in ('imagenet_sketch', 'imagenetsketch'):
        dataset = ImageNetSketchDataset(data_root=data_root)
    else:
        raise ValueError(f"Unknown natural shift dataset: {dataset_name}")

    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator if shuffle else None
    )
