"""프로젝트 공통 설정: 경로, 하이퍼파라미터, 클래스 매핑.

데이터 경로 우선순위: CLI 인자 > 환경변수 > 아래 기본값
  SKIN_DATA_ROOT   : 데이터 루트 (이미지/라벨 루트의 기본값)
  SKIN_IMAGE_ROOT  : JSON의 photograph.file_path 가 상대경로로 붙는 기준 디렉토리
  SKIN_LABEL_ROOT  : .json 을 재귀 수집할 디렉토리
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ 의 상위 = 프로젝트 루트

# ---------------------------------------------------------------- paths
DATA_ROOT = Path(os.environ.get("SKIN_DATA_ROOT", PROJECT_ROOT / "data"))
IMAGE_ROOT = Path(os.environ.get("SKIN_IMAGE_ROOT", DATA_ROOT))
LABEL_ROOT = Path(os.environ.get("SKIN_LABEL_ROOT", DATA_ROOT))
SPLIT_DIR = Path(os.environ.get("SKIN_SPLIT_DIR", PROJECT_ROOT / "splits"))
RESULTS_DIR = Path(os.environ.get("SKIN_RESULTS_DIR", PROJECT_ROOT / "results"))
CHECKPOINT_DIR = Path(os.environ.get("SKIN_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))

# ---------------------------------------------------------------- classes
CLASS_NAMES = ["건선", "아토피", "여드름", "정상", "주사", "지루"]
CLASS_NAMES_EN = ["psoriasis", "atopic", "acne", "normal", "rosacea", "seborrheic"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

VIEWS = ["정면", "측면"]
VIEW_EN = {"정면": "front", "측면": "side"}

# 명세상 기대 개수 (다르면 prepare_data.py 가 경고만 출력)
EXPECTED_PER_CLASS = 2000
EXPECTED_PER_CLASS_VIEW = 1000

# ---------------------------------------------------------------- baseline (NIA 공식값)
SEED = 42
SPLIT_RATIO = (0.8, 0.1, 0.1)  # train / valid / test
IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 11
LR = 1e-3
DROPOUT = 0.5
OPTIMIZER = "adam"

# ---------------------------------------------------------------- runtime
NUM_WORKERS = int(os.environ.get("SKIN_NUM_WORKERS", 8))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def add_path_args(parser):
    """데이터 경로 관련 공통 CLI 인자."""
    parser.add_argument("--data-root", type=Path, default=None,
                        help="이미지/라벨 루트를 한 번에 지정 (각각 --image-root/--label-root 로 덮어쓰기 가능)")
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--label-root", type=Path, default=None)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    return parser


def resolve_roots(args):
    """CLI 인자 > 환경변수 > 기본값 순으로 (image_root, label_root) 결정."""
    image_root = args.image_root or args.data_root or IMAGE_ROOT
    label_root = args.label_root or args.data_root or LABEL_ROOT
    return Path(image_root).resolve(), Path(label_root).resolve()
