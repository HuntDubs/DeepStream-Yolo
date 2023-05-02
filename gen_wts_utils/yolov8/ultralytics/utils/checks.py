import re
from pathlib import Path
import glob
from ultralytics.utils.torch_utils import LOGGER

from ultralytics.utils.torch_utils import ROOT, clean_url, url2file

def check_suffix(file='yolov8n.pt', suffix='.pt', msg=''):
    """Check file(s) for acceptable suffix."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix, )
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()  # file suffix
            if len(s):
                assert s in suffix, f'{msg}{f} acceptable suffix is {suffix}, not {s}'

def check_yolov5u_filename(file: str, verbose: bool = True):
    """Replace legacy YOLOv5 filenames with updated YOLOv5u filenames."""
    if ('yolov3' in file or 'yolov5' in file) and 'u' not in file:
        original_file = file
        file = re.sub(r'(.*yolov5([nsmlx]))\.pt', '\\1u.pt', file)  # i.e. yolov5n.pt -> yolov5nu.pt
        file = re.sub(r'(.*yolov5([nsmlx])6)\.pt', '\\1u.pt', file)  # i.e. yolov5n6.pt -> yolov5n6u.pt
        file = re.sub(r'(.*yolov3(|-tiny|-spp))\.pt', '\\1u.pt', file)  # i.e. yolov3-spp.pt -> yolov3-sppu.pt
        if file != original_file and verbose:
            LOGGER.info(f"PRO TIP 💡 Replace 'model={original_file}' with new 'model={file}'.\nYOLOv5 'u' models are "
                        f'trained with https://github.com/ultralytics/ultralytics and feature improved performance vs '
                        f'standard YOLOv5 models trained with https://github.com/ultralytics/yolov5.\n')
    return file

def check_file(file, suffix='', download=True, hard=True):
    """Search/download file (if necessary) and return path."""
    check_suffix(file, suffix)  # optional
    file = str(file).strip()  # convert to string and strip spaces
    file = check_yolov5u_filename(file)  # yolov5n -> yolov5nu
    if not file or ('://' not in file and Path(file).exists()):  # exists ('://' check required in Windows Python<3.10)
        return file
    elif download and file.lower().startswith(('https://', 'http://', 'rtsp://', 'rtmp://')):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = url2file(file)  # '%2F' to '/', split https://url.com/file.txt?auth
        if Path(file).exists():
            LOGGER.info(f'Found {clean_url(url)} locally at {file}')  # file already exists
        else:
            # safe_download(url=url, file=file, unzip=False)
            LOGGER.info("Commented out safe-downlaod functinality. Safe path to file does not exist.\n")
        return file
    else:  # search
        files = []
        for d in 'models', 'datasets', 'tracker/cfg', 'yolo/cfg':  # search directories
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  # find file
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else []  # return file

def check_yaml(file, suffix=('.yaml', '.yml'), hard=True):
    """Search/download YAML file (if necessary) and return path, checking suffix."""
    return check_file(file, suffix, hard=hard)