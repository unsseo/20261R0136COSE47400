import os
import random
import numpy as np
import pandas as pd

import torch
import torchaudio
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold

try:
    from IPython.display import clear_output as _clear_output

    def clear_output(wait=False):
        _clear_output(wait=wait)
except ImportError:
    def clear_output(wait=False):
        os.system("cls" if os.name == "nt" else "clear")


ELICE_PROJECT_DIR = "/home/elicer"
ELICE_DATASET_DIR = "/mnt/elice/dataset"

PROJECT_DIR = ELICE_PROJECT_DIR if os.path.isdir(ELICE_PROJECT_DIR) else (
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
)

DEFAULT_DATA_PATH = os.path.join(ELICE_DATASET_DIR, "birdclef-2026") if os.path.isdir(ELICE_DATASET_DIR) else os.path.join(PROJECT_DIR, "birdclef-2026")
DATA_PATH = os.environ.get("BIRDCLEF_DATA_PATH", DEFAULT_DATA_PATH)
HISTORY_DIR = os.path.join(PROJECT_DIR, "history")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Spectrogram(nn.Module):
    def __init__(self, sr=32000, n_fft=2048, n_mels=256, hop_length=512, f_min=20, f_max=16000, channels=1, norm="slaney", mel_scale="htk", target_size=(256, 256), top_db=80.0, delta_win=5,):
        super().__init__()
        self.channels = channels
        self.top_db = top_db

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            mel_scale=mel_scale,
            pad_mode="reflect",
            power=2.0,
            norm=norm,
            center=True,
        )

        self.resize = torchvision.transforms.Resize(size=target_size)

    def power_to_db(self, S):
        amin = 1e-10
        log_spec = 10.0 * torch.log10(S.clamp(min=amin))
        log_spec -= 10.0 * torch.log10(torch.tensor(amin).to(S))
        if self.top_db is not None:
            max_val = log_spec.flatten(-2).max(dim=-1).values[..., None, None]
            log_spec = torch.maximum(log_spec, max_val - self.top_db)
        return log_spec

    def forward(self, x):
        # x: (B, T) or (T,)
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        mel_spec = self.mel_transform(x)          # (B, n_mels, time)
        mel_spec = self.power_to_db(mel_spec)

        mel_spec = mel_spec.unsqueeze(1).repeat(1, self.channels, 1, 1)
        mel_spec = self.resize(mel_spec)           # (B, C, H, W)

        B, C = mel_spec.shape[:2]
        flat = mel_spec.view(B, C, -1)
        mins = flat.min(dim=-1).values[..., None, None]
        maxs = flat.max(dim=-1).values[..., None, None]
        mel_spec = (mel_spec - mins) / (maxs - mins + 1e-7)

        if squeeze:
            mel_spec = mel_spec.squeeze(0)

        return mel_spec


class BirdDataset(Dataset):
    PATH = DATA_PATH

    config = {"sr":32000, 'seed':2, 'train_only':False}

    def __init__(self, is_train=True, fold=0, config={}):
        self.config.update(config)

        self.PATH = self.config.get("data_path", self.PATH)
        self.SND_PATH = os.path.join(self.PATH, "train_audio")

        df = pd.read_csv(os.path.join(self.PATH, "train.csv"))
        tax = pd.read_csv(os.path.join(self.PATH, "taxonomy.csv"))
        self.LABELS = list(np.unique(tax.primary_label.dropna()))

        df.index = df.filename.values

        IDX = np.unique(df.index)
        np.random.seed(self.config['seed'])
        np.random.shuffle(IDX)

        if self.config['train_only']: idx = IDX
        else:
            skf = StratifiedKFold(n_splits=5)
            FOLDS = list(skf.split(IDX, df.loc[IDX].primary_label.fillna('none').values))
            train_idx = IDX[FOLDS[fold][0]].tolist()
            val_idx = IDX[FOLDS[fold][1]].tolist()
            idx = train_idx if is_train else val_idx

        DF = df.loc[idx].copy()

        self.paths = [os.path.join(self.SND_PATH, filename) for filename in DF['filename'].values]
        labels = DF['primary_label'].apply(self.make_labels).values
        self.labels = labels

        self.is_train = is_train

    def make_labels(self, X):
        out = np.zeros(len(self.LABELS)).astype(bool)
        out[self.LABELS.index(X)] = True

        return out

    def load_sound(self, filepath, start=0, DUR=5*32000):
        wav, sr = torchaudio.load(filepath)
        wav = wav[0]

        l = len(wav)

        if l < DUR:
            # If the audio is less than 5s long -> Pad it with 0's
            wav2 = torch.zeros((DUR))
            s = np.random.randint(DUR-l)
            wav2[s:s+l] = wav
            wav = wav2
        else:
            # Otherwise -> Crop it
            if self.is_train:
                s = random.randint(0, l-DUR)
                wav = wav[s:s+DUR]
            else:
                wav = wav[:DUR]

        return wav

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        DUR = int(5 * self.config['sr'])

        audio = self.load_sound(path, DUR=DUR)
        labels = self.labels[idx]

        return (
            audio,
            torch.tensor(labels, dtype=torch.float32)
        )


def create_dataloaders(config={'batch_size':32, 'num_workers':0}, fold=0):
    train_dataset = BirdDataset(is_train=True, config=config, fold=fold)
    val_dataset = BirdDataset(is_train=False, config=config, fold=fold)

    torch.manual_seed(config['seed'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    targets = {"labels":train_dataset.LABELS}

    return train_loader, val_loader, targets


from sklearn.metrics import roc_auc_score


def AUC(targets, outputs, verbose=False):
    targets = (targets>0).astype(float)
    num_classes = targets.shape[1]
    scored_classes = (np.sum(targets,axis=0) > 0)
    auc = roc_auc_score(targets[:,scored_classes], outputs[:,scored_classes], average='macro')
    return auc


import timm


class BirdModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = {
            'scale':1,
            'backbone_pooling':'avg',
            'backbone':'tf_efficientnetv2_b0',
            'dropout':0.1,
            'pretrained':True,
            'channels':1,
            'num_labels':234,
        }
        if config: self.config.update(config)

        self.training = True

        self.backbone = timm.create_model(
            self.config['backbone'],
            pretrained=self.config['pretrained'],
            num_classes=self.config['num_labels'],
            global_pool=self.config['backbone_pooling'],
            in_chans=self.config['channels'],
            drop_rate=self.config['dropout'],
        )
        feature_dim = self.backbone.num_features
        print(self.config['num_labels'])

    def forward(self, x):
        labels = self.backbone(x)
        return labels


class Mixup(nn.Module):
    def __init__(self, alpha=0.5, theta=1):
        super().__init__()
        self.alpha = alpha
        self.theta = theta

    def forward(self, x, y):
        batch_size = x.size(0)

        lam = torch.tensor(np.random.beta(self.alpha,self.alpha, batch_size)).to(x.device)
        lam = torch.maximum(lam, 1-lam).float()

        idx = torch.randperm(batch_size).to(x.device)

        view_shape = (batch_size,) + (1,) * (x.dim() - 1)
        lam_x = lam.view(view_shape)
        x = lam_x * x + (1 - lam_x) * x[idx]
        if isinstance(y, list):
            for i in range(len(y)):
                y[i] = lam[...,None] * y[i] + (1 - lam)[...,None] * y[i][idx]
                y[i][y[i]>=self.theta] = 1
        else:
            y = lam[...,None] * y + (1 - lam)[...,None] * y[idx]
            y[y>=self.theta] = 1

        return x, y


from tqdm import tqdm
import datetime
import hashlib
import time


CFG = {
    'seed':2,
    "batch_size":32,
    "num_workers":4,
    "train_only":False,
    "lr":5e-4,
    "loss":nn.BCEWithLogitsLoss(),
    'backbone_pooling':'avg',
    'backbone':'tf_efficientnetv2_b0',
    'dropout':.2,
    'verbose':2,
    'mel':{'n_mels':256, 'f_min':20, 'n_fft':2048, 'target_size':(256,256), 'mel_scale':'slaney', 'norm':'slaney'},
    'metrics':[AUC],
    'scheduler':True,
    'model':BirdModel,
    'num_labels':234
}


class Trainer:

    def __init__(self, config={}, fold=0, epochs=16):
        self.config = CFG.copy()
        self.config.update(config)
        set_seed(self.config['seed'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.exp_id = hashlib.sha256(str(time.time()).encode()).hexdigest()

        cols = ['id', 'epoch', 'train_loss', 'val_loss', 'lr', 'timestamp', 'fold']+['val_'+m.__name__ for m in self.config['metrics']]+list(self.config.keys())
        self.history = pd.DataFrame([], columns=cols)

        self.batch_size = self.config['batch_size']
        self.fold = fold

        self.mel = Spectrogram(**config['mel']).to(self.device)

    def train_one_epoch(self, epoch=0):
        self.model.train()
        Loss = 0
        n_steps = len(self.train_loader)
        batch_size = self.config['batch_size']

        if self.config['verbose']==2: pbar = tqdm(enumerate(self.train_loader), total=n_steps, desc="Training")
        else: pbar = enumerate(self.train_loader)

        mix = Mixup(**self.config['mix'])

        for batch_idx, (x, y) in pbar:
            set_seed(self.config['seed']+2048*epoch+batch_idx)
            self.optimizer.zero_grad()

            x = x.to(self.device)
            y = y.to(self.device)

            x, y = mix(x, y)
            x = self.mel(x)

            logits = self.model(x)

            L = self.loss_fn(logits, y)
            L.backward()

            self.optimizer.step()

            Loss += L.detach().item()

        return Loss

    def validate(self):
        self.model.eval()
        Loss = 0

        n_steps = len(self.val_loader)
        out_shape = (n_steps*self.batch_size, self.config['num_labels'])

        if self.config['verbose']==2: pbar = tqdm(enumerate(self.val_loader), total=n_steps, desc="Validation")
        else: pbar = enumerate(self.val_loader)

        pred = np.zeros((n_steps, self.batch_size, self.config['num_labels']))
        target = np.zeros((n_steps, self.batch_size, self.config['num_labels']))

        with torch.no_grad():
            for batch_idx, (x, y) in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                x = self.mel(x)

                logits = self.model(x)
                loss = self.loss_fn(logits, y)

                Loss += loss.detach().item()
                pred[batch_idx] = logits.sigmoid().detach().cpu().numpy()
                target[batch_idx] = y.detach().cpu().numpy()

        target = np.reshape(target, out_shape)
        pred = np.reshape(pred, out_shape)
        scores = []

        for m in self.config['metrics']:
            scores.append(m(target, pred))
        return scores, Loss

    def train(self, epochs=16, checkpoint_freq='once'):
        set_seed(self.config['seed'])

        self.train_loader, self.val_loader, self.targets = create_dataloaders(fold=self.fold, config=self.config)
        self.config['num_labels'] = len(self.targets['labels'])
        self.epochs = epochs

        self.model = self.config['model'](config=self.config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs, eta_min=1e-8)
        self.loss_fn = self.config['loss']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        best = (np.inf, 0, 0)



        Epochs = range(epochs)
        for epoch in Epochs:
            torch.manual_seed(self.config['seed']+epoch)

            train_loss = self.train_one_epoch(epoch=epoch)

            if not self.config['train_only']:
                val_scores, val_loss = self.validate()
                if val_loss<best[0]: best = (val_loss, val_scores,epoch)
            else: val_scores, val_loss = [None] * len(self.config['metrics']), None


            print(len([self.exp_id, epoch, train_loss, val_loss, self.optimizer.param_groups[0]['lr'], str(datetime.datetime.now()), self.fold, *val_scores]+list(self.config.values())))
            print(len(val_scores))
            print(len(self.config))
            print(self.history.shape)
            self.history.loc[len(self.history)] = [self.exp_id, epoch, train_loss, val_loss, self.optimizer.param_groups[0]['lr'], str(datetime.datetime.now()), self.fold, *val_scores]+list(self.config.values())
            self.history.to_csv(os.path.join(HISTORY_DIR, f"{self.exp_id}.csv"), index=False)
            if checkpoint_freq=='epoch': torch.save(self.model.state_dict(), os.path.join(MODELS_DIR, f"{self.exp_id}_{epoch}.pth"))

            clear_output(wait=False)
            print(self.exp_id, '\n')
            print(f"\033[1m Epoch {epoch+1}/{epochs}")
            print(f'\033[1m Training \t|\t loss={np.round(train_loss, 3)}' + '\033[0m')
            if not self.config['train_only']:
                print(f'\033[1m Validation \t|\t loss={np.round(val_loss, 3)}  -  ' + '  -  '.join([f'{m.__name__}={np.round(s,3)}' for m,s in zip(self.config['metrics'], val_scores)])+'\033[0m')
                print()
                print(f"\033[1m Best : {'  -  '.join([f'{m.__name__}={np.round(s,3)}' for m,s in zip(self.config['metrics'], best[1])])} at epoch {best[2]}")

            if self.config['scheduler']: self.scheduler.step()

        if not self.config['train_only']:
            _ = self.validate()

        if checkpoint_freq=='once': torch.save(self.model.state_dict(), f"models/{self.exp_id}.pth")

        return self.model


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    config = {
    'seed': 2,
    'batch_size': 32,
    'num_workers': 4,
    "backbone": "tf_efficientnetv2_b0",
    "loss": nn.BCEWithLogitsLoss(),
    'mel': {'n_mels': 256, 'f_min': 20, 'n_fft': 2048, 'target_size': (256,256)},
    "mix": {"alpha": 0.5, "theta": 0.8},
    "pretrained": True,
    "model": BirdModel,
    'train_only': False,
    'data_path': DATA_PATH,
}

    trainer = Trainer(config=config, fold=0)
    model = trainer.train(epochs=20)
