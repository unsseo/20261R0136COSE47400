import os
import random
import time
import datetime
import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import timm
import torch
import torchaudio
import torchvision
import torch.nn as nn
from IPython.display import clear_output
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


os.makedirs("history", exist_ok=True)
os.makedirs("models", exist_ok=True)


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
    def __init__(
        self,
        sr=32000,
        n_fft=2048,
        n_mels=256,
        hop_length=512,
        f_min=20,
        f_max=16000,
        channels=1,
        norm="slaney",
        mel_scale="htk",
        target_size=(256, 256),
        top_db=80.0,
        delta_win=5,
    ):
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

    def power_to_db(self, spectrogram):
        amin = 1e-10
        log_spec = 10.0 * torch.log10(spectrogram.clamp(min=amin))
        log_spec -= 10.0 * torch.log10(torch.tensor(amin).to(spectrogram))
        if self.top_db is not None:
            max_val = log_spec.flatten(-2).max(dim=-1).values[..., None, None]
            log_spec = torch.maximum(log_spec, max_val - self.top_db)
        return log_spec

    def forward(self, audio, resize=True):
        squeeze = False
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze = True

        mel_spec = self.mel_transform(audio)
        mel_spec = self.power_to_db(mel_spec)

        mel_spec = mel_spec.unsqueeze(1).repeat(1, self.channels, 1, 1)
        if resize:
            mel_spec = self.resize(mel_spec)

        batch_size, channels = mel_spec.shape[:2]
        flat = mel_spec.view(batch_size, channels, -1)
        mins = flat.min(dim=-1).values[..., None, None]
        maxs = flat.max(dim=-1).values[..., None, None]
        mel_spec = (mel_spec - mins) / (maxs - mins + 1e-7)

        if squeeze:
            mel_spec = mel_spec.squeeze(0)

        return mel_spec


class BirdTrainDataset(Dataset):
    PATH = "/kaggle/input/competitions/birdclef-2026/"
    config = {"sr": 32000, "seed": 2, "train_only": False}

    def __init__(self, is_train=True, fold=0, config=None):
        self.config.update(config or {})

        self.SND_PATH = self.PATH + "train_audio/"

        df = pd.read_csv(self.PATH + "train.csv")
        tax = pd.read_csv(self.PATH + "taxonomy.csv")
        self.LABELS = list(np.unique(tax.primary_label.dropna()))

        df.index = df.filename.values

        idx_values = np.unique(df.index)
        np.random.seed(self.config["seed"])
        np.random.shuffle(idx_values)

        if self.config["train_only"]:
            idx = idx_values
        else:
            skf = StratifiedKFold(n_splits=5)
            folds = list(
                skf.split(
                    idx_values,
                    df.loc[idx_values].primary_label.fillna("none").values,
                )
            )
            train_idx = idx_values[folds[fold][0]].tolist()
            val_idx = idx_values[folds[fold][1]].tolist()
            idx = train_idx if is_train else val_idx

        df = df.loc[idx].copy()

        self.paths = list(self.SND_PATH + df["filename"].values)
        self.labels = df["primary_label"].apply(self.make_labels).values
        self.is_train = is_train

    def make_labels(self, label):
        out = np.zeros(len(self.LABELS)).astype(bool)
        out[self.LABELS.index(label)] = True
        return out

    def load_sound(self, filepath, start=0, duration=5 * 32000):
        del start
        wav, sr = torchaudio.load(filepath)
        del sr
        wav = wav[0]

        length = len(wav)
        if length < duration:
            wav2 = torch.zeros((duration))
            offset = np.random.randint(duration - length)
            wav2[offset : offset + length] = wav
            wav = wav2
        else:
            if self.is_train:
                offset = random.randint(0, length - duration)
                wav = wav[offset : offset + duration]
            else:
                wav = wav[:duration]

        return wav

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        duration = int(5 * self.config["sr"])

        audio = self.load_sound(path, duration=duration)
        labels = self.labels[idx]

        return audio, torch.tensor(labels, dtype=torch.float32)


def create_dataloaders(config=None, fold=0):
    cfg = {"batch_size": 32, "num_workers": 0, "seed": 2, "train_only": False, "sr": 32000}
    cfg.update(config or {})

    train_dataset = BirdTrainDataset(is_train=True, config=cfg, fold=fold)
    val_dataset = BirdTrainDataset(is_train=False, config=cfg, fold=fold)

    torch.manual_seed(cfg["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    targets = {"labels": train_dataset.LABELS}
    return train_loader, val_loader, targets


def AUC(targets, outputs, verbose=False):
    del verbose
    targets = (targets > 0).astype(float)
    scored_classes = np.sum(targets, axis=0) > 0
    auc = roc_auc_score(
        targets[:, scored_classes],
        outputs[:, scored_classes],
        average="macro",
    )
    return auc


class BirdModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = {
            "scale": 1,
            "backbone_pooling": "avg",
            "backbone": "tf_efficientnetv2_b0",
            "dropout": 0.1,
            "pretrained": True,
            "channels": 1,
            "num_labels": 234,
        }
        if config:
            self.config.update(config)

        self.backbone = timm.create_model(
            self.config["backbone"],
            pretrained=self.config["pretrained"],
            num_classes=self.config["num_labels"],
            global_pool=self.config["backbone_pooling"],
            in_chans=self.config["channels"],
            drop_rate=self.config["dropout"],
        )

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

        lam = torch.tensor(np.random.beta(self.alpha, self.alpha, batch_size)).to(x.device)
        lam = torch.maximum(lam, 1 - lam).float()

        idx = torch.randperm(batch_size).to(x.device)

        x = lam[..., None, None, None] * x + (1 - lam)[..., None, None, None] * x[idx]
        if isinstance(y, list):
            for i in range(len(y)):
                y[i] = lam[..., None] * y[i] + (1 - lam)[..., None] * y[i][idx]
                y[i][y[i] >= self.theta] = 1
        else:
            y = lam[..., None] * y + (1 - lam)[..., None] * y[idx]
            y[y >= self.theta] = 1

        return x, y


CFG = {
    "seed": 2,
    "batch_size": 32,
    "num_workers": 4,
    "train_only": False,
    "lr": 5e-4,
    "loss": nn.BCEWithLogitsLoss(),
    "backbone_pooling": "avg",
    "backbone": "tf_efficientnetv2_b0",
    "dropout": 0.2,
    "verbose": 2,
    "mel": {
        "n_mels": 256,
        "f_min": 20,
        "n_fft": 2048,
        "target_size": (256, 256),
        "mel_scale": "slaney",
        "norm": "slaney",
    },
    "metrics": [AUC],
    "scheduler": True,
    "model": BirdModel,
    "num_labels": 234,
}


class Trainer:
    def __init__(self, config=None, fold=0, epochs=16):
        del epochs
        self.config = CFG.copy()
        self.config.update(config or {})
        set_seed(self.config["seed"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_id = hashlib.sha256(str(time.time()).encode()).hexdigest()

        cols = [
            "id",
            "epoch",
            "train_loss",
            "val_loss",
            "lr",
            "timestamp",
            "fold",
        ] + [f"val_{metric.__name__}" for metric in self.config["metrics"]] + list(self.config.keys())
        self.history = pd.DataFrame([], columns=cols)

        self.batch_size = self.config["batch_size"]
        self.fold = fold
        self.mel = Spectrogram(**self.config["mel"]).to(self.device)

    def train_one_epoch(self, epoch=0):
        self.model.train()
        loss_total = 0
        n_steps = len(self.train_loader)

        if self.config["verbose"] == 2:
            pbar = tqdm(enumerate(self.train_loader), total=n_steps, desc="Training")
        else:
            pbar = enumerate(self.train_loader)

        mix = Mixup(**self.config["mix"])

        for batch_idx, (x, y) in pbar:
            set_seed(self.config["seed"] + 2048 * epoch + batch_idx)
            self.optimizer.zero_grad()

            x = x.to(self.device)
            y = y.to(self.device)

            x = self.mel(x)
            x, y = mix(x, y)
            logits = self.model(x)

            loss = self.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.detach().item()

        return loss_total

    def validate(self):
        self.model.eval()
        loss_total = 0

        n_steps = len(self.val_loader)
        out_shape = (n_steps * self.batch_size, self.config["num_labels"])

        if self.config["verbose"] == 2:
            pbar = tqdm(enumerate(self.val_loader), total=n_steps, desc="Validation")
        else:
            pbar = enumerate(self.val_loader)

        pred = np.zeros((n_steps, self.batch_size, self.config["num_labels"]))
        target = np.zeros((n_steps, self.batch_size, self.config["num_labels"]))

        with torch.no_grad():
            for batch_idx, (x, y) in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                x = self.mel(x)
                logits = self.model(x)
                loss = self.loss_fn(logits, y)

                loss_total += loss.detach().item()
                pred[batch_idx] = logits.sigmoid().detach().cpu().numpy()
                target[batch_idx] = y.detach().cpu().numpy()

        target = np.reshape(target, out_shape)
        pred = np.reshape(pred, out_shape)
        scores = []

        for metric in self.config["metrics"]:
            scores.append(metric(target, pred))
        return scores, loss_total

    def train(self, epochs=16, checkpoint_freq="once"):
        set_seed(self.config["seed"])

        self.train_loader, self.val_loader, self.targets = create_dataloaders(
            fold=self.fold,
            config=self.config,
        )
        self.config["num_labels"] = len(self.targets["labels"])
        self.epochs = epochs

        self.model = self.config["model"](config=self.config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["lr"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            epochs,
            eta_min=1e-8,
        )
        self.loss_fn = self.config["loss"]
        self.model = self.model.to(self.device)

        best = (np.inf, 0, 0)

        for epoch in range(epochs):
            torch.manual_seed(self.config["seed"] + epoch)
            train_loss = self.train_one_epoch(epoch=epoch)

            if not self.config["train_only"]:
                val_scores, val_loss = self.validate()
                if val_loss < best[0]:
                    best = (val_loss, val_scores, epoch)
            else:
                val_scores, val_loss = [None] * len(self.config["metrics"]), None

            self.history.loc[len(self.history)] = [
                self.exp_id,
                epoch,
                train_loss,
                val_loss,
                self.optimizer.param_groups[0]["lr"],
                str(datetime.datetime.now()),
                self.fold,
                *val_scores,
            ] + list(self.config.values())
            self.history.to_csv(f"history/{self.exp_id}.csv", index=False)

            if checkpoint_freq == "epoch":
                torch.save(self.model.state_dict(), f"models/{self.exp_id}_{epoch}.pth")

            clear_output(wait=False)
            print(self.exp_id, "\n")
            print(f"\033[1m Epoch {epoch + 1}/{epochs}")
            print(f"\033[1m Training \t|\t loss={np.round(train_loss, 3)}\033[0m")
            if not self.config["train_only"]:
                score_text = "  -  ".join(
                    [
                        f"{metric.__name__}={np.round(score, 3)}"
                        for metric, score in zip(self.config["metrics"], val_scores)
                    ]
                )
                best_text = "  -  ".join(
                    [
                        f"{metric.__name__}={np.round(score, 3)}"
                        for metric, score in zip(self.config["metrics"], best[1])
                    ]
                )
                print(
                    f"\033[1m Validation \t|\t loss={np.round(val_loss, 3)}  -  {score_text}\033[0m"
                )
                print()
                print(f"\033[1m Best : {best_text} at epoch {best[2]}")

            if self.config["scheduler"]:
                self.scheduler.step()

        if not self.config["train_only"]:
            _ = self.validate()

        if checkpoint_freq == "once":
            torch.save(self.model.state_dict(), f"models/{self.exp_id}.pth")

        return self.model


TRAIN_CONFIG = {
    "seed": 2,
    "batch_size": 32,
    "backbone": "tf_efficientnetv2_b0",
    "loss": nn.BCEWithLogitsLoss(),
    "mel": {
        "n_mels": 256,
        "f_min": 20,
        "n_fft": 2048,
        "target_size": (256, 256),
    },
    "mix": {"alpha": 0.5, "theta": 0.8},
    "pretrained": True,
    "model": BirdModel,
    "train_only": False,
}


class BirdInferenceDataset(Dataset):
    PATH = "/kaggle/input/competitions/birdclef-2026/"
    TEST_PATH = PATH + "test_soundscapes/"
    TRAIN_PATH = PATH + "train_soundscapes/"
    taxonomy = pd.read_csv(PATH + "taxonomy.csv")

    LABELS = list(np.unique(taxonomy.primary_label))
    CLASSES = list(np.unique(taxonomy.class_name))
    BATCH_SIZE = 32
    DUR = 5
    SR = 32000

    def __init__(self, split_size=0.2, seed=2, n_repeat=1, is_train=True):
        del split_size, seed, n_repeat, is_train
        paths = [self.TEST_PATH + x for x in os.listdir(self.TEST_PATH) if ".ogg" in x]
        if len(paths) == 0:
            paths = [self.TRAIN_PATH + x for x in os.listdir(self.TRAIN_PATH) if ".ogg" in x]
            paths = sorted(paths)[:16]
        self.paths = paths.copy()

    def __len__(self):
        return len(self.paths)


def format_time(seconds):
    hours = int(seconds / 3600)
    seconds -= hours * 3600
    minutes = int(seconds / 60)
    seconds -= minutes * 60

    out = ""
    if hours > 0:
        out += str(hours) + ":"
    return out + str(minutes).zfill(2) + ":" + str(int(seconds)).zfill(2)


def decode_config(cfg):
    for key in cfg:
        try:
            cfg[key] = eval(cfg[key])
        except Exception:
            pass
    return cfg


def predict(filepath, dataset, spec, model):
    wav, sr = torchaudio.load(filepath)
    del sr
    n_seg = int(60 / dataset.DUR)
    wav = wav.float()[:, : dataset.SR * 60]
    n_repeat = 1
    wav = wav.float().reshape((n_seg, dataset.SR * dataset.DUR))

    activation = nn.Sigmoid()
    preds = []
    with torch.no_grad():
        mel = torch.stack([spec(wav[i]) for i in range(len(wav))])
        for _ in range(n_repeat):
            preds.append(activation(model(mel).unsqueeze(0)))
    preds = torch.concat(preds)
    pred = torch.mean(preds, dim=0)

    names = [
        clip_id + "_" + t
        for clip_id, t in zip(
            [filepath.split("/")[-1].split(".")[0]] * n_seg,
            (np.array(range(n_seg)) * dataset.DUR + dataset.DUR).astype(str),
        )
    ]

    return pred.numpy(), names


def build_submission(history_path, model_path, max_workers=4):
    dataset = BirdInferenceDataset()
    history = pd.read_csv(history_path)
    config = {x: history.iloc[0][x] for x in history.columns[7:]}

    cfg = decode_config(config)
    cfg.update({"pretrained": False})

    spec = Spectrogram(**cfg["mel"])
    model = BirdModel(config=cfg)
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))
    )

    labels_audio = np.unique(pd.read_csv(dataset.PATH + "train.csv").primary_label)

    pred = []
    names = []
    model.eval()
    start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for p, n in executor.map(lambda path: predict(path, dataset, spec, model), dataset.paths):
            pred.append(p)
            names.append(n)

            fps = len(pred) / (time.time() - start)
            if len(pred) % 16 == 0:
                remaining = (len(dataset) - len(pred)) / fps
                print(
                    np.round(100 * len(pred) / len(dataset), 1),
                    "%",
                    format_time(time.time() - start),
                    "  -  remaining:",
                    format_time(remaining),
                )

    pred = np.concatenate(pred, axis=0)
    row_ids = np.concatenate(names, axis=0)

    pred_df = pd.DataFrame(np.zeros((len(pred), 234)), columns=dataset.LABELS)
    pred_df.loc[:, labels_audio] = pred
    pred_df.insert(0, "row_id", row_ids)
    return pred_df, dataset, spec


def visualize_prediction_window(pred, dataset, spec, file_index=0):
    import plotly.express as px

    filepath = dataset.paths[file_index]
    wav, sr = torchaudio.load(filepath)
    del sr
    mel = spec(wav, resize=False)[0, 0]

    fig = px.imshow(
        pred.T[:, file_index * 12 : (file_index + 1) * 12],
        color_continuous_scale="Oranges",
        aspect="auto",
        zmax=1,
    )
    fig.update_yaxes(autorange=True)
    fig.show(renderer="iframe")

    fig = px.imshow(mel, aspect="auto")
    fig.show(renderer="iframe")


if __name__ == "__main__":
    print("Saved BirdCLEF 2026 baseline code.")
    print("Use TRAIN_CONFIG with Trainer(...) for training.")
    print("Use build_submission(history_path, model_path) for inference.")
