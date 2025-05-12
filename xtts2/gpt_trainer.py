import os
import gc

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.utils.manage import ModelManager
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs,
    GPTTrainer,
    GPTTrainerConfig,
    XttsAudioConfig,
)


def train(
    metadatas,
    num_epochs,
    batch_size,
    grad_acumm,
    output_path,
    max_audio_length,
    max_text_length,
    lr,
    weight_decay,
    save_step,
):
    # Preparación de info del dataset
    configList = []
    for metadata in metadatas:
        train_csv, eval_csv, language = metadata.split(",")

        config_dataset = BaseDatasetConfig(
            formatter="coqui",
            dataset_name="ft_dataset",
            path=os.path.dirname(train_csv),
            meta_file_train=os.path.basename(train_csv),
            meta_file_val=os.path.basename(eval_csv),
            language=language,
        )

        configList.append(config_dataset)

    # Descarga del modelo pre entrenado
    pretrainedPath = os.path.join(output_path, "xtts2_original/")
    os.makedirs(pretrainedPath, exist_ok=True)

    DVAE_CHECKPOINT_LINK = (
        "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    )
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

    DVAE_CHECKPOINT = os.path.join(
        pretrainedPath, os.path.basename(DVAE_CHECKPOINT_LINK)
    )
    MEL_NORM_FILE = os.path.join(pretrainedPath, os.path.basename(MEL_NORM_LINK))

    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        ModelManager._download_model_files(
            [MEL_NORM_LINK, DVAE_CHECKPOINT_LINK],
            pretrainedPath,
            progress_bar=True,
        )

        # Download XTTS v2.0 checkpoint if needed
    TOKENIZER_FILE_LINK = (
        "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    )
    XTTS_CHECKPOINT_LINK = (
        "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
    )
    XTTS_CONFIG_LINK = (
        "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"
    )

    # XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
    TOKENIZER_FILE = os.path.join(
        pretrainedPath, os.path.basename(TOKENIZER_FILE_LINK)
    )  # vocab.json file
    XTTS_CHECKPOINT = os.path.join(
        pretrainedPath, os.path.basename(XTTS_CHECKPOINT_LINK)
    )  # model.pth file
    XTTS_CONFIG_FILE = os.path.join(
        pretrainedPath, os.path.basename(XTTS_CONFIG_LINK)
    )  # config.json file

    # download XTTS v2.0 files if needed
    if not os.path.isfile(TOKENIZER_FILE):
        print(" > Downloading XTTS v2.0 tokenizer!")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK], pretrainedPath, progress_bar=True
        )
    if not os.path.isfile(XTTS_CHECKPOINT):
        print(" > Downloading XTTS v2.0 checkpoint!")
        ModelManager._download_model_files(
            [XTTS_CHECKPOINT_LINK], pretrainedPath, progress_bar=True
        )
    if not os.path.isfile(XTTS_CONFIG_FILE):
        print(" > Downloading XTTS v2.0 config!")
        ModelManager._download_model_files(
            [XTTS_CONFIG_LINK], pretrainedPath, progress_bar=True
        )

        # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=11025,  # 0.5 secs
        debug_loading_failures=False,
        max_wav_length=max_audio_length,  # ~11.6 seconds
        max_text_length=max_text_length,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config
    audio_config = XttsAudioConfig(
        sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000
    )
    # training parameters config

    config = GPTTrainerConfig()

    config.load_json(XTTS_CONFIG_FILE)

    config.epochs = num_epochs
    config.output_path = output_path
    config.model_args = model_args
    config.run_name = "xtts2gal"
    config.project_name = "xtts2gal"
    config.run_description = (
        """
        Gal gtp training
        """,
    )
    config.dashboard_logger = "tensorboard"
    config.logger_uri = None
    config.audio = audio_config
    config.batch_size = batch_size
    config.num_loader_workers = 8
    config.eval_split_max_size = 256
    config.print_step = 50
    config.plot_step = 100
    config.log_model_step = 100
    config.save_step = save_step
    config.save_n_checkpoints = 1
    config.save_checkpoints = True
    config.print_eval = False
    config.optimizer = "AdamW"
    config.optimizer_wd_only_on_weights = True  # cambiar si se usan multiGpu
    config.optimizer_params = {
        "betas": [0.9, 0.96],
        "eps": 1e-8,
        "weight_decay": weight_decay,
    }
    config.lr = lr
    config.lr_scheduler = "MultiStepLR"
    config.lr_scheduler_params = {
        "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
        "gamma": 0.5,
        "last_epoch": -1,
    }
    config.test_sentences = []

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        configList,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=grad_acumm,
        ),
        config,
        output_path=os.path.join(output_path, "run", "training"),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    # get the longest text audio file to use as speaker reference
    samples_len = [len(item["text"].split(" ")) for item in train_samples]
    longest_text_idx = samples_len.index(max(samples_len))
    speaker_ref = train_samples[longest_text_idx]["audio_file"]

    trainer_out_path = trainer.output_path

    # deallocate VRAM and RAM
    del model, trainer, train_samples, eval_samples
    gc.collect()

    return trainer_out_path


if __name__ == "__main__":
    train("", 3, 8, 4, "checkpoint/", 330750, 400, "5e-6", "1e-2", 50000)
