import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from dataset import BilingualDataset
from model import build_transformer
from torch.utils.tensorboard import SummaryWriter
from config import get_config, get_weights_file_path
from tqdm import tqdm
import warnings

def get_all_sentences(ds, lang):
    # each item in the dataset is a pair of sentences, one in english and one in hindi and we are
    # extracting one language at a time
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) # if tokenizer sees unseen word, map to [UNK]
        tokenizer.pre_tokenizer = Whitespace()
        # trainer to build the tokenizer
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        # method that gives all the sentences from the dataset
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# loading dataset
def get_ds(config):
    ds_raw = load_dataset('Helsinki-NLP/opus_100', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size 
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Maximum length of source sentences: {max_len_src}")
    print(f"Maximum length of target sentences: {max_len_tgt}")

    train_datalaoder = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_datalaoder = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_datalaoder, val_datalaoder, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

# building the training loop

def train_model(config):
    # define the device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') # mps for apple silicon, rename to cuda if on windows
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_datalaoder, val_datalaoder, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard to visualise the loss
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload']) 
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()

        batch_iterator = tqdm(train_datalaoder, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # run the tensor through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})  

            # log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            # backprop the loss
            loss.backward()

            # update the weights
            optimizer.step()  
            optimizer.zero_grad()
            global_step += 1

        # save the model

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_filename)

if __name__ == "__main__":  
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)