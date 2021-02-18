import os
import spacy
from PIL import Image
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms


spacy_eng = spacy.load('en_core_web_sm')

class Vocabulary():
    """
        freq_threshold is the threshold for the words to be kept if
        their world count frequency is greater than or equal to the freq_threshold
    """
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.stoi = {"<PAD>" : 0,
                    "<SOS>" : 1,
                    "<EOS>" : 2,
                    "<UNK>" : 3}
        self.itos = {0:"<PAD>",
                    1 :"<SOS>",
                    2 : "<EOS>",
                    3 : "<UNK>"}
        
    def __len__(self):
        return len(self.itos)
      
    @staticmethod
    def tokenizer_english(text):
        tokens = []
        for word in spacy_eng.tokenizer(text):
            tokens.append(word.text.lower())
        return tokens
    
    def build_vocab(self, sentence_list):
        idx = 4
        frequencies = {}
        
        for sentence in sentence_list:
            for token in self.tokenizer_english(sentence):
                if token not in frequencies:
                    frequencies[token] = 0
                else :
                    frequencies[token]+=1
                if frequencies[token] == self.freq_threshold:
                    self.stoi[token] = idx
                    self.itos[idx] = token
                    idx+=1
                
    def numericalize(self, text):
        numeric_text = []
        for token in self.tokenizer_english(text):
            if token in self.stoi:
                numeric_text.append(self.stoi[token])
            else:
                numeric_text.append(self.stoi["<UNK>"])
        return numeric_text
    
class AddPaddingCollate:
  
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self,batch):
#         images, captions = batch[0],  batch[1]
#         all_images = [im.unsqueeze(0) for im in images]
#         for item in batch:
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)

        captions = [item[1] for item in batch]
            
#         print(images.shape)
#         print(captions.shape)

#         all_images = all_iamges
#         all_images = torch.cat(images, dim=0)
        captions = pad_sequence(captions, batch_first=False, padding_value=self.pad_idx)
        return images, captions
      
class FlickrDataset(Dataset):
    def __init__(self, root_dir, caption_file,transform=None,freq_threshold=5) :
        self.root_dir = root_dir
        self.caption_df = pd.read_csv(caption_file)
        self.transform = transform
        self.images = self.caption_df['image']
        self.captions = self.caption_df['caption']
        
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())
        
    def __len__(self):
        return len(self.caption_df)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        image_id = self.images[index]
        image_path = os.path.join(self.root_dir, image_id)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        token_caption = [self.vocab.stoi["<SOS>"]]
        token_caption +=  self.vocab.numericalize(caption)
        token_caption.append(self.vocab.stoi["<EOS>"])
        
        return image, torch.tensor(token_caption)
      
      
def get_loader(root_folder, caption_file, transform=None, batch_size=32, num_workers=8, shuffle=True, pin_memory=True):
    dataset = FlickrDataset(root_folder, caption_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(dataset, 
                        shuffle=shuffle, 
                        num_workers=num_workers, 
                        batch_size=batch_size,
                        pin_memory=pin_memory,
                        collate_fn=AddPaddingCollate(pad_idx=pad_idx)
                       )
    return loader, dataset
      
def imageCaptionDataset():
    image_folder = "./flickr8k/images"
    csv = "./flickr8k/captions.txt"
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),transforms.ToTensor()]
    )
    loader, dataset = get_loader(image_folder, csv, transform=transform)
    return loader, dataset
  
if __name__ == "__main__"  :
    loader, dataset = imageCaptionDataset()
    for x, y in loader:
        print(x.shape)
        print(y.shape)
        break