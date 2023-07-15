import torch
import numpy as np

from datasets import load_dataset  # pip install datasets
from transformers import AutoTokenizer, AutoModel  # pip install transformers
from transformers import BertForSequenceClassification, BertConfig

print(torch.backends.mps.is_built()) # check if the environment supports mps

# remember to move to MPS with .to(device)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')


# load the first 1K rows of the TREC dataset
trec = load_dataset('trec', split='train[:1000]')

# take the first 64 rows of the trec data
text = trec['text'][:64]
# tokenize text using the BERT tokenizer
tokens = tokenizer(
    text, max_length=512,
    truncation=True, padding=True,
    return_tensors='pt'
)

device = torch.device('mps')
model.to(device)
tokens.to(device)
model(**tokens)


class TrecDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __getitem__(self, idx):
        input_ids = self.tokens[idx].ids
        attention_mask = self.tokens[idx].attention_mask
        labels = self.labels[idx]
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }

    def __len__(self):
        return len(self.labels)

labels = np.zeros(
    (len(trec), max(trec['coarse_label'])+1)
)
# one-hot encode
labels[np.arange(len(trec)), trec['coarse_label']] = 1
dataset = TrecDataset(tokens, labels)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=32
)


config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = max(trec['coarse_label'])+1  # create 6 outputs
modelt = BertForSequenceClassification(config).to(device)
# activate training mode of model
modelt.train()

# initialize adam optimizer
optim = torch.optim.Adam(modelt.parameters(), lr=5e-5)

# begin training loop
for batch in loader:
  	# note that we move everything to the MPS device
    batch_mps = {
        'input_ids': batch['input_ids'].to(device),
        'attention_mask': batch['attention_mask'].to(device),
        'labels': batch['labels'].float().to(device)
    }
    #torch.from_numpy(equalized_depth_array.astype(np.float32)).to(depth_tensor.device)
    # initialize calculated gradients (from prev step)
    optim.zero_grad()
    # train model on batch and return outputs (incl. loss)
    outputs = modelt(**batch_mps)
    # extract loss
    loss = outputs[0]
    # calculate loss for every parameter that needs grad update
    loss.backward()
    # update parameters
    optim.step()
