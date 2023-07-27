
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import types
import torch.nn.functional as F
from torch import nn
from transformers import GPT2LMHeadModel

def get_model( self ):

    if self.config.model == 'MNIST':
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(28*28, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, 10)
                self.loss_function = nn.CrossEntropyLoss()

            def forward(self, inputs, labels ):
                out = types.SimpleNamespace()
                inputs = inputs.view(-1, 28*28)
                inputs = F.relu(self.fc1(inputs))
                inputs = F.relu(self.fc2(inputs))
                out.outputs = self.fc3(inputs)
                out.loss = self.loss_function(out.outputs, labels)
                return out

        return Net()
    
    elif self.config.model == 'GPT2':

        # Return GPT2 model.
        return GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id = self.dataset.tokenizer.eos_token_id)
    
    else:
        raise ValueError(f'No known model for {self.config.model}')

