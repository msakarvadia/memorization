import torch

# Source: https://github.com/pratyushmaini/localizing-memorization/blob/b09ba96e412c03301ddd0b944b511495d863c60b/models/dropout.py#L37
class ExampleTiedDropout(torch.nn.Module):
    # this class is similar to batch tied dropout,
    # but instead of tying neurons in a batch, we tie neurons in a set of examples

    def __init__(self, p_fixed = 0.2, p_mem = 0.1, num_batches = 100, drop_mode = "train"):
        super(ExampleTiedDropout, self).__init__()
        self.max_id = 100000 # change based on number of examples in train data
        self.p_mem = p_mem
        self.p_fixed = p_fixed
        self.drop_mode = drop_mode
        self.mask_tensor = None

    def forward(self, X, idx):
        if self.p_fixed == 1:
            return X
        
        # idx is a tensor of size [batch_size] that uniquely IDs each training example with an integer index
        # to see how this is done, see the `IndexedDataset` class
        if self.training and torch.any(idx >= self.max_id):
            # this implementation stores all the masks in memory, so
            # theres a necessary upper limit to the number of training examples that has to be 
            # enforced here - we could try to get around this with some fancy programming
            # but for research, this should be fine - just change max_id if you need more and make sure
            # you have the memory necessary to hold such a large tensor
            # you might also try to save memory by reducing the hidden dim or the context length to reduce the
            # mask tensor size
            raise RuntimeError(f"Example Index {idx} is invalid, expected idx < {self.max_id}")

        # this changes based on whether model.train() or model.eval() is called, better than a drop mode for scripting
        if self.training:
            # create a mask based on the index (idx)

            mask = torch.zeros_like(X).cpu()
            shape = X.shape[1] # ctx_len

            if self.mask_tensor is not None:
                #get mask from self.mask_tensor
                mask = self.mask_tensor[idx]

            else:
                #keep all neurons with index less than self.p_fixed*shape
                mask[:, :int(self.p_fixed*shape)] = 1

                # Fraction of elements to keep
                p_mem = self.p_mem

                # Generate a random mask for each row in the input tensor
                shape_of_mask = shape - int(self.p_fixed*shape)
                for i in range(X.shape[0]):
                    torch.manual_seed(idx[i].item())
                    curr_mask = torch.bernoulli(torch.full((1, shape_of_mask), p_mem))
                    for j in range(shape_of_mask):
                        mask_ind = j + int(self.p_fixed*shape)
                        mask_val = 0 if curr_mask[0][j] == 0 else 1
                        mask[i][mask_ind] = mask_val

                if self.mask_tensor is None:
                    self.mask_tensor = torch.zeros(self.max_id, X.shape[1], X.shape[2])
                #assign mask at positions given by idx
                self.mask_tensor[idx] = mask

            # Apply the mask to the input tensor
            X = X * mask.to(X.device)

        else:
            shape = X.shape[1]
            X[:, int(self.p_fixed*shape):, :] = 0
            X[:, :int(self.p_fixed*shape), :] = X[:, :int(self.p_fixed*shape), :]*(self.p_fixed + self.p_mem)/self.p_fixed

        return X