import torch

# Source: https://github.com/pratyushmaini/localizing-memorization/blob/b09ba96e412c03301ddd0b944b511495d863c60b/models/dropout.py#L37
class ExampleTiedDropout(torch.nn.Module):
    # this class is similar to batch tied dropout,
    # but instead of tying neurons in a batch, we tie neurons in a set of examples

    def __init__(self, p_fixed = 0.2, p_mem = 0.1, num_batches = 100, drop_mode = "train"):
        super(ExampleTiedDropout, self).__init__()
        self.max_id = 60000
        self.p_mem = p_mem
        self.p_fixed = p_fixed
        self.drop_mode = drop_mode
        self.mask_tensor = None

    def forward(self, X, idx):
        if self.p_fixed == 1:
            return X

        # this changes based on whether model.train() or model.eval() is called, better than a drop mode for scripting
        if self.training:#self.drop_mode == "train":
            # create a mask based on the index (idx)

            mask = torch.zeros_like(X).cpu()
            shape = X.shape[1]

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
                    #repeat curr_mask along dimension 2 and 3 to have the same shape as X
                    # curr_mask = curr_mask.unsqueeze(-1).unsqueeze(-1) # don't need this for token sequences
                    mask[i][int(self.p_fixed*shape):] = curr_mask

                if self.mask_tensor is None:
                    self.mask_tensor = torch.zeros(self.max_id, X.shape[1], X.shape[2], X.shape[3])
                #assign mask at positions given by idx
                self.mask_tensor[idx] = mask

            # Apply the mask to the input tensor
            X = X * mask.to(X.device)


        # elif self.drop_mode == "test":
        #     #At test time we will renormalize outputs from the non-fixed neurons based on the number of neuron sets
        #     #we will keep the fixed neurons unmodified
        #     shape = X.shape[1]
        #     X[:, :int(self.p_fixed*shape)] = X[:, :int(self.p_fixed*shape)]
        #     X[:, int(self.p_fixed*shape):] = X[:, int(self.p_fixed*shape):]*self.p_mem

        else:# self.drop_mode == "drop":
            shape = X.shape[1]
            X[:, int(self.p_fixed*shape):] = 0
            X[:, :int(self.p_fixed*shape)] = X[:, :int(self.p_fixed*shape)]*(self.p_fixed + self.p_mem)/self.p_fixed

        return X