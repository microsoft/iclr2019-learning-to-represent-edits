# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np


def evaluate_nll(model, test_set, batch_size=32, return_nll_list=False):
    was_training = model.training
    model.eval()

    cum_nll = 0.
    cum_ppl = 0.
    cum_examples = 0.

    nll_dict = dict()

    with torch.no_grad():
        for batch_examples in test_set.batch_iter(batch_size):
            log_probs = -model(batch_examples)
            batch_code_tokens_num = torch.tensor([len(e.updated_code_chunk) for e in batch_examples],
                                                 dtype=torch.float,
                                                 device=log_probs.device)

            batch_nlls = log_probs.cpu().numpy()
            batch_ppls = (log_probs / batch_code_tokens_num).cpu().numpy()
            for batch_id in range(len(batch_examples)):
                nll_dict[batch_examples[batch_id].id] = batch_nlls[batch_id]

            cum_ppl += batch_ppls.sum()
            cum_nll += batch_nlls.sum()
            cum_examples += len(batch_examples)

            del log_probs

    avg_ppl = np.exp(cum_ppl / cum_examples)
    avg_nll = cum_nll / cum_examples

    if was_training:
        model.train(was_training)

    if return_nll_list:
        return avg_nll, avg_ppl, nll_dict
    else:
        return avg_nll, avg_ppl
