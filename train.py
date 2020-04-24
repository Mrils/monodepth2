# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()

if __name__ == "__main__":
    if opts.multi_gpu:
        gpu_devices = ','.join([str(id) for id in opts.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        os.environ['MASTER_ADDR'] = '127.0.0.1'              #
        os.environ['MASTER_PORT'] = '23456' 
        ngpus_per_node = torch.cuda.device_count()
        opts.world_size = ngpus_per_node * opts.world_size
        mp.spawn(Trainer,
        args=(ngpus_per_node, opts),
        nprocs=ngpus_per_node)
    else:
        trainer = Trainer(1, self.ngpus_per_node, opts)
        trainer.train()
