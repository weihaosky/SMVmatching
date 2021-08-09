#!/bin/bash

# ==== model without probability ===
python unsuperlearn.py --model=stackhourglass --name=stage1 --batch_size=4 --epoch=500


# ==== probability model, uncertainty estimation ===
python unsuperlearn_prob_eval.py --model stackhourglass --name=stage2 --batch_size=4 --epoch=1000   \
                                 --resume=500 --resume_model=log/stage1/model_stage1_best.pth