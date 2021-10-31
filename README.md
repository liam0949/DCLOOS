# Out-of-Scope Intent Detection with Self-Supervision and Discriminative Training (ACL 2021).
Please refer to [this link](https://github.com/fanolabs/out-of-scope-intent-detection) for the code.
The open-domain and CLINC150 datasets can be found at [this link](https://drive.google.com/drive/folders/1TqKIebKZfKIulCO9sA50uA01gbsJ_XU3?usp=sharing).

For training:

 ```nohup python main.py --dataset_pos oos --dataset_neg squad  --loss_ce_only --know_only --known_cls_ratio 0.75 --train_batch_size 200 --n_oos 200 --num_convex 400  --num_convex_val 200  --temp 0.1 --patient 100 --seed 888 --lr 1e-5 --num_train_epochs 1000 --datetime "20210401" --dl_large 1>oos.out 2>&1 &```

Note that the training procedure can stop earlier than 1000 epochs, pls set a smaller patient number. We borrow the dataloader from [ADB](https://github.com/thuiar/Adaptive-Decision-Boundary) and his work is also for OOS intent detection, check it out if you are interested.

Apologies for that there is still a large proportion of legacy codes in our collaborator's link, but these codes have been commented out.
