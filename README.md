Unsupervised Learning for Physical Interaction through Video Prediction
==============================

Based on the paper from C. Finn, I. Goodfellow and S. Levine: "Unsupervised Learning for Physical Interaction through Video Prediction". Implemented in Pytorch.

Creating the data need for training
------------
```bash 
$ sh download_data.sh push_datafiles.txt # Will download all the data from Google's ftp to data/raw
$ python ./tfrecord_to_dataset.py
```

Running the training process
------------
```bash
$ python ./train.py \
  --data_dir data/processed/push/push_train \ # path to the training set.
  --model CDNA \ # the model type to use - DNA, CDNA, or STP
  --output_dir ./weights \ # where to save model checkpoints
  --pretrained_model model \ # path to model to initialize from, random if emtpy
  --sequence_length 10 \ # the number of total frames in a sequence
  --context_frames 2 \ # the number of ground truth frames to pass in at start
  --num_masks 10 \ # the number of transformations and corresponding masks
  --schedsamp_k 900.0 \ # the constant used for scheduled sampling or -1
  --train_val_split 0.95 \ # the percentage of training data for validation
  --batch_size 32 \ # the training batch size
  --learning_rate 0.001 \ # the initial learning rate for the Adam optimizer
  --epochs 10 \ # total training epoch
  --print_interval 10 \ # iterations to output loss
  --device cuda \ # the device used for training
  --use_state \ # whether or not to condition on actions and the initial state
```
