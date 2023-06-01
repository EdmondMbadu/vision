# Vision Assignment

To run the code cd to the directory and run:
`python3 cnn.py output.png convnet --lr .001 --batch_size 9 --epochs 30 --aug`

Note than you can change the network to `linear` or name change the output file name to anythnig you wish.
`--lr` is the learning rate which was set here to .001.
`--batch-size` can also be set to different size, as well as the epochs `--epochs` and whether you want to aument the
data which actually increases the performance slightly.

## Downlaod Data from Internet

The dataset will be downloaded directly once you run the code the first time, and will not in later attempts. Make sure
that this line is present for the get request `ssl._create_default_https_context = ssl._create_unverified_context`. It
is already included in code; so do not worry about it.

