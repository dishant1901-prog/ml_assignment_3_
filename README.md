# Machine Learning Assignment 3: MLPs, CNNs, autoencoders, SVMs and clustering on CIFAR-10

Dishant Kaushik, M24MA2003

Everything in the report can be regenerated from this repository. All randomness is seeded with 242003
(the digits of the roll number), and the train/val split is saved in `common/split_seed242003.npz`.

## Layout
```
common/      data loading, the saved split, training loop, MLP/CNN definitions, job runner
configs/     final configuration of every experiment (hyperparameters, search grids, values selected on val)
Q1/ Q2/ Q3/ Q4/
   *.py        experiment scripts
   notes.md    design decisions and results for that question
   results/    json results (every number in the report comes from these files)
   logs/       training logs of the final runs
   figures/    figures used in the report (made by plots.py)
jobs/        the job list used to run everything one after another
report/      the compiled report (PDF) and its LaTeX source (CVPR template)
requirements.txt
```
Checkpoints (`*/checkpoints/`) and the dataset (`data/`) are not in the repository because of their size;
they are recreated by the commands below.

## Environment
Python 3.13, CUDA 12.6 build of PyTorch. Tested on an RTX 4050 laptop GPU (6 GB), Windows 11.
```
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Data and split
All results use the CIFAR-10 parquet release on Hugging Face (`uoft-cs/cifar10`, revision
`0b2714987fa478483af9968de7c934580d0bb9a2`), because the official Toronto server was too slow to download from.
It contains the same images as the official python batches (identical per-channel statistics, test set in the
official order; check in `logs/data_check.log`), but the 50,000 training images are stored in a different
order. The saved split holds indices into this order, so build the data from the parquet files:
```
mkdir -p data/hf
REV=0b2714987fa478483af9968de7c934580d0bb9a2
curl -L -o data/hf/train-00000-of-00001.parquet https://huggingface.co/datasets/uoft-cs/cifar10/resolve/$REV/plain_text/train-00000-of-00001.parquet
curl -L -o data/hf/test-00000-of-00001.parquet  https://huggingface.co/datasets/uoft-cs/cifar10/resolve/$REV/plain_text/test-00000-of-00001.parquet
python -m common.prepare_data          # -> data/cifar10.npz  (md5 442c33db77f0c69c5a0934f02f4a93e6)
python -m common.make_split            # -> common/split_seed242003.npz (already in the repo, md5 0204a12a1885b4c07b244978623e8adc)
```
md5 of the parquet files: train `7170dcdaf6355deedaa01d9915ae0d31`, test `a52165db3ae649545cf55cf1791cd12d`.

## Reproduction commands
Run from the repository root, in this order. Training scripts skip runs whose result file already exists.

Q1
```
python -m Q1.q1a_train                       # MLP, wide MLP, MLP+aug, CNN, lr grid {0.01,0.03,0.1,0.3,1.0}
python -m Q1.q1a_select_test                 # lr chosen on val, then one test evaluation per model
python -m Q1.q1b_memorisation --noise 0.2 0.0 0.4
python -m Q1.plots
```
Q2 (the clean condition reuses the Q1(a) runs)
```
python -m Q2.q2_train --condition perm   --model mlp cnn
python -m Q2.q2_train --condition patch8 --model mlp cnn
python -m Q2.q2_train --condition patch4  --model mlp
python -m Q2.q2_train --condition patch16 --model mlp
python -m Q2.q2_train --condition patch4  --model cnn --lr-from patch8
python -m Q2.q2_train --condition patch16 --model cnn --lr-from patch8
python -m Q2.q2_eval
python -m Q2.plots
```
Q3
```
python -m Q3.q3_train_ae --bottleneck 16 128 1024
python -m Q3.q3_train_ae --bottleneck 1024 --noise 0.1
python -m Q3.q3_eval --noise 0.1             # needs the Q1(a) CNN checkpoint
python -m Q3.plots
```
Q4
```
python -m Q4.q4_svm features --encoder ae_d1024_noise0.1
python -m Q4.q4_svm sweep
python -m Q4.q4_svm test
python -m Q4.q4_analysis pairs
python -m Q4.q4_analysis cluster-select
python -m Q4.q4_analysis cluster-test
python -m Q4.q4_analysis compare
python -m Q4.plots
```
The same steps are listed in `jobs/night.txt` and can be run unattended with
`python -m common.run_queue jobs/night.txt`, which runs one job at a time and pauses or waits when the
machine's CPU or memory use gets too high (limits in `common/run_queue.py`).

## Report
The compiled report is `report/ML_Assignment3_M24MA2003_Dishant_Kaushik.pdf`.
`report/` holds the LaTeX source (CVPR template). Run `python report/collect_figures.py` to copy the figures
into `report/figures/`, then compile `report/main.tex` with pdfLaTeX + BibTeX (for example on Overleaf by
uploading the `report/` folder).
