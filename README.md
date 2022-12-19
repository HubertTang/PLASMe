# PLASMe

PLASMe is a tool to identify plasmid contigs from short-read assemblies using the Transformer. PLASMe capitalizes on the strength of alignment and learning-based methods. Closely related plasmids can be easily identified using the alignment component in PLASMe, while diverged plasmids can be predicted using order-specific Transformer models.


# Required Dependencies

* Python 3.x
* Pytorch
* diamond
* blast
* biopython
* numpy
* pandas

# Quick install

1. Download PLASMe by "git clone"

```bash
git clone https://github.com/HubertTang/PLASMe.git
cd PLASMe
```

2. We recommend using `conda` to install all the dependencies.

```bash
# install the plasme
conda env create -f plasme.yaml
# activate the environment
conda activate plasme
```

3. Download the reference dataset (12.4GB) from [Google Drive](https://drive.google.com/file/d/1a7iKLI6NFUGHnGAd79wU_CoNvsG4OiBl/view?usp=sharing) (or [OneDrive](https://portland-my.sharepoint.com/:u:/g/personal/xubotang2-c_my_cityu_edu_hk/EW3nhkuiozpMhnkEuiafZhQBRAIsGzKL50RBQP1CSX6RXw?e=8kAGUj)) to the same directory with `PLASMe.py`. (No need to uncompress it, PLASMe will extract the files and build the database the first time you use it. It will take several minutes.)

# Usage

PLASMe requires input assembled contigs in Fasta format and outputs the predicted plasmid sequences in Fasta format.

```bash
python PLASMe.py [INPUT_CONTIG] [OUTPUT_PLASMIDS]
```

 more ptional arguments:

   -c, --coverage: the minimum coverage of BLASTN. Default: 0.9.

   -i, --identity: the minimum identity of BLASTN. Default: 0.9.

   -p, --probability: the minimum probability of Transformer. Default: 0.9.

   -t, --thread: the number of threads. Default: 8.

   --temp: the path of directory saving temporary files. Default: temp.


# Example

```bash
# run PLASMe using coverage of 0.6, identity of 0.6, probability of 0.5, and 8 threads to identify the palsmids.
python PLASMe.py test.fasta test.plasme.fna -c 0.6 -i 0.6 -p 0.5 -t 8
```
