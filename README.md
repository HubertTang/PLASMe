# PLASMe

[![DOI](https://zenodo.org/badge/578918028.svg)](https://zenodo.org/badge/latestdoi/578918028)

PLASMe is a tool to identify plasmid contigs from short-read assemblies using the Transformer. PLASMe capitalizes on the strength of alignment and learning-based methods. Closely related plasmids can be easily identified using the alignment component in PLASMe, while diverged plasmids can be predicted using order-specific Transformer models.

## Required Dependencies

* Python 3.x
* Pytorch
* diamond
* blast
* biopython
* numpy
* pandas

## Quick install (Linux only)

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

   > **Reminder:**
   >
   > 1. Lower versions of Anaconda may not be able to install PLASMe (some users have reported that Anaconda version 4.8.4 cannot install PLASMe). If you encounter a `PackagesNotFoundError`, please upgrade Anaconda to a newer version.
   >
   > 2. If you encounter the conda package conflicts issue during installation, please set the `channel_priority`  to `flexible`. The method to set it is as follows:
   >
   >    ```bash
   >    conda config --set channel_priority flexible
   >    ```


3. Download the reference database using `PLASMe_db.py`

   ```bash
   python PLASMe_db.py
   ```

   more optional arguments:

   --keep_zip: Keep the compressed database. Default: False

   --threads: The number of threads used to build the database. Default:  8

   > **Alternative 1:**
   >
   > Download the reference dataset (12.4GB) manually from [Zenodo](https://zenodo.org/record/8046934/files/DB.zip?download=1) ([OneDrive](https://portland-my.sharepoint.com/:u:/g/personal/xubotang2-c_my_cityu_edu_hk/ERYxOA6rEUVLpyWyyWRECWABFKb4F51IYmGlobFvH8GTLw?e=W2zl00)) to the same directory with `PLASMe.py`. (No need to uncompress it, PLASMe will extract the files and build the database the first time you use it. It will take several minutes.)
   >
   > **Alternative 2:**
   >
   > Download the reference dataset (12.4GB) manually from [Zenodo](https://zenodo.org/record/8046934/files/DB.zip?download=1) ([OneDrive](https://portland-my.sharepoint.com/:u:/g/personal/xubotang2-c_my_cityu_edu_hk/ERYxOA6rEUVLpyWyyWRECWABFKb4F51IYmGlobFvH8GTLw?e=W2zl00)) to the any directory and uncompress it, you will obtain a database folder named `DB`. When using `PLASMe.py`, use the `-d` option to specify the `DB`'s absolute path (not relative path).

## Usage

PLASMe requires input assembled contigs in Fasta format and outputs the predicted plasmid sequences in Fasta format.

```bash
python PLASMe.py [INPUT_CONTIG] [OUTPUT_PLASMIDS] [OPTIONS]
```

 more optional arguments:

   -d, --database: the database directory. (Use the absolute path to specify the location of the database. Default: PLASMe/DB)

   -c, --coverage: the minimum coverage of BLASTN. Default: 0.9.

   -i, --identity: the minimum identity of BLASTN. Default: 0.9.

   -p, --probability: the minimum probability of Transformer. Default: 0.5.

   -t, --thread: the number of threads. Default: 8.

   -u, --unified: Using unified Transformer model to predict  (default: False).

   -m, --mode: Using pre-set parameters (default: None). We have preset three sets of parameters for user convenience, namely `high-sensitivity`, `balance`, and `high-precision`. In `high-sensitivity` mode, the sensitivity is higher, but it may introduce false positives (identity threshold: 0.7, probability threshold: 0.5). In `high-precision` mode, the precision is higher, but it may introduce false negatives (identity threshold: 0.9, probability threshold: 0.9). In `balance` mode, there is a better balance between precision and sensitivity (identity threshold: 0.9, probability threshold: 0.5).

   --temp: the path of directory saving temporary files. Default: temp.

## Outputs

### Output files

| Files                        | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| <OUTPUT_PLASMIDS>            | Fasta file of all predicted plasmid contigs                  |
| <OUTPUT_PLASMIDS>_report.csv | Report file of the description of the identified plasmid contigs |

### Output report format

| Field      | Description                                           |
| ---------- | ----------------------------------------------------- |
| contig     | Sequence ID of the query contig                       |
| length     | Length of the query contig                            |
| reference  | The best-hit aligned reference plasmid                |
| order      | Assigned order                                        |
| evidence   | BLASTn or Transformer                                 |
| score      | The prediction score (applicable only to Transformer) |
| amb_region | The ambiguous regions*                                |

\* The ambiguous regions refer to regions that may be shared with the chromosomes. If a query contig contains a large proportion of ambiguous regions, caution must be exercised as it could potentially originate from a chromosome.

## Example

```bash
# run PLASMe using coverage of 0.6, identity of 0.6, probability of 0.5, and 8 threads to identify the palsmids.
python PLASMe.py test.fasta test.plasme.fna -c 0.6 -i 0.6 -p 0.5 -t 8
```

## Train the PC-based Transformer model using customized dataset

Considering that you may want to build protein cluster-based Transformer models from scratch, we provide `train_pc_model.py` to demonstrate how to train models using customized protein databases. It includes building the protein cluster database, converting query sequences into numerical vectors, training and evaluating models, and making predictions. To run this script, in addition to installing the required dependencies mentioned above, you will also need to install `mcl` using the following command:

```bash
conda install -c bioconda mcl
```

To achieve better results, we have the following recommendations:

1. The protein database should be as comprehensive as possible.
2. Setting stricter alignment thresholds when aligning query sequences to the PC database can further improve precision.
3. In classification tasks, PC clusters that lack discriminative power may introduce noise and reduce classification performance. Therefore, it is advisable to remove PC clusters that lack discriminative power.

## Supplementary data

We have uploaded the supplmentary data into [OneDrive](https://portland-my.sharepoint.com/:f:/g/personal/xubotang2-c_my_cityu_edu_hk/Es13c1PbeOtHi10FyeThOP8BCaJ3MyEMCNj33-GUby0DRw?e=wVFybc), including the PLSDB test set and real data. The detailed information can be found in `README.txt`.