import argparse
from Bio import SeqIO
import bio_script
import trans_model
import trans_data
import numpy as np
import os
import pandas as pd
import pickle as pkl
import subprocess
import shutil
import  torch
from    torch import nn
from    torch import optim
import  torch.utils.data as Data


def plasme_cmd():
    parser = argparse.ArgumentParser(description="PLASMe arguments.")

    # argument for dataset
    parser.add_argument(
        'input',
        type=str,
        help="Path of the input file."
    )

    # argument for dataset
    parser.add_argument(
        'output',
        type=str,
        help="Directory of the output files."
    )

    # parser.add_argument(
    #     '-d', "--database",
    #     type=str,
    #     default="DB",
    #     help="The default database."
    #     )

    parser.add_argument(
        '-c', "--coverage",
        default=0.9,
        type=float,
        help="The minimun coverage of BLASTN (default: 0.9).")
    
    parser.add_argument(
        '-i', "--identity",
        default=0.9,
        type=float,
        help="The minimun identity of BLASTN (default: 0.9).")
    
    parser.add_argument(
        '-p', "--probability",
        default=0.5,
        type=float,
        help="The minimun predicted probability of Transformer  (default: 0.5).")
    
    parser.add_argument(
        '-t', "--thread",
        default=8,
        type=int,
        help="The number of threads  (default: 8).")

    parser.add_argument(
        '-u', "--unified",
        default=False,
        type=bool,
        help="Using unified Transformer model to predict  (default: False).")

    parser.add_argument(
        '-m', "--mode",
        default=None,
        type=str,
        help="Using pre-set parameters (default: None).")

    parser.add_argument(
        "--temp",
        type=str,
        default=None,
        help="The temporary directory (default: None)."
        )

    parser.add_argument(
        "--taxonomy",
        type=str,
        default=None,
        help="Specify a specific taxonomy in order level (default: None)."
        )

    # version
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='PLASMe_v1.1'
    )

    plasme_args = parser.parse_args()

    assert plasme_args.mode in ['high-precision', 'balance', 'high-sensitivity', None], 'Unknown mode'

    return plasme_args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LAYER = 1


def reset_model(vocab_size, pad_idx, max_len):
    model = trans_model.Transformer(
                src_vocab_size=vocab_size, 
                src_pad_idx=pad_idx, 
                num_layers=NUM_LAYER,
                device=device, 
                max_length=max_len, 
                dropout=0.1
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.BCEWithLogitsLoss()
    return model, optimizer, loss_func


def return_batch(train_sentence, label, flag):
    # X_train = torch.from_numpy(train_sentence).to(device)
    # y_train = torch.from_numpy(label).float().to(device)
    X_train = torch.from_numpy(train_sentence)
    y_train = torch.from_numpy(label).float()
    train_dataset = Data.TensorDataset(X_train, y_train)
    training_loader = Data.DataLoader(
        dataset=train_dataset,    
        batch_size=512,
        shuffle=flag,               
        num_workers=0,              
    )
    return training_loader


def return_tensor(var, device):
    return torch.from_numpy(var).to(device)


# test the performance of the model
def test(data_loader, model):
    model.eval()
    all_pred = []
    all_logit = []

    for (batch_x, _) in data_loader:
        sentense = batch_x.to(device).to(torch.int64)

        with torch.no_grad():
            logit = model(sentense)
            logit  = torch.sigmoid(logit.squeeze(1)).cpu().detach().numpy()
            pred  = [1 if item > 0.5 else 0 for item in logit]
            all_pred += pred
            all_logit += [i for i in logit]

    return all_pred, all_logit


def find_ranges(lst):
    from itertools import groupby
    pos = (j - i for i, j in enumerate(lst))
    t = 0
    for i, els in groupby(pos):
        l = len(list(els))
        if l > 1:
            el = lst[t]
            t += l
            yield (el, el+l-1)


def predict(contig_path, ref_plas_db_path, ref_tax_path, temp_dir, out_path, ref_ol_path="plas_overlap.csv",
            db_dir='DB', min_cov=0.15, num_threads=8, use_unified=False, taxonomy=None):
    """Identify the plasmids from the contig data.
    """
    ### create the temporary directory for saving the temporary results
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    ### Use BLAST to align the testing data to reference plasmids
    # check if the BLAST database exists
    # if os.path.exists(f"{ref_plas_db_path}.nin"):
    #     subprocess.call(f"blastn -query {contig_path} -db {ref_plas_db_path} -num_threads {num_threads} -out {temp_dir}/blastn.csv -outfmt 6", shell=True)
    # else:
    #     subprocess.call(f"makeblastdb -in {ref_plas_db_path} -dbtype nucl -out {ref_plas_db_path}", shell=True)
    print(f"Align the contigs to the reference plasmids ... ...")
    subprocess.call(f"blastn -query {contig_path} -db {ref_plas_db_path} -num_threads {num_threads} -out {temp_dir}/blastn.csv -outfmt 6", shell=True)
    
    ### assign the taxonomy orders of contigs, and see if they are in the overlapping area
    # load the taxonomy for each reference
    all_order_list = ['Enterobacterales', 'Lactobacillales', 'Bacillales', 'Pseudomonadales', 
                      'Rhodobacterales', 'Hyphomicrobiales', 'Spirochaetales', 'Corynebacteriales', 
                      'Burkholderiales', 'Xanthomonadales', 'Campylobacterales', 'Thiotrichales', 
                      'Vibrionales', 'Aeromonadales', 'Sphingomonadales', 'Eubacteriales', 
                      'Rhodospirillales', 'Micrococcales', 'Streptomycetales', 'Nostocales', 
                      'Pasteurellales', 'Neisseriales', 'Bacteroidales', 'Legionellales', 
                      'Synechococcales', 'Cytophagales', 'Mycoplasmatales', 'Alteromonadales', 
                      'Chlamydiales', 'Chroococcales', 'Flavobacteriales', 'Thermales', 
                      'Entomoplasmatales', 'Deinococcales', 'other']

    ref_tax_dict = {}
    with open(ref_tax_path) as ctp:
        for l in ctp:
            l = l.strip().split()
            seq_id = l[0]
            taxon = l[1]
            if taxon not in all_order_list:
                taxon = 'other'
            ref_tax_dict[seq_id] = taxon

    # load the dictionary of query length
    query_len_dict = {}
    for s in SeqIO.parse(contig_path, 'fasta'):
        query_len_dict[s.id] = len(s.seq)

    # load the reference overlapped regions from plasmids
    ref_ol_dict = {}
    with open(ref_ol_path) as rop:
        for l in rop:
            l = l.strip().split()
            ref_id = l[0]
            ref_ol_list = [(int(i.split('-')[0]), int(i.split('-')[1])) for i in l[1: ]]
            ref_ol_dict[ref_id] = ref_ol_list

    # load the dictionary of order and queries
    order_query_dict = {o: {} for o in all_order_list}

    query_set = set()
    with open(f"{temp_dir}/blastn.csv") as blast_rst:
        for l in blast_rst:
            record = l.strip().split()
            query = record[0]
            ref = record[1]
            query_start = min(int(record[6]), int(record[7])) - 1
            query_end = max(int(record[6]), int(record[7]))
            ref_start = min(int(record[8]), int(record[9])) - 1
            ref_end = max(int(record[8]), int(record[9]))
            ident = float(record[2])/ 100
            aln_len = int(record[3])
            query_cov = aln_len/ query_len_dict[query]
            
            # assign the order of the aligned reference
            if ref in ref_tax_dict:
                order = ref_tax_dict[ref]
                if order not in all_order_list:
                    order = 'other'
            else:
                order = 'other'

            # only use the best hit
            if query in query_set:
                continue
            else:
                query_set.add(query)

            # filter the queries whose coverages are smaller than 0.15
            if query_cov <= min_cov:
                continue

            # load the overlapped regions in the query plasmids
            ol_set = set()
            try:
                # some references don't contain overlapped regions
                ol_region = ref_ol_dict[ref]
                for ol in ol_region:
                    ol_set.update([i for i in range(ol[0], ol[1])])
            except KeyError:
                pass

            # count the start and end positions in the queries
            query_aln_set = set([i for i in range(ref_start, ref_end)])
            inter_list = list(set.intersection(query_aln_set, ol_set))
            
            query_aln_regions_out = "overlap regions: "
            if len(inter_list) > 0:
                ranges_list = list(find_ranges(inter_list))
                if len(ranges_list) > 0:
                    for (ref_s, ref_e) in ranges_list:
                        query_ov_start = query_start + ref_s - ref_start
                        query_ov_end = query_start + ref_e - ref_start
                        query_aln_regions_out += f"{query_ov_start}-{query_ov_end} "
            else:
                pass

            # assign orders for queries
            order_query_dict[order][query] = (ref, ident, query_cov, query_aln_regions_out)

    ### run PC transformer in the non-overlapped region
    ## save the non-overlapped contigs and extract the proteins
    print(f"{len(query_set)} contigs are aligned to the reference database.")
    aln_seq_list = []
    for s in SeqIO.parse(contig_path, 'fasta'):
        if s.id in query_set:
            aln_seq_list.append(s)
    SeqIO.write(aln_seq_list, f"{temp_dir}/align.fna", 'fasta')

    print(f"Predict the proteins in contigs ... ...")
    bio_script.run_multi_prodigal(contig_path=f"{temp_dir}/align.fna", 
                       threads=num_threads)

    bio_script.count_aa(aa_fasta=f"{temp_dir}/align.fna.aa")

    # run alignment
    p2a_plsdb_mar30_db = f"{db_dir}/plsdb_Mar30.dmnd"
    # p2a_refseq_may05_db = "/home/xubotang2/2020_work/plasmid/Deeplasmid_train/ncbi_refseq_plasmid/refseq_plas.dmnd"
    print(f"Align the proteins to PC database ... ...")
    bio_script.run_diamond(db_path=p2a_plsdb_mar30_db, 
                query_path=f"{temp_dir}/align.fna.aa",
                threads=num_threads)

    ## generate the data for the transformer
    p2a_plsdb_mar30 = f"{db_dir}/plsdb_Mar30.clusters.p2a"
    trans_data.contig2sentance(p2a_path=p2a_plsdb_mar30, 
                               blast_path=f"{temp_dir}/align.fna.aa.diamond", 
                               aa_count_path=f"{temp_dir}/align.fna.aa.aa_count", 
                               test_aa_path=f"{temp_dir}/align.fna.aa", 
                               pc_thres=f"{db_dir}/plas_chrom_thres.csv",
                               out_dir=temp_dir, feat_len=400)
    
    ## load the model and run the transformer
    op = open(out_path, 'w')
    op.write(f"order,query,identity,coverage,PLASMe,overlap\n")
    pcs2idx = pkl.load(open(f'{temp_dir}/pc2wordsid.dict', 'rb'))
    num_pcs = len(set(pcs2idx.keys()))
    src_vocab_size = num_pcs+2
    src_pad_idx = 0
    
    # print("Loading the test data ...")
    print(f"Predict using PLASMe ... ...")
    test_feat = pkl.load(open(f'{temp_dir}/sentence.feat', 'rb'))
    test_seq_list = []
    with open(f"{temp_dir}/sentence_id.list") as sent_list:
        for l in sent_list:
            test_seq_list.append(l.strip())
    seq_feat_dict = {}
    for s, f in zip(test_seq_list, test_feat):
        seq_feat_dict[s] = f
    
    # predict the plasmids for each order
    for order, seqs in order_query_dict.items():
        temp_feat = []
        temp_seq_list = []
        if len(seqs) > 0:
            for seq in seqs:
                if seq in seq_feat_dict:
                    temp_feat.append(seq_feat_dict[seq])
                    temp_seq_list.append(seq)

            temp_label = np.ones((len(temp_feat)))
            test_loader = return_batch(np.array(temp_feat), temp_label, flag=False)

            model, _, _ = reset_model(vocab_size=src_vocab_size, 
                                    pad_idx=src_pad_idx,
                                    max_len=400)

            if use_unified:
                model_path = f"{db_dir}/trans_model/unified.pt"
            elif taxonomy:
                model_path = f"{db_dir}/trans_model/{taxonomy}.pt"
            else:
                model_path = f"{db_dir}/trans_model/{order}.pt"
            
            model.load_state_dict(torch.load(model_path, map_location=device))
            y_pred, y_logit = test(test_loader, model)
            
            for s, pred in zip(temp_seq_list, y_logit):
                op.write(f"{order},{s},{seqs[s][1]},{seqs[s][2]},{pred},{seqs[s][3]}\n")

            # output the results from Blast only
            for seq in seqs:
                if seq not in seq_feat_dict:
                    op.write(f"{order},{seq},{seqs[seq][1]},{seqs[seq][2]},-1,{seqs[seq][3]}\n")

    op.close()


def build_db(db_dir, num_threads=8):
    """Build the database.
    """
    # unzip plasmid files
    print("Unzip the reference sequences ... ...")
    shutil.unpack_archive(f"{db_dir}/plsdb.zip", db_dir)
    os.remove(f"{db_dir}/plsdb.zip")

    # build BLASTN and DIAMOND database
    print("Build DIAMOND and BLAST database ... ...")
    subprocess.run(f"diamond makedb --in {db_dir}/plsdb_Mar30.fna.aa -d {db_dir}/plsdb_Mar30 -p {num_threads}", shell=True)
    subprocess.run(f"makeblastdb -in {db_dir}/plsdb_Mar30.fna -dbtype nucl -out {db_dir}/plsdb_Mar30", shell=True)
    
    os.remove(f"{db_dir}/plsdb_Mar30.fna")
    os.remove(f"{db_dir}/plsdb_Mar30.fna.aa")


# def plasme_output(rst_path, contig_path, ident_thres, cov_thres, pred_thres, output_path):
#     """Generate the PLASMe output files.
#     """

#     rst_df = pd.read_csv(rst_path, sep=',')
#     pred_plasmid = []
#     pred_plasmid_overlap_dict = {}
#     for order, contig, ident_v, cov_v, pred_v, overlap in zip(rst_df['order'], rst_df['query'], 
#                                                     rst_df['identity'], rst_df['coverage'], 
#                                                     rst_df['PLASMe'], rst_df['overlap']):
#         if ident_v >= ident_thres and cov_v >= cov_thres:
#             pred_plasmid.append(contig)
#         else:
#             if pred_v > pred_thres:
#                 pred_plasmid.append(contig)
#         pred_plasmid_overlap_dict[contig] = overlap

#     output_seqs = []
#     for s in SeqIO.parse(contig_path, 'fasta'):
#         if s.id in pred_plasmid:
#             s.description = pred_plasmid_overlap_dict[s.id]
#             output_seqs.append(s)

#     SeqIO.write(output_seqs, output_path, 'fasta')

#     print(f"Finished. The identified plasmid contigs are saved in {output_path}.")


def merge_ranges(ranges):
    ranges.sort(key=lambda x: x[0])

    merged = [ranges[0]]
    for current_range in ranges[1:]:
        last_range = merged[-1]
        if current_range[0]-1 <= last_range[1]:
            merged[-1] = (last_range[0], max(last_range[1], current_range[1]))
        else:
            merged.append(current_range)
    return merged


def plasme_output(rst_path, blastn_rst, contig_path, ident_thres, cov_thres, pred_thres, output_path):
    """Generate the PLASMe output files.
    """

    rst_df = pd.read_csv(rst_path, sep=',')
    pred_plasmid = []
    pred_plasmid_overlap_dict = {}
    contig_seq_index = SeqIO.index(contig_path, 'fasta')
    out_info = open(f"{output_path}_report.csv", 'w')

    # load blastn results
    query_set = set()
    query_ref_dict = {}
    with open(blastn_rst) as b_rst:
        for l in b_rst:
            l = l.strip().split()
            query, ref = l[0], l[1]
            if query not in query_set:
                query_ref_dict[query] = ref
                query_set.add(query)

    out_info.write(f"contig\tlength\treference\torder\tevidence\tscore\tamb_region\n")

    for order, contig, ident_v, cov_v, pred_v, overlap in zip(rst_df['order'], rst_df['query'], 
                                                    rst_df['identity'], rst_df['coverage'], 
                                                    rst_df['PLASMe'], rst_df['overlap']):
        if ident_v >= ident_thres and cov_v >= cov_thres:
            pred_plasmid.append(contig)
        else:
            if pred_v > pred_thres:
                pred_plasmid.append(contig)

        # parse the overlap
        overlap_info = overlap.split(' ')
        overlap_site_list = []
        for site in overlap_info:
            if '-' in site:
                overlap_site_list.append([int(i) for i in site.split('-')])
        # if len(overlap_site_list) > 0:
        #     pred_plasmid_overlap_dict[contig] = merge_ranges(overlap_site_list)
        # else:
        #     pred_plasmid_overlap_dict[contig] = []

        evidence, score = '', ''
        if pred_v > 0:
            evidence = 'Transformer'
            score = pred_v
        else:
            evidence = 'BLASTn'
            score = 'nan'

        out_m = []
        if len(overlap_site_list) > 0:
            m = merge_ranges(overlap_site_list)
            for m_e in m:
                out_m.append('-'.join([str(i) for i in m_e]))

        if contig in pred_plasmid:
            out_info.write(f"{contig}\t{len(contig_seq_index[contig].seq)}\t{query_ref_dict[contig]}\t{order}\t{evidence}\t{score}\t{','.join(out_m)}\n")

    output_seqs = []
    for s in SeqIO.parse(contig_path, 'fasta'):
        if s.id in pred_plasmid:
            # s.description = pred_plasmid_overlap_dict[s.id]
            output_seqs.append(s)

    SeqIO.write(output_seqs, output_path, 'fasta')

    print(f"Finished. The identified plasmid contigs are saved in {output_path}.")
    out_info.close()


if __name__ == "__main__":
    
    plasme_work_dir_path = os.getcwd()
    plasme_args = plasme_cmd()

    db_dir = f"{plasme_work_dir_path}/DB"
    if os.path.exists(db_dir) and os.listdir(db_dir) != 0:
        if not os.path.exists(f"{db_dir}/plsdb_Mar30.dmnd"):
            build_db(db_dir=db_dir, 
                     num_threads=plasme_args.thread)
    else:
        if os.path.exists(f"{plasme_work_dir_path}/DB.zip"):
            print("Unzip the reference plasmid database ... ...")
            shutil.unpack_archive(f"{plasme_work_dir_path}/DB.zip", plasme_work_dir_path)
            build_db(db_dir=db_dir, 
                    num_threads=plasme_args.thread)
        else:
            print(f"Please download the database from [Google Drive](https://drive.google.com/file/d/1a7iKLI6NFUGHnGAd79wU_CoNvsG4OiBl/view?usp=sharing) (or [OneDrive](https://portland-my.sharepoint.com/:u:/g/personal/xubotang2-c_my_cityu_edu_hk/EW3nhkuiozpMhnkEuiafZhQBRAIsGzKL50RBQP1CSX6RXw?e=8kAGUj)")

    temp_dir = ''
    if plasme_args.temp:
        temp_dir = plasme_args.temp
    else:
        temp_dir = f"{plasme_work_dir_path}/temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # build the output directory
    if os.path.exists(plasme_args.output):
        print(f"The output file already exists. Please rename the output file.")
        exit(0)

    # Specify the taxonomy to be used for the prediction
    all_order_list = ['Enterobacterales', 'Lactobacillales', 'Bacillales', 'Pseudomonadales', 
                      'Rhodobacterales', 'Hyphomicrobiales', 'Spirochaetales', 'Corynebacteriales', 
                      'Burkholderiales', 'Xanthomonadales', 'Campylobacterales', 'Thiotrichales', 
                      'Vibrionales', 'Aeromonadales', 'Sphingomonadales', 'Eubacteriales', 
                      'Rhodospirillales', 'Micrococcales', 'Streptomycetales', 'Nostocales', 
                      'Pasteurellales', 'Neisseriales', 'Bacteroidales', 'Legionellales', 
                      'Synechococcales', 'Cytophagales', 'Mycoplasmatales', 'Alteromonadales', 
                      'Chlamydiales', 'Chroococcales', 'Flavobacteriales', 'Thermales', 
                      'Entomoplasmatales', 'Deinococcales', 'other']

    taxonomy = plasme_args.taxonomy
    if taxonomy:
        if taxonomy not in all_order_list:
            taxonomy = 'unified'
            print(f"Unknown taxonomy, predict using the 'unified' model.")

    predict(contig_path=plasme_args.input, 
            ref_plas_db_path=f"{db_dir}/plsdb_Mar30", 
            ref_tax_path=f"{db_dir}/plsdb_taxon.tsv", 
            temp_dir=temp_dir, 
            out_path=f"{temp_dir}/PLASMe_candidate.csv", 
            ref_ol_path=f"{db_dir}/plas_overlap.csv",
            db_dir=db_dir, min_cov=0.0, 
            num_threads=plasme_args.thread,
            taxonomy=taxonomy)

    # 'high-precision', 'balance', 'high-sensitivity'
    ident = plasme_args.identity
    covg = plasme_args.coverage
    prob = plasme_args.probability
    if plasme_args.mode == 'high-precision':
        ident, covg = 0.9, 0.9
        prob = 0.9
    elif plasme_args.mode == 'balance':
        ident, covg = 0.9, 0.9
        prob = 0.5
    elif plasme_args.mode == 'high-sensitivity':
        ident, covg = 0.7, 0.1
        prob = 0.5
    else:
        pass

    plasme_output(rst_path=f"{temp_dir}/PLASMe_candidate.csv", 
                  blastn_rst=f"{temp_dir}/blastn.csv", 
                  contig_path=plasme_args.input, ident_thres=ident, 
                  cov_thres=covg, pred_thres=prob, 
                  output_path=f"{plasme_args.output}")
