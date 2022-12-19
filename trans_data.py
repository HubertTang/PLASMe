from Bio import SeqIO
from collections import Counter
import numpy as np
import os
import pandas as pd
import pickle as pkl


def contig2sentance(p2a_path, blast_path, aa_count_path, test_aa_path, pc_thres, out_dir, feat_len=400):
    """Convert contigs to sentences, pad the word if this word doesn't hit any PCs.
    Updated version, build a directory of query on the best hit.
    """
    # Load the threshold of each protein cluster
    pc_thres_dict = {}
    with open(pc_thres) as pt:
        for l in pt:
            l = l.split()
            pc = l[0]
            thres = float(l[1])
            if thres > 0:
                pc_thres_dict[pc] = thres

    # Build the output directory if the directory doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    pc2db_aa_df = pd.read_csv(p2a_path, sep=',', header=None)
    pc2wordsid = {pc: idx for idx, pc in enumerate(sorted(set(pc2db_aa_df[0].values)))}
    protein2pc = {protein: pc for protein, pc in zip(pc2db_aa_df[1].values, pc2db_aa_df[0].values)}
    # blast_df = pd.read_csv(abc_path, sep=' ', names=['query', 'ref', 'evalue'])

    # Built the dictionay of query: the number of protein
    num_aa_dict = {}
    with open(aa_count_path) as acp:
        for l in acp:
            l = l.split()
            num_aa_dict[l[0]] = int(l[1])

    # # Parse the abc results
    # query_dict = {}
    # for query, ref, evalue in zip(blast_df['query'].values, blast_df['ref'].values, blast_df['evalue'].values):
    #     if query in query_dict:
    #         if ref in protein2pc:
    #             if evalue < query_dict[query][1]:
    #                 query_dict[query] = [ref, evalue]
    #     else:
    #         query_dict[query] = [ref, evalue]

    # Parse the abc results
    query_dict = {}
    with open(blast_path) as bp:
        for l in bp:
            record = l.split()
            query = record[0]
            ref = record[1]
            ident = float(record[2])
            evalue = float(record[10])
            if query in query_dict:
                if ref in protein2pc:
                    if evalue < query_dict[query][1]:
                        query_dict[query] = [ref, evalue, ident]
            else:
                query_dict[query] = [ref, evalue, ident]
    
    contig2pcs = {}
    nonref_seq_query = set()

    for query, info in query_dict.items():
        contig = query.rsplit('_', 1)[0]
        idx    = int(query.rsplit('_', 1)[1]) - 1

        ref = info[0]
        evalue = info[1]
        ident = info[2]

        # filter out the protein whose identity is smaller than the threshold
        try: 
            if ident < pc_thres_dict[protein2pc[ref]]:
                continue
        except KeyError:
            # num_no_ref += 1
            continue

        try:
            pc = pc2wordsid[protein2pc[ref]]
        except KeyError:
            # num_no_ref += 1
            nonref_seq_query.add(query)
            continue
        
        try:
            contig2pcs[contig].append((idx, pc, evalue))
        except:
            contig2pcs[contig] = [(idx, pc, evalue)]

    # print("The number of no reference aa:", num_no_ref, len(nonref_seq_query))
    # print("The number of no reference aa:", len(nonref_seq_query))

    # Sorted by position
    contig_id_list_f = open(out_dir + '/' + 'sentence_id.list', 'w')
    for contig in contig2pcs:
        contig2pcs[contig] = sorted(contig2pcs[contig], key=lambda tup: tup[0])
        contig_id_list_f.write(f"{contig}\n")
    contig_id_list_f.close()

    # Contigs2sentence
    contig2id = {contig:idx for idx, contig in enumerate(contig2pcs.keys())}
    id2contig = {idx:contig for idx, contig in enumerate(contig2pcs.keys())}
    sentence = np.zeros((len(contig2id.keys()), feat_len))
    sentence_weight = np.ones((len(contig2id.keys()), feat_len))
    for row in range(sentence.shape[0]):
        contig = id2contig[row]
        pcs = contig2pcs[contig]

        for i in range(num_aa_dict[contig]):
            if i < feat_len:
                sentence[row][i] = 1
        
        for (idx, pc, pc_w) in pcs:
            if idx < feat_len:
                sentence[row][idx] = pc + 2
                sentence_weight[row][idx] = pc_w

        # for (idx, pc, pc_w) in pcs:
        #     if idx < feat_len:
        #         try:
        #             sentence[row][idx] = pc + 1
        #             sentence_weight[row][idx] = pc_w
        #         except:
        #             break

    # Corresponding Evalue weight
    sentence_weight[sentence_weight<1e-200] = 1e-200
    sentence_weight = -np.log10(sentence_weight)/200

    rec = []
    for record in SeqIO.parse(test_aa_path, 'fasta'):
        name = record.id
        name = name.rsplit('_', 1)[0]
        rec.append(name)
    counter = Counter(rec)
    total_num = np.array([counter[item] for item in id2contig.values()])
    # proportion = mapped_num/total_num

    # Store the parameters
    pkl.dump(sentence,        open(out_dir + '/' + 'sentence.feat', 'wb'))
    pkl.dump(sentence_weight, open(out_dir + '/' + 'sentence_weight.feat', 'wb'))
    pkl.dump(id2contig,       open(out_dir + '/' + 'sentence_id2contig.dict', 'wb'))
    # pkl.dump(proportion,      open(out_dir + '/' + 'sentence_proportion.feat', 'wb'))
    pkl.dump(pc2wordsid,      open(out_dir + '/' + 'pc2wordsid.dict', 'wb'))
