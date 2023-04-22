# Fill in the username and password here after applying for access <https://trec.nist.gov/results.html>
AUTH=username:password

# msmarco-passage/trec-dl-2019
mkdir dl19-runs
for run in input.bm25base_ax_p input.bm25base_p input.bm25base_prf_p input.bm25base_rm3_p input.bm25tuned_ax_p input.bm25tuned_p input.bm25tuned_prf_p input.bm25tuned_rm3_p input.ICT-BERT2 input.ICT-CKNRM_B input.ICT-CKNRM_B50 input.idst_bert_p1 input.idst_bert_p2 input.idst_bert_p3 input.idst_bert_pr1 input.idst_bert_pr2 input.ms_duet_passage input.p_bert input.p_exp_bert input.p_exp_rm3_bert input.runid2 input.runid3 input.runid4 input.runid5 input.srchvrs_ps_run1 input.srchvrs_ps_run2 input.srchvrs_ps_run3 input.test1 input.TUA1-1 input.TUW19-p1-f input.TUW19-p1-re input.TUW19-p2-f input.TUW19-p2-re input.TUW19-p3-f input.TUW19-p3-re input.UNH_bm25 input.UNH_exDL_bm25
do
    echo dl19 $run
    curl -u $AUTH "https://trec.nist.gov/results/trec28/deep/$run.gz" | gunzip | python filter.py msmarco-passage/trec-dl-2019/judged > dl19-runs/$run
done

# msmarco-passage/trec-dl-2020
mkdir dl20-runs
for run in input.1 input.2 input.bcai_bertl_pass input.bcai_class_pass input.bert_6 input.bigIR-BERT-R input.bigIR-DCT-T5-F input.bigIR-T5-BERT-F input.bigIR-T5-R input.bigIR-T5xp-T5-F input.bl_bcai_mdl1_vs input.bl_bcai_mdl1_vt input.bm25_bert_token input.CoRT-bm25 input.CoRT-electra input.CoRT-standalone input.DLH_d_5_t_25 input.DoRA_Large_1k input.DoRA_Large input.DoRA_Med input.DoRA_Small input.fr_pass_roberta input.indri-fdm input.indri-lmds input.indri-sdm input.med_1k input.NLE_pr1 input.NLE_pr2 input.NLE_pr3 input.nlm-bert-rr input.nlm-ens-bst-2 input.nlm-ens-bst-3 input.nlm-prfun-bert input.pash_f1 input.pash_f2 input.pash_f3 input.pash_r1 input.pash_r2 input.pash_r3 input.p_bm25 input.p_bm25rm3_duo input.p_bm25rm3 input.p_d2q_bm25_duo input.p_d2q_bm25 input.p_d2q_bm25rm3 input.p_d2q_rm3_duo input.pinganNLP1 input.pinganNLP2 input.pinganNLP3 input.relemb_mlm_0_2 input.RMIT-Bart input.rr-pass-roberta input.small_1k input.terrier-BM25 input.terrier-DPH input.TF_IDF_d_2_t_50 input.TUW-TK-2Layer input.TUW-TK-Sparse
do
    echo dl20 $run
    curl -u $AUTH "https://trec.nist.gov/results/trec29/deep/$run.gz" | gunzip | python filter.py msmarco-passage/trec-dl-2020/judged > dl20-runs/$run
done

# msmarco-passage-v2/trec-dl-2021
mkdir dl21-runs
for run in input.bcai_p_mbert input.bcai_p_vbert input.bl_bcai_p_nn_rt input.bl_bcai_p_trad input.bl_bcai_wloo_p input.Fast_Forward_3 input.Fast_ForwardP_2 input.Fast_ForwardP_5 input.ielab-AD-uni input.ielab-robertav1 input.ielab-robertav2 input.ielab-TILDEv2 input.ielab-uniCOIL input.ihsm_bicolbert input.ihsm_colbert64 input.ihsm_poly8q input.mono_d3 input.mono_electra_h3 input.mono_h3 input.NLE_P_quick input.NLE_P_v1 input.NLE_P_V1andV2 input.pash_f1 input.pash_f2 input.pash_f3 input.pash_r1 input.pash_r2 input.pash_r3 input.pass_full_1000 input.pass_full_1000e input.pass_rank_100 input.paug_bm25 input.paug_bm25rm3 input.p_bm25 input.p_bm25rm3 input.p_f10_mdt53b input.p_f10_mdt5base input.p_f10_mt53b input.p_fusion00 input.p_fusion10 input.p_tct0 input.p_tct1 input.p_unicoil0 input.top1000 input.TUW_DR_Base input.TUW_TAS-B_768 input.TUW_TAS-B_ANN input.uogTrBasePD input.uogTrBasePDQ input.uogTrPC input.uogTrPCP input.uogTrPot5 input.watpfd input.watpff input.watpfp input.watprd input.WLUPassage input.WLUPassage1
do
    echo dl21 $run
    curl -u $AUTH "https://trec.nist.gov/results/trec30/deep/$run.gz" | gunzip | python filter.py msmarco-passage-v2/trec-dl-2021/judged > dl21-runs/$run
done
