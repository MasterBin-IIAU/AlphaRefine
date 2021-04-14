for code in a b c d e;
do
    python eval_got10k.py --tracker ATOM_RF --rf_code $code &
done