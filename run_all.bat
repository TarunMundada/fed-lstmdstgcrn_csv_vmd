@ECHO OFF
ECHO Starting training for CHI-Taxi...

python test.py --trip_csv "CHI-taxi/tripdata_full.csv" --dataset "CHI-Taxi" --rounds 20 --epochs_per_round 20 --clients 3 --tin 12 --tout 3 --save_dir "./ckpts/CHI-Taxi" --batch_size 16 --use_attention 1 --attn_heads 4 --vmd_k 3 --save_vmd 1

ECHO Finished CHI-Taxi. Starting training for NYC-Bike...

python test.py --trip_csv "NYC-Bike/tripdata_full.csv" --dataset "NYC-Bike" --rounds 20 --epochs_per_round 20 --clients 3 --tin 12 --tout 3 --save_dir "./ckpts/NYC-Bike" --batch_size 16 --use_attention 1 --attn_heads 4 --vmd_k 3 --save_vmd 1

ECHO Finished NYC-Bike. Starting training for NYC-Taxi...

python test.py --trip_csv "NYC-Taxi/tripdata_full.csv" --dataset "NYC-Taxi" --rounds 20 --epochs_per_round 20 --clients 3 --tin 12 --tout 3 --save_dir "./ckpts/NYC-Taxi" --batch_size 16 --use_attention 1 --attn_heads 4 --vmd_k 3 --save_vmd 1

ECHO All training runs are complete! 🚀
PAUSE