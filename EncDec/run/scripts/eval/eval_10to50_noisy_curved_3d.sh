
python eval_result.py --number_of_tracks=1000 \
                --max_size_tracks=30 \
                --max_size_hits=450 \
                --batch_size=64 \
                --file_name="./data/experiment_3d_noisy-100k-events-10-to-50-curved-tracks/events_all/hits_and_tracks_3d_events_all.csv" \
                --load_dir="./saved_models/10to50_noisy_curved_3d/cp.ckpt" \
                --output_dir="./saved_models/10to50_noisy_curved_3d/plots" \
                --parsed_data_dir="./data/experiment_3d_noisy-100k-events-10-to-50-curved-tracks/parsed_data"