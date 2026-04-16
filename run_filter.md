# main_filter.py
# ========== 1. Zero-shot ==========
python3 main_filter.py --split sample_data --method zeroshot --context full --output res_zeroshot_full.json
python3 main_filter.py --split sample_data --method zeroshot --context filtered --output res_zeroshot_reranked.json

# ========== 2. One-shot ==========
python3 main_filter.py --split sample_data --method oneshot --context full --output res_oneshot_full.json
python3 main_filter.py --split sample_data --method oneshot --context filtered --output res_oneshot_reranked.json

# ========== 3. Few-shot (3-shot) ==========
python3 main_filter.py --split sample_data --method fewshot --num_shots 3 --context full --output res_fewshot_full.json
python3 main_filter.py --split sample_data --method fewshot --num_shots 3 --context filtered --output res_fewshot_reranked.json

# ========== 4. CoT Zero-shot ==========
python3 main_filter.py --split sample_data --method cot_zeroshot --context full --output res_cot_zeroshot_full.json
python3 main_filter.py --split sample_data --method cot_zeroshot --context filtered --output res_cot_zeroshot_reranked.json

# ========== 5. CoT One-shot（1-shot） ==========
python3 main_filter.py --split sample_data --method cot_fewshot --num_shots 1 --context full --output res_cot_oneshot_full.json
python3 main_filter.py --split sample_data --method cot_fewshot --num_shots 1 --context filtered --output res_cot_oneshot_reranked.json

# ========== 6. CoT Few-shot (3-shot) ==========
python3 main_filter.py --split sample_data --method cot_fewshot --num_shots 3 --context full --output res_cot_fewshot_full.json
python3 main_filter.py --split sample_data --method cot_fewshot --num_shots 3 --context filtered --output res_cot_fewshot_reranked.json


