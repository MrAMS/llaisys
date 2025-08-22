./run_build.sh && \
    # LD_PRELOAD=$(g++ -print-file-name=libasan.so) \
    python test/test_infer.py --model models/DeepSeek-R1-Distill-Qwen-1.5B --test