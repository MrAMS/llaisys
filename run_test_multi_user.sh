./run_build.sh && \
    # LD_PRELOAD=$(g++ -print-file-name=libasan.so) \
    python test/test_multi_user.py --model models/DeepSeek-R1-Distill-Qwen-1.5B --test