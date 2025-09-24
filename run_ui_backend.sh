./run_build.sh && \
    # LD_PRELOAD=$(g++ -print-file-name=libasan.so) \
    cd webui/backend && python3 server.py --model ../../models/DeepSeek-R1-Distill-Qwen-1.5B --test