python convert_eomt_dinov3_to_hf.py \
    /Users/nielsrogge/Downloads/pytorch_model.bin \
    . \
    --backbone-repo-id facebook/dinov3-vits16-pretrain-lvd1689m \
    --verify \
    --original-repo-path /Users/nielsrogge/Documents/python_projects/eomt \
    --push-to-hub