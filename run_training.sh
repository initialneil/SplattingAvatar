# python train_splatting_avatar.py --config configs/splatting_avatar.yaml --dat_dir V:/Dataset/SplattingAvatar/yufeng --model_path output-splatting/last_checkpoint
# python train_splatting_avatar.py --config configs/splatting_avatar.yaml --dat_dir V:/Dataset/SplattingAvatar/malte_1 --model_path output-splatting/last_checkpoint
# python train_splatting_avatar.py --config configs/splatting_avatar.yaml --dat_dir V:/Dataset/SplattingAvatar/marcel --model_path output-splatting/last_checkpoint
# python train_splatting_avatar.py --config configs/splatting_avatar.yaml --dat_dir V:/Dataset/SplattingAvatar/nf_01 --model_path output-splatting/last_checkpoint
# python train_splatting_avatar.py --config configs/splatting_avatar.yaml --dat_dir V:/Dataset/SplattingAvatar/nf_03 --model_path output-splatting/last_checkpoint
# python train_splatting_avatar.py --config configs/splatting_avatar.yaml --dat_dir V:/Dataset/SplattingAvatar/obama --model_path output-splatting/last_checkpoint
# python train_splatting_avatar.py --config configs/splatting_avatar.yaml --dat_dir V:/Dataset/SplattingAvatar/person_0004 --model_path output-splatting/last_checkpoint
# python train_splatting_avatar.py --config configs/splatting_avatar.yaml --dat_dir V:/Dataset/SplattingAvatar/wojtek_1 --model_path output-splatting/last_checkpoint

python train_splatting_avatar.py --config "configs/splatting_avatar.yaml;configs/instant_avatar.yaml" --dat_dir V:/Dataset/PeopleSnapshot/instant_avatar/male-3-casual --model_path output-splatting/last_checkpoint
python train_splatting_avatar.py --config "configs/splatting_avatar.yaml;configs/instant_avatar.yaml" --dat_dir V:/Dataset/PeopleSnapshot/instant_avatar/male-4-casual --model_path output-splatting/last_checkpoint
python train_splatting_avatar.py --config "configs/splatting_avatar.yaml;configs/instant_avatar.yaml" --dat_dir V:/Dataset/PeopleSnapshot/instant_avatar/female-3-casual --model_path output-splatting/last_checkpoint
python train_splatting_avatar.py --config "configs/splatting_avatar.yaml;configs/instant_avatar.yaml" --dat_dir V:/Dataset/PeopleSnapshot/instant_avatar/female-4-casual --model_path output-splatting/last_checkpoint

