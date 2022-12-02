export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
python stereoset/experiment.py --persistent_dir "/export/home/kraft/data/stereoset_data" --model "LukeForMaskedLM" --model_name_or_path "studio-ousia/luke-base"
#python stereoset/experiment.py --persistent_dir "/export/home/kraft/data/stereoset_data" --model "RobertaForMaskedLM" --model_name_or_path "roberta-base"

