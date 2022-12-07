export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
#python stereoset/experiment.py --persistent_dir "/export/home/kraft/data/stereoset_data" --model "LukeForMaskedLM" --model_name_or_path "studio-ousia/luke-base"
#python stereoset/experiment.py --persistent_dir "/export/home/kraft/data/stereoset_data" --model "RobertaForMaskedLM" --model_name_or_path "roberta-base"

python stereoset/stereoset_evaluation.py --persistent_dir "/export/home/kraft/data/stereoset_data" --predictions_dir "/export/home/kraft/data/stereoset_data/results/stereoset" --predictions_file "/export/home/kraft/data/stereoset_data/results/stereoset/stereoset_m-LukeForMaskedLM_c-studio-ousia-luke-base.json" --output_file "/export/home/kraft/data/stereoset_data/results/stereoset/aggr-results_stereoset_m-LukeForMaskedLM_c-studio-ousia-luke-base.json"