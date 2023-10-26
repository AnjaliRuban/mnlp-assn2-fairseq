
#!/bin/sh

LS_ROOT=$1
SAVE_DIR=$2

# fairseq-train ${LS_ROOT} --save-dir ${SAVE_DIR} \
#  --config-yaml config.yaml --train-subset train_manifest --valid-subset dev_manifest \
#  --num-workers 4 --max-tokens 40000 --max-update 300000 \
#  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
#  --arch s2t_transformer_s --share-decoder-input-output-embed \
#  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
#  --clip-norm 10.0 --seed 1 --update-freq 8 --patience 10

# fairseq-train ${LS_ROOT} --save-dir ${SAVE_DIR} \
#	--arch wav2vec2 \
#	--train-subset train_manifest --valid-subset dev_manifest \
#	--max-tokens 40000 \
#	--task speech_to_text \
#	--config-yaml config.yaml \
#	--finetune-from-model ../gn_w2v2/wav2vec_small.pt  

# fairseq-hydra-train \
#	task.data=/data/tir/projects/tir4/users/akanthar/classes/mling11737/mnlp-assn2-f23/gn/ \
#	model.w2v_path=/data/tir/projects/tir4/users/akanthar/classes/mling11737/mnlp-assn2-f23/gn_w2v2/wav2vec_small.pt \
#	--config-dir /data/tir/projects/tir4/users/akanthar/classes/mling11737/mnlp-assn2-f23/fairseq/examples/wav2vec/config/finetuning \
#	--config-name base_100h

CHECKPOINT_FILENAME=checkpoint_best.pt
for SUBSET in test_manifest; do
	  fairseq-generate ${LS_ROOT} --config-yaml config.yaml --gen-subset ${SUBSET} \
		      	--task speech_to_text --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} \
			--max-tokens 50000 --beam 5 --scoring wer
done
