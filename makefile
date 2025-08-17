PY?=python

venv:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r training/requirements.txt

train:
	$(PY) training/train.py --train data/train --val data/val --epochs 15 --batch_size 8 --image_size 256 --bit_len 32 --lr 1e-3 --lambda_rec 0.3 --lambda_bit 3.0 --lambda_adv 0.0 --lambda_tv 0.002 --model_dir model_out

presence:
	$(PY) scripts/presence_eval.py --encoder model_out/encoder.pt --bit_len 32 --key aegis

robust:
	$(PY) scripts/robust_eval.py --encoder model_out/encoder.pt --decoder model_out/decoder.pt --bit_len 32

demo:
	$(PY) scripts/embed_savebits.py --encoder model_out/encoder.pt --in_img data/val/$(shell ls data/val | head -n 1) --out_img test_wm.jpg --bits_out test_bits.npy
	$(PY) scripts/detect_pn_shift_scale.py --encoder model_out/encoder.pt --img test_wm.jpg --bits test_bits.npy
