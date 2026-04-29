.PHONY: help clean clean-output clean-cache train train-upper eval peek diagnose extract smoke

PYTHON ?= python
MPS_PREFIX = PYTORCH_ENABLE_MPS_FALLBACK=1
MAX_STEPS ?= 50
TOTAL_TIMESTEPS ?= 100000
MODEL_PATH ?= outputs/ppo/ppo_maxsteps$(MAX_STEPS).zip
LOG ?= outputs/ppo/run_$(MAX_STEPS).log
EPISODES ?= 50

help:
	@echo "Targets:"
	@echo "  train          Train PPO (MAX_STEPS=$(MAX_STEPS), TOTAL_TIMESTEPS=$(TOTAL_TIMESTEPS))"
	@echo "  train-upper    Train with --upper_hemisphere_only"
	@echo "  eval           Evaluate MODEL_PATH=$(MODEL_PATH) on EPISODES=$(EPISODES) episodes"
	@echo "  peek           Save sample renders to outputs/peek/"
	@echo "  diagnose       Verify dataset layout"
	@echo "  extract        Unzip ShapeNetCore archives"
	@echo "  smoke          Quick env + reward smoke test"
	@echo "  clean-output   Remove outputs/ppo/ and outputs/eval/"
	@echo "  clean-cache    Remove __pycache__ dirs"
	@echo "  clean          clean-output + clean-cache"

train:
	mkdir -p outputs/ppo
	$(MPS_PREFIX) $(PYTHON) train_ppo.py --max_steps $(MAX_STEPS) --total_timesteps $(TOTAL_TIMESTEPS) 2>&1 | tee $(LOG)

train-upper:
	mkdir -p outputs/ppo
	$(MPS_PREFIX) $(PYTHON) train_ppo.py --max_steps $(MAX_STEPS) --total_timesteps $(TOTAL_TIMESTEPS) --upper_hemisphere_only 2>&1 | tee $(LOG)

eval:
	mkdir -p outputs/eval
	$(MPS_PREFIX) $(PYTHON) evaluate.py --model_path $(MODEL_PATH) --episodes $(EPISODES) --max_steps $(MAX_STEPS)

peek:
	$(PYTHON) peek_renders.py

diagnose:
	$(PYTHON) diagnose_dataset.py --root data/ShapeNetCore

extract:
	$(PYTHON) extract_shapenet.py --root data/ShapeNetCore

smoke:
	$(PYTHON) -c "from ultralytics import YOLO; from shapenet_gym.wrappers import make_training_env; from train_ppo import make_yolo_entropy_reward; \
yolo = YOLO('yolov8n-cls.pt'); \
env = make_training_env('data/ShapeNetCore', categories=['02691156','02958343'], yolo_model=yolo, normalize=False, reward_fn=make_yolo_entropy_reward(scale=15.0, correctness_bonus=1.0)); \
obs, _ = env.reset(); print('keys:', list(obs.keys())); print('summary:', obs['summary']); \
obs, r, *_ = env.step(0); print('r=', r)"

clean-output:
	rm -rf outputs/ppo outputs/eval

clean-cache:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean: clean-output clean-cache
