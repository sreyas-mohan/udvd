import argparse
import os
import logging
import numpy as np
import random
import sys
import torch

from datetime import datetime
from torch.serialization import default_restore_location


def add_logging_arguments(parser):
	parser.add_argument("--seed", default=0, type=int, help="random number generator seed")
	parser.add_argument("--output-dir", default="experiments", help="path to experiment directories")
	parser.add_argument("--experiment", default=None, help="experiment name to be used with Tensorboard")
	parser.add_argument("--resume-training", action="store_true", help="whether to resume training")
	parser.add_argument("--restore-file", default=None, help="filename to load checkpoint")
	parser.add_argument("--no-save", action="store_true", help="don't save models or checkpoints")
	parser.add_argument("--step-checkpoints", action="store_true", help="store all step checkpoints")
	parser.add_argument("--no-log", action="store_true", help="don't save logs to file or Tensorboard directory")
	parser.add_argument("--log-interval", type=int, default=100, help="log every N steps")
	parser.add_argument("--no-visual", action="store_true", help="don't use Tensorboard")
	parser.add_argument("--visual-interval", type=int, default=100, help="log every N steps")
	parser.add_argument("--no-progress", action="store_true", help="don't use progress bar")
	parser.add_argument("--draft", action="store_true", help="save experiment results to draft directory")
	parser.add_argument("--dry-run", action="store_true", help="no log, no save, no visualization")
	return parser


def setup_experiment(args):
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	if args.dry_run:
		args.no_save = args.no_log = args.no_visual = True
		return

	args.experiment = args.experiment or f"{args.model.replace('_', '-')}"
	args.experiment = "-".join([args.experiment, 'BF' if (not args.bias) else 'B', str(args.min_noise), str(args.max_noise)])
	if not args.resume_training:
		args.experiment = "-".join([args.experiment, datetime.now().strftime("%b-%d-%H:%M:%S")])

	args.experiment_dir = os.path.join(args.output_dir, args.model, (f"drafts/" if args.draft else "") + args.experiment)
	os.makedirs(args.experiment_dir, exist_ok=True)

	if not args.no_save:
		args.checkpoint_dir = os.path.join(args.experiment_dir, "checkpoints")
		os.makedirs(args.checkpoint_dir, exist_ok=True)

	if not args.no_log:
		args.log_dir = os.path.join(args.experiment_dir, "logs")
		os.makedirs(args.log_dir, exist_ok=True)
		args.log_file = os.path.join(args.log_dir, "train.log")


def init_logging(args):
	handlers = [logging.StreamHandler()]
	if not args.no_log and args.log_file is not None:
		mode = "a" if os.path.isfile(args.resume_training) else "w"
		handlers.append(logging.FileHandler(args.log_file, mode=mode))
	logging.basicConfig(handlers=handlers, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
	logging.info("COMMAND: %s" % " ".join(sys.argv))
	logging.info("Arguments: {}".format(vars(args)))


def save_checkpoint(args, step, model, optimizer=None, scheduler=None, score=None, mode="min"):
	assert mode == "min" or mode == "max"
	last_step = getattr(save_checkpoint, "last_step", -1)
	save_checkpoint.last_step = max(last_step, step)

	default_score = float("inf") if mode == "min" else float("-inf")
	best_score = getattr(save_checkpoint, "best_score", default_score)
	if (score < best_score and mode == "min") or (score > best_score and mode == "max"):
		save_checkpoint.best_step = step
		save_checkpoint.best_score = score

	if not args.no_save and step % args.save_interval == 0:
		os.makedirs(args.checkpoint_dir, exist_ok=True)
		model = [model] if model is not None and not isinstance(model, list) else model
		optimizer = [optimizer] if optimizer is not None and not isinstance(optimizer, list) else optimizer
		scheduler = [scheduler] if scheduler is not None and not isinstance(scheduler, list) else scheduler
		state_dict = {
			"step": step,
			"score": score,
			"last_step": save_checkpoint.last_step,
			"best_step": save_checkpoint.best_step,
			"best_score": getattr(save_checkpoint, "best_score", None),
			"model": [m.state_dict() for m in model] if model is not None else None,
			"optimizer": [o.state_dict() for o in optimizer] if optimizer is not None else None,
			"scheduler": [s.state_dict() for s in scheduler] if scheduler is not None else None,
			"args": argparse.Namespace(**{k: v for k, v in vars(args).items() if not callable(v)}),
		}

		if args.step_checkpoints:
			torch.save(state_dict, os.path.join(args.checkpoint_dir, "checkpoint{}.pt".format(step)))
		if (score < best_score and mode == "min") or (score > best_score and mode == "max"):
			torch.save(state_dict, os.path.join(args.checkpoint_dir, "checkpoint_best.pt"))
		if step > last_step:
			torch.save(state_dict, os.path.join(args.checkpoint_dir, "checkpoint_last.pt"))


def load_checkpoint(args, model=None, optimizer=None, scheduler=None):
	if args.restore_file is not None and os.path.isfile(args.restore_file):
		print('restoring model..')
		state_dict = torch.load(args.restore_file, map_location=lambda s, l: default_restore_location(s, "cpu"))

		model = [model] if model is not None and not isinstance(model, list) else model
		optimizer = [optimizer] if optimizer is not None and not isinstance(optimizer, list) else optimizer
		scheduler = [scheduler] if scheduler is not None and not isinstance(scheduler, list) else scheduler

		if "best_score" in state_dict:
			save_checkpoint.best_score = state_dict["best_score"]
			save_checkpoint.best_step = state_dict["best_step"]
		if "last_step" in state_dict:
			save_checkpoint.last_step = state_dict["last_step"]
		if model is not None and state_dict.get("model", None) is not None:
			for m, state in zip(model, state_dict["model"]):
				m.load_state_dict(state)
		if optimizer is not None and state_dict.get("optimizer", None) is not None:
			for o, state in zip(optimizer, state_dict["optimizer"]):
				o.load_state_dict(state)
		if scheduler is not None and state_dict.get("scheduler", None) is not None:
			for s, state in zip(scheduler, state_dict["scheduler"]):
				milestones = s.milestones
				state['milestones'] = milestones
				s.load_state_dict(state)
				s.milestones = milestones

		logging.info("Loaded checkpoint {}".format(args.restore_file))
		return state_dict
