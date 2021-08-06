import argparse
import logging
import sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import data, models, utils


def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    utils.setup_experiment(args)
    utils.init_logging(args)

    # Build data loaders, a model and an optimizer
    model = models.build_model(args).to(device)
    cpf = model.c # channels per frame
    mid = args.n_frames // 2
    model = nn.DataParallel(model)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 60, 70, 80, 90, 100], gamma=0.5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.5)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, threshold=0.001, threshold_mode='abs', min_lr=5e-6)
    logging.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

    if args.resume_training:
        state_dict = utils.load_checkpoint(args, model, optimizer, scheduler)
        global_step = state_dict['last_step']
        start_epoch = int(state_dict['last_step']/(403200/state_dict['args'].batch_size))+1
    else:
        global_step = -1
        start_epoch = 0

    train_loader, valid_loader = data.build_dataset(args.dataset, args.data_path, batch_size=args.batch_size, dataset=args.dataset_aux, video=args.video, image_size=args.image_size, stride=args.stride, n_frames=args.n_frames, aug=args.aug, dist=args.noise_dist, mode=args.noise_mode, noise_std=args.noise_std, min_noise=args.min_noise, max_noise=args.max_noise, sample=args.sample, heldout=args.heldout)

    # Track moving average of loss values
    train_meters = {name: utils.RunningAverageMeter(0.98) for name in (["train_loss", "train_psnr", "train_ssim"])}
    # frame_meters = {name: utils.RunningAverageMeter(0.98) for name in (["frame_loss", "frame_psnr", "frame_ssim"])}
    if args.loss == "loglike":
        mean_meters = {name: utils.AverageMeter() for name in (["mean_psnr", "mean_ssim"])}
    valid_meters = {name: utils.AverageMeter() for name in (["valid_loss", "valid_psnr", "valid_ssim"])}
    writer = SummaryWriter(log_dir=args.experiment_dir) if not args.no_visual else None

    flag28 = True
    flag29 = True
    flag30 = True
    flag31 = True

    # Load Pretrained Model
    # PATH = "/scratch/ds6516/video_denoising/experiments/blind-video-net-4/blind-video-net-4-BF-5-50-Sep-24-12:11:16/checkpoints/checkpoint_last.pt"
    #
    # state_dict = torch.load(PATH)["model"][0]
    # own_state = model.state_dict()
    #
    # for name, param in state_dict.items():
    #     if name not in own_state:
    #         print("here")
    #         continue
    #     if isinstance(param, nn.Parameter):
    #         # backwards compatibility for serialized parameters
    #         param = param.data
    #     own_state[name].copy_(param)

    for epoch in range(start_epoch, args.num_epochs):
        if args.resume_training:
            if epoch %10 == 0:
                optimizer.param_groups[0]["lr"] /= 2
                print('learning rate reduced by factor of 2')

        train_bar = utils.ProgressBar(train_loader, epoch)
        for meter in train_meters.values():
            meter.reset()
        if args.loss == "loglike":
            for meter in mean_meters.values():
                meter.reset()
        # for meter in frame_meters.values():
        # 	meter.reset()

        for batch_id, (inputs, noisy_inputs) in enumerate(train_bar):
#             if batch_id > 10:
#                 break
            model.train()

            global_step += 1
            inputs = inputs.to(device)
            noisy_inputs = noisy_inputs.to(device)
            # print("here", inputs.shape)
#             noise = utils.get_noise(inputs, dist = args.noise_dist, mode = args.noise_mode,
#                                             min_noise = args.min_noise, max_noise = args.max_noise,
#                                             noise_std = args.noise_std)

#             noisy_inputs = noise + inputs;
#             N, C, H, W = inputs.shape
#             noise_map = (args.noise_std/255)*torch.ones(N, 1, H, W).to(device)
#             out = model(noisy_inputs, noise_map)
            outputs, est_sigma = model(noisy_inputs)
            # loss = F.mse_loss(outputs, inputs, reduction="sum") / (inputs.size(0) * 2)
#             frame = inputs[:, (mid*cpf):((mid+1)*cpf), :, :]
#             noisy_frame = frame.clone().to(device)
            noisy_frame = noisy_inputs[:, (mid*cpf):((mid+1)*cpf), :, :]
            # noisy_frame1 = noisy_inputs[:, (1*cpf):(2*cpf), :, :]
            # noisy_frame3 = noisy_inputs[:, (3*cpf):(4*cpf), :, :]
            # frame_loss = (utils.loss_function(out1, noisy_frame1, mode=args.loss, sigma=args.noise_std, device=device)
            # 			  + utils.loss_function(out2, noisy_frame2, mode=args.loss, sigma=args.noise_std, device=device)
            # 			  + utils.loss_function(out3, noisy_frame3, mode=args.loss, sigma=args.noise_std, device=device))
#             loss = utils.loss_function(out, frame, mode=args.loss, sigma=args.noise_std, device=device)
            if args.blind_noise:
                loss = utils.loss_function(outputs, noisy_frame, mode=args.loss, sigma=est_sigma, device=device)
            else:
                loss = utils.loss_function(outputs, noisy_frame, mode=args.loss, sigma=args.noise_std/255, device=device)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if args.loss == "loglike":
                with torch.no_grad():
                    if args.blind_noise:
                        outputs, mean_image = utils.post_process(outputs, noisy_frame, model=args.model, sigma=est_sigma, device=device)
                    else:
                        outputs, mean_image = utils.post_process(outputs, noisy_frame, model=args.model, sigma=args.noise_std/255, device=device)
                # outputs1, _ = utils.post_process(out1, noisy_frame1, model=args.model, sigma=args.noise_std, device=device)
                # outputs2, _ = utils.post_process(out2, noisy_frame2, model=args.model, sigma=args.noise_std, device=device)
                # outputs3, _ = utils.post_process(out3, noisy_frame3, model=args.model, sigma=args.noise_std, device=device)

            train_psnr = utils.psnr(inputs[:, (mid*cpf):((mid+1)*cpf), :, :], outputs, normalized=True, raw=False)
            train_ssim = utils.ssim(inputs[:, (mid*cpf):((mid+1)*cpf), :, :], outputs, normalized=True, raw=False)
            train_meters["train_loss"].update(loss.item())
            train_meters["train_psnr"].update(train_psnr.item())
            train_meters["train_ssim"].update(train_ssim.item())

            if args.loss == "loglike":
                mean_psnr = utils.psnr(inputs[:, (mid*cpf):((mid+1)*cpf), :, :], mean_image, normalized=True, raw=False)
                mean_ssim = utils.ssim(inputs[:, (mid*cpf):((mid+1)*cpf), :, :], mean_image, normalized=True, raw=False)
                mean_meters["mean_psnr"].update(mean_psnr.item())
                mean_meters["mean_ssim"].update(mean_ssim.item())

            # frame_psnr = (utils.psnr(inputs[:, (1*cpf):(2*cpf), :, :], outputs1)
            # 			  + utils.psnr(inputs[:, (mid*cpf):((mid+1)*cpf), :, :], outputs2)
            # 			  + utils.psnr(inputs[:, (3*cpf):(4*cpf), :, :], outputs3))/3
            # frame_ssim = (utils.ssim(inputs[:, (1*cpf):(2*cpf), :, :], outputs1)
            # 			  + utils.ssim(inputs[:, (mid*cpf):((mid+1)*cpf), :, :], outputs2)
            # 			  + utils.ssim(inputs[:, (3*cpf):(4*cpf), :, :], outputs3))/3
            # frame_meters["frame_loss"].update(frame_loss.item()/3)
            # frame_meters["frame_psnr"].update(frame_psnr.item())
            # frame_meters["frame_ssim"].update(frame_ssim.item())

            if args.loss == "loglike":
                train_bar.log(dict(**train_meters, **mean_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)
            else:
                train_bar.log(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)

            if writer is not None and global_step % args.log_interval == 0:
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("loss/train", loss.item(), global_step)
                writer.add_scalar("psnr/train", train_psnr.item(), global_step)
                writer.add_scalar("ssim/train", train_ssim.item(), global_step)
                if args.loss == "loglike":
                    writer.add_scalar("psnr/mean", mean_psnr.item(), global_step)
                    writer.add_scalar("ssim/mean", mean_ssim.item(), global_step)
                # writer.add_scalar("loss/frame", frame_loss.item(), global_step)
#                 writer.add_scalar("psnr/frame", frame_psnr.item(), global_step)
#                 writer.add_scalar("ssim/frame", frame_ssim.item(), global_step)
                gradients = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None], dim=0)
                writer.add_histogram("gradients", gradients, global_step)
                sys.stdout.flush()

            if (batch_id+1) % 200 == 0:
                if args.loss == "loglike":
                    logging.info(train_bar.print(dict(**train_meters, **mean_meters, lr=optimizer.param_groups[0]["lr"]))+f" | {batch_id+1} mini-batches ended")
                else:
                    logging.info(train_bar.print(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]))+f" | {batch_id+1} mini-batches ended")
#                 if optimizer.param_groups[0]["lr"] > 1e-6:
#                     if train_psnr.item() > 31:
#                         if flag31:
#                             flag31 = False
#                             for g in optimizer.param_groups:
#                                 g['lr'] = g['lr']/2
#                     elif train_psnr.item() > 30:
#                         if flag30:
#                             flag30 = False
#                             for g in optimizer.param_groups:
#                                 g['lr'] = g['lr']/2
#                     elif train_psnr.item() > 29:
#                         if flag29:
#                             flag29 = False
#                             for g in optimizer.param_groups:
#                                 g['lr'] = g['lr']/2
#                     elif train_psnr.item() > 28:
#                         if flag28:
#                             flag28 = False
#                             for g in optimizer.param_groups:
#                                 g['lr'] = g['lr']/5
            if (batch_id+1) % 2000 == 0:
                model.eval()
                for meter in valid_meters.values():
                    meter.reset()
                if args.loss == "loglike":
                    for meter in mean_meters.values():
                        meter.reset()

                valid_bar = utils.ProgressBar(valid_loader)
                running_valid_psnr = 0.0
                for sample_id, (sample, noisy_inputs) in enumerate(valid_bar):
#                     if sample_id > 100:
#                         break
                    if args.heldout and (not sample_id == len(valid_loader.dataset)-3):
                        continue
                    with torch.no_grad():
                        sample = sample.to(device)
                        noisy_inputs = noisy_inputs.to(device)
#                         noise = utils.get_noise(sample, dist = args.noise_dist, mode = 'S',
#                                                         noise_std = args.noise_std) # (args.min_noise +  args.max_noise)/(2*255.)
#                         noise = utils.get_noise(sample, dist = args.noise_dist, mode = args.noise_mode,
#                                             min_noise = args.min_noise, max_noise = args.max_noise,
#                                             noise_std = args.noise_std)
#                         noisy_inputs = noise + sample;
    #                     N, C, H, W = sample.shape
    #                     noise_map = (args.noise_std/255)*torch.ones(N, 1, H, W).to(device)
    #                     out = model(noisy_inputs, noise_map)
                        outputs, est_sigma = model(noisy_inputs)
    #                     frame = sample[:, (mid*cpf):((mid+1)*cpf), :, :]
    #                     noisy_frame = frame.clone().to(device)
                        noisy_frame = noisy_inputs[:, (mid*cpf):((mid+1)*cpf), :, :]

                        if args.blind_noise:
                            loss = utils.loss_function(outputs, noisy_frame, mode=args.loss, sigma=est_sigma, device=device)
                        else:
                            loss = utils.loss_function(outputs, noisy_frame, mode=args.loss, sigma=args.noise_std/255, device=device)

                        if args.loss == "loglike":
                            if args.blind_noise:
                                outputs, mean_image = utils.post_process(outputs, noisy_frame, model=args.model, sigma=est_sigma, device=device)
                            else:
                                outputs, mean_image = utils.post_process(outputs, noisy_frame, model=args.model, sigma=args.noise_std/255, device=device)

                        valid_psnr = utils.psnr(sample[:, (mid*cpf):((mid+1)*cpf), :, :], outputs, normalized=True, raw=False)
                        valid_ssim = utils.ssim(sample[:, (mid*cpf):((mid+1)*cpf), :, :], outputs, normalized=True, raw=False)
                        running_valid_psnr += valid_psnr
                        valid_meters["valid_loss"].update(loss.item())
                        valid_meters["valid_psnr"].update(valid_psnr.item())
                        valid_meters["valid_ssim"].update(valid_ssim.item())

                        if args.loss == "loglike":
                            mean_psnr = utils.psnr(sample[:, (mid*cpf):((mid+1)*cpf), :, :], mean_image, normalized=True, raw=False)
                            mean_ssim = utils.ssim(sample[:, (mid*cpf):((mid+1)*cpf), :, :], mean_image, normalized=True, raw=False)
                            mean_meters["mean_psnr"].update(mean_psnr.item())
                            mean_meters["mean_ssim"].update(mean_ssim.item())

                        # if writer is not None and sample_id < 10:
                        # 	image = torch.cat([sample, noisy_inputs, outputs], dim=0)
                        # 	image = torchvision.utils.make_grid(image.clamp(0, 1), nrow=3, normalize=False)
                        # 	writer.add_image(f"valid_samples/{sample_id}", image, global_step)
                running_valid_psnr /= (sample_id+1)

                if writer is not None:
                    writer.add_scalar("psnr/valid", valid_meters['valid_psnr'].avg, global_step)
                    writer.add_scalar("ssim/valid", valid_meters['valid_ssim'].avg, global_step)
                    sys.stdout.flush()

                if args.loss == "loglike":
                    logging.info("EVAL:"+train_bar.print(dict(**valid_meters, **mean_meters, lr=optimizer.param_groups[0]["lr"])))
                else:
                    logging.info("EVAL:"+train_bar.print(dict(**valid_meters, lr=optimizer.param_groups[0]["lr"])))
                utils.save_checkpoint(args, global_step, model, optimizer, score=valid_meters["valid_loss"].avg, mode="min")
#                 scheduler.step(running_valid_psnr)
        scheduler.step()

        if args.loss == "loglike":
            logging.info(train_bar.print(dict(**train_meters, **mean_meters, lr=optimizer.param_groups[0]["lr"])))
        else:
            logging.info(train_bar.print(dict(**train_meters, lr=optimizer.param_groups[0]["lr"])))

            # if (batch_id+1) % int(174636/4) == 0:
        if (epoch+1) % args.valid_interval == 0:
            model.eval()
            for meter in valid_meters.values():
                meter.reset()
            if args.loss == "loglike":
                for meter in mean_meters.values():
                    meter.reset()

            valid_bar = utils.ProgressBar(valid_loader)
            running_valid_psnr = 0.0
            for sample_id, (sample, noisy_inputs) in enumerate(valid_bar):
#                 if sample_id > 100:
#                     break
                if args.heldout and (not sample_id == len(valid_loader.dataset)-3):
                    continue
                with torch.no_grad():
                    sample = sample.to(device)
                    noisy_inputs = noisy_inputs.to(device)
#                     noise = utils.get_noise(sample, dist = args.noise_dist, mode = 'S',
#                                                     noise_std = args.noise_std) # (args.min_noise +  args.max_noise)/(2*255.)
#                     noise = utils.get_noise(sample, dist = args.noise_dist, mode = args.noise_mode,
#                                             min_noise = args.min_noise, max_noise = args.max_noise,
#                                             noise_std = args.noise_std)

#                     noisy_inputs = noise + sample;
#                     N, C, H, W = sample.shape
#                     noise_map = (args.noise_std/255)*torch.ones(N, 1, H, W).to(device)
#                     out = model(noisy_inputs, noise_map)
                    outputs, est_sigma = model(noisy_inputs)
#                     frame = sample[:, (mid*cpf):((mid+1)*cpf), :, :]
#                     noisy_frame = frame.clone().to(device)
                    noisy_frame = noisy_inputs[:, (mid*cpf):((mid+1)*cpf), :, :]

                    if args.blind_noise:
                        loss = utils.loss_function(outputs, noisy_frame, mode=args.loss, sigma=est_sigma, device=device)
                    else:
                        loss = utils.loss_function(outputs, noisy_frame, mode=args.loss, sigma=args.noise_std/255, device=device)

                    if args.loss == "loglike":
                        if args.blind_noise:
                            outputs, mean_image = utils.post_process(outputs, noisy_frame, model=args.model, sigma=est_sigma, device=device)
                        else:
                            outputs, mean_image = utils.post_process(outputs, noisy_frame, model=args.model, sigma=args.noise_std/255, device=device)

                    valid_psnr = utils.psnr(sample[:, (mid*cpf):((mid+1)*cpf), :, :], outputs, normalized=True, raw=False)
                    valid_ssim = utils.ssim(sample[:, (mid*cpf):((mid+1)*cpf), :, :], outputs, normalized=True, raw=False)
                    running_valid_psnr += valid_psnr
                    valid_meters["valid_loss"].update(loss.item())
                    valid_meters["valid_psnr"].update(valid_psnr.item())
                    valid_meters["valid_ssim"].update(valid_ssim.item())

                    if args.loss == "loglike":
                        mean_psnr = utils.psnr(sample[:, (mid*cpf):((mid+1)*cpf), :, :], mean_image, normalized=True, raw=False)
                        mean_ssim = utils.ssim(sample[:, (mid*cpf):((mid+1)*cpf), :, :], mean_image, normalized=True, raw=False)
                        mean_meters["mean_psnr"].update(mean_psnr.item())
                        mean_meters["mean_ssim"].update(mean_ssim.item())

                    # if writer is not None and sample_id < 10:
                    # 	image = torch.cat([sample, noisy_inputs, outputs], dim=0)
                    # 	image = torchvision.utils.make_grid(image.clamp(0, 1), nrow=3, normalize=False)
                    # 	writer.add_image(f"valid_samples/{sample_id}", image, global_step)
            running_valid_psnr /= (sample_id+1)

            if writer is not None:
                writer.add_scalar("psnr/valid", valid_meters['valid_psnr'].avg, global_step)
                writer.add_scalar("ssim/valid", valid_meters['valid_ssim'].avg, global_step)
                sys.stdout.flush()

            if args.loss == "loglike":
                logging.info("EVAL:"+train_bar.print(dict(**valid_meters, **mean_meters, lr=optimizer.param_groups[0]["lr"])))
            else:
                logging.info("EVAL:"+train_bar.print(dict(**valid_meters, lr=optimizer.param_groups[0]["lr"])))
            utils.save_checkpoint(args, global_step, model, optimizer, score=valid_meters["valid_loss"].avg, mode="min")
#             scheduler.step(running_valid_psnr)

#         if (epoch+1) % args.valid_interval == 0:
#             utils.save_checkpoint(args, global_step, model, optimizer, score=train_meters["train_psnr"].avg, mode="max")

    # input_psnr = utils.psnr(noisy_inputs, sample)
    # print(input_psnr)

    logging.info(f"Done training! Best PSNR {utils.save_checkpoint.best_score:.3f} obtained after step {utils.save_checkpoint.best_step}.")


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--data-path", default="data", help="path to data directory")
    parser.add_argument("--dataset", default="SingleVideo", help="train dataset name")
    parser.add_argument("--dataset-aux", default="GoPro", help="video dataset name")
    parser.add_argument("--video", default="rafting", help="video name")
    parser.add_argument("--aug", default=0, type=int, help="augmentations")
    parser.add_argument("--sample", action='store_true', help="sample noise")
    parser.add_argument("--batch-size", default=8, type=int, help="train batch size")
    parser.add_argument("--image-size", default=128, type=int, help="image size for train")
    parser.add_argument("--n-frames", default=5, type=int, help="number of frames for training")
    parser.add_argument("--stride", default=64, type=int, help="stride for patch extraction")
    parser.add_argument("--heldout", action='store_true', help="heldout evaluation")

    # Add model arguments
    parser.add_argument("--model", default="blind-video-net-4", help="model architecture")

    # Add loss function
    parser.add_argument("--loss", default="loglike", help="loss function used for training")

    # Add noise arguments
    parser.add_argument("--noise_dist", default="G", help="G - Gaussian, P - Poisson")
    parser.add_argument("--noise_mode", default="S", help="B - Blind, S - one noise level")
    parser.add_argument('--noise_std', default = 30, type = float,
                        help = 'noise level when mode is S')
    parser.add_argument('--min_noise', default = 0, type = float,
                        help = 'minimum noise level when mode is B')
    parser.add_argument('--max_noise', default = 100, type = float,
                        help = 'maximum noise level when mode is B')

    # Add optimization arguments
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--num-epochs", default=100, type=int, help="force stop training at specified epoch")
    parser.add_argument("--valid-interval", default=1, type=int, help="evaluate every N epochs")
    parser.add_argument("--save-interval", default=1, type=int, help="save a checkpoint every N steps")

    # Parse twice as model arguments are not known the first time
    parser = utils.add_logging_arguments(parser)
    args, _ = parser.parse_known_args()
    models.MODEL_REGISTRY[args.model].add_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
