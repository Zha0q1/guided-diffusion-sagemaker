"""
Train a diffusion model on images.
"""

import argparse

# from guided_diffusion import dist_util, logger ## SageMaker
from guided_diffusion import logger ## SageMaker
from guided_diffusion.image_datasets import load_data
# from guided_diffusion.resample import create_named_schedule_sampler ## SageMaker
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
# from guided_diffusion.train_util import TrainLoop  ## SageMaker


def main():
    args = create_argparser().parse_args()

    ######################################## 1. Add path for SageMaker #######################################
    args = check_sagemaker(args)
    
    if args.sagemakerdp:
        from guided_diffusion import dist_util_smdp as dist_util
        from guided_diffusion.resample_smdp import create_named_schedule_sampler
        from guided_diffusion.train_util_smdp import TrainLoop
    else:
        from guided_diffusion import dist_util
        from guided_diffusion.resample import create_named_schedule_sampler
        from guided_diffusion.train_util import TrainLoop
    
    ##########################################################################################################
    dist_util.setup_dist()
    logger.configure()
    
    resource_check()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    
#     print(f"use_fp16 : {args.use_fp16}")
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        s3_log_path="s3://",
        sagemakerdp=False      ##### 2. SageMaker DDP
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

## 3. Add function for SageMaker
def check_sagemaker(args):
    import os
    import json
    
    ## SageMaker
#     print(f"os.environ : {os.environ}")
    if os.environ['SM_MODEL_DIR'] is not None:
        args.data_dir = os.environ['SM_CHANNEL_TRAINING']
        try:
            job_name = os.environ['SAGEMAKER_JOB_NAME']
        except:
            import datetime
            job_name = datetime.datetime.now().strftime("diffusion-%Y-%m-%d-%H-%M-%S-%f")
            pass
        os.environ['JOB_NAME'] = job_name
        os.environ['OPENAI_LOGDIR'] = '/tmp/' + os.environ['JOB_NAME']
        os.environ['DIFFUSION_BLOB_LOGDIR'] = '/opt/ml/checkpoints/' + os.environ['JOB_NAME']
        os.environ['S3_LOG_PATH'] = args.s3_log_path + "/" + os.environ['JOB_NAME']
        
        ## Create Directory
        os.makedirs(os.environ['OPENAI_LOGDIR'], exist_ok=True)
        os.makedirs(os.environ['DIFFUSION_BLOB_LOGDIR'], exist_ok=True)
    return args


def resource_check():
    import os
    if os.environ["RANK"] == '0':
        import subprocess

        result = subprocess.run(['df', '-h'], stdout=subprocess.PIPE)
        print("Disk Size:", result.stdout.decode('utf-8'))
        result = subprocess.run(['df', '-ih'], stdout=subprocess.PIPE)
        print("Disk Size (inode) : ", result.stdout.decode('utf-8'))        

        result = subprocess.run(['df', '-h', '/opt/ml/checkpoints'], stdout=subprocess.PIPE)
        print("/opt/ml/checkpoints : ", result.stdout.decode('utf-8'))
        result = subprocess.run(['df', '-h', '/opt/ml/model'], stdout=subprocess.PIPE)
        print("/opt/ml/model : ", result.stdout.decode('utf-8'))   
        
        result = subprocess.run(['df', '-h', '/opt/ml/code'], stdout=subprocess.PIPE)
        print("/opt/ml/code", result.stdout.decode('utf-8'))
        result = subprocess.run(['df', '-h', '/opt/ml/input/data'], stdout=subprocess.PIPE)
        print("/opt/ml/input/data", result.stdout.decode('utf-8'))   
        
        
        
if __name__ == "__main__":
    main()
