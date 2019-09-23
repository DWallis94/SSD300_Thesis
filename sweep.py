import pathlib
#import train_ssd
import numpy as np
import os
import shutil
import argparse
from argparse import ArgumentParser

parser = argparse.ArgumentParser(description='Which sort of sweep to run?')
parser.add_argument('--qw', action="store_true",
                    help='quantize weights sweep?')
parser.add_argument('--qa', action="store_true",
                    help='quantize activations sweep?')
parser.add_argument('--pw', action="store_true",
                    help='prune weights sweep?')
parser.add_argument('--pa', action="store_true",
                    help='prune activations sweep?')
parser.add_argument('--start_step', help='step to begin sweep on.', default=138000)
parser.add_argument('--freq', help='frequency to change sweep condition.', default=6000)
parser.add_argument('--class_set', help='which class set to use', default='original')
parser.add_argument('--q_range', help='range to quantize over', default=np.append(np.linspace(32, 2, 16),1))
parser.add_argument('--p_range', help='range to prune over', default=np.linspace(0, 1, 21))
parser.add_argument('--q_incremental', action="store_true", help='perform incremental quantization?', default=True)
parser.add_argument('--p_incremental', action="store_true", help='perform incremental pruning?', default=True)
parser.add_argument('--specify_gpu', default=None, help='Which GPU to use?')

args = parser.parse_args()

def sweep(start_step, freq, range, baseline, increment, train_cmd, eval_cmd, eval_generic, save_dir):

    for file in os.listdir('./logs/'):
        fname = pathlib.Path('./logs/' + file)
        if fname.is_file():
            os.remove(str(fname))

    for file in os.listdir(baseline):
        fname = pathlib.Path(baseline) / file
        if fname.is_file():
            shutil.copy(str(fname), pathlib.Path('./logs/' + file))

    for ind, val in enumerate(range):
        end_step = start_step + (ind+1)*freq
        save_dir_val = save_dir / (str(val) + "/")
        os.makedirs(save_dir_val, exist_ok=True)

        dir_files = os.listdir(save_dir_val)

        if 'predict' in dir_files:
            continue

        elif dir_files:

            for file in os.listdir(save_dir_val):
                fname = pathlib.Path(save_dir_val / file)
                shutil.copy(str(fname), './logs/' + file)

            os.system(eval_cmd.replace("<val>", str(val)))
            os.system(eval_generic)

            shutil.copytree('./logs/predict', str(save_dir_val / 'predict'))

        else:
            cmd1 = train_cmd.replace("<steps_end>", str(end_step)).replace("<val>", str(val))
            cmd2 = eval_cmd.replace("<val>", str(val))

            if not increment and ind > 0:
                # Copy baseline weights to logs dir
                for file in os.listdir('./logs/'):
                    fname = pathlib.Path('./logs/' + file)
                    if fname.is_file():
                        os.remove(str(fname))

                for file in os.listdir(baseline):
                    fname = pathlib.Path(baseline) / file
                    if fname.is_file():
                        shutil.copy(str(fname), pathlib.Path('./logs/' + file))

            # Run model
            with open("./logs/command_log.txt", "w+") as f:
                f.write(cmd1)
                f.write(cmd2)
                f.write(eval_generic)
                f.close()

            os.system(cmd1)
            os.system(cmd2)
            os.system(eval_generic)

            # Copy output to folder
            for file in os.listdir("./logs/"):
                fname = pathlib.Path("./logs/" + file)
                if fname.is_file():
                    shutil.copy(str(fname), str(pathlib.Path(save_dir_val) / file))
                    if not increment:
                        os.remove(fname)
                elif fname.is_dir() and file == "predict":
                    shutil.copytree(str(fname), str(pathlib.Path(save_dir_val) / file))
                    shutil.rmtree(str(fname))


    for file in os.listdir('./logs/'):
        fname = pathlib.Path('./logs/' + file)
        if fname.is_file():
            os.remove(str(fname))

    return


if __name__ == "__main__":
    baseline = "./logs/" + args.class_set + "/baseline/"
    if not pathlib.Path(baseline).is_dir():
        os.mkdir(baseline)
    start_step = int(args.start_step)
    freq = int(args.freq)
    class_set = args.class_set
    logs_dir = pathlib.Path("./logs/" + class_set + "/")
    qw_save_dir = logs_dir / 'q_weights'
    qa_save_dir = logs_dir / 'q_act'
    pw_save_dir = logs_dir / 'p_weights'
    pa_save_dir = logs_dir / 'p_act'
    q_range = [int(round(q)) for q in args.q_range]
    p_range = [float(round(p,2)) for p in args.p_range]

    qw_cmd = "python train_ssd.py --qw_en --qw_bits <val> --max_number_of_steps <steps_end> --batch_size 28 --class_set <class_set>".replace("<class_set>", args.class_set)
    qa_cmd = "python train_ssd.py --qa_en --qa_bits <val> --max_number_of_steps <steps_end> --batch_size 28 --class_set <class_set>".replace("<class_set>", args.class_set)
    pw_cmd = "python train_ssd.py --pw_en --threshold_w <val> --max_number_of_steps <steps_end> --batch_size 28 --class_set <class_set>".replace("<class_set>", args.class_set)
    pa_cmd = "python train_ssd.py --pa_en --threshold_a <val> --max_number_of_steps <steps_end> --batch_size 28 --class_set <class_set>".replace("<class_set>", args.class_set)

    qw_eval_cmd = "python eval_ssd.py --qw_en --qw_bits <val> --class_set <class_set>".replace("<class_set>", args.class_set)
    qa_eval_cmd = "python eval_ssd.py --qa_en --qa_bits <val> --class_set <class_set>".replace("<class_set>", args.class_set)
    pw_eval_cmd = "python eval_ssd.py --pw_en --threshold_w <val> --class_set <class_set>".replace("<class_set>", args.class_set)
    pa_eval_cmd = "python eval_ssd.py --pa_en --threshold_a <val> --class_set <class_set>".replace("<class_set>", args.class_set)

    eval_cmd = "python voc_eval.py --class_set <class_set>".replace("<class_set>", args.class_set)

    if args.specify_gpu is not None:
        qw_cmd = qw_cmd + " --specify_gpu " + args.specify_gpu
        qa_cmd = qa_cmd + " --specify_gpu " + args.specify_gpu
        pw_cmd = pw_cmd + " --specify_gpu " + args.specify_gpu
        pa_cmd = pa_cmd + " --specify_gpu " + args.specify_gpu

        qw_eval_cmd = qw_eval_cmd + " --specify_gpu " + args.specify_gpu
        qa_eval_cmd = qa_eval_cmd + " --specify_gpu " + args.specify_gpu
        pw_eval_cmd = pw_eval_cmd + " --specify_gpu " + args.specify_gpu
        pa_eval_cmd = pa_eval_cmd + " --specify_gpu " + args.specify_gpu


    if args.qw:
        sweep(start_step, freq, q_range, baseline, args.q_incremental, qw_cmd, qw_eval_cmd, eval_cmd, qw_save_dir)
    if args.qa:
        sweep(start_step, freq, q_range, baseline, args.q_incremental, qa_cmd, qa_eval_cmd, eval_cmd, qa_save_dir)
    if args.pw:
        sweep(start_step, freq, p_range, baseline, args.p_incremental, pw_cmd, pw_eval_cmd, eval_cmd, pw_save_dir)
    if args.pa:
        sweep(start_step, freq, p_range, baseline, args.p_incremental, pa_cmd, pa_eval_cmd, eval_cmd, pa_save_dir)
    exit()
