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
parser.add_argument('--baseline', help='which model dir to use as baseline for model runs', default='./logs/baseline/')
parser.add_argument('--q_incremental', action="store_true", help='perform incremental quantization?', default=True)
parser.add_argument('--p_incremental', action="store_true", help='perform incremental pruning?', default=True)

args = parser.parse_args()

def q_weights(start_step, freq, range, baseline, increment):
    for ind, val in enumerate(range):
        val = int(val)
        end_step = start_step + (ind+1)*freq
        save_dir = "./logs/" + args.class_set + "/q_weights/" + str(val) + "/"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if not os.listdir(save_dir):
            command_scaffold = "python train_ssd.py --batch_size 28 --max_number_of_steps <steps_end> --qw_en --qw_bits <q> --class_set <class_set>"
            command_scaffold = command_scaffold.replace("<steps_end>", str(end_step)).replace("<q>", str(val)).replace("<class_set>", args.class_set)
            eval_scaffold = "python eval_ssd.py --qw_en --qw_bits <q> --class_set <class_set>".replace("<q>", str(val)).replace("<class_set>", args.class_set)
            voc_scaffold = "python voc_eval.py --class_set <class_set>".replace("<class_set>", args.class_set)

            if not increment or ind == 0:
                # Copy baseline weights to logs dir
                for file in os.listdir(baseline):
                    if os.path.isfile(baseline + file):
                        shutil.copy(baseline + file, "./logs/" + file)

            # Run model
            with open("./logs/command_log.txt", "w+") as f:
                f.write(command_scaffold)
                f.close()
            os.system(command_scaffold)
            os.system(eval_scaffold)
            os.system(voc_scaffold)

            # Copy output to folder
            for file in os.listdir("./logs/"):
                file_path = "./logs/" + file
                if os.path.isfile(file_path):
                    shutil.copy(file_path, save_dir + file)
                    if not increment:
                        os.remove(file_path)
                elif os.path.isdir(file_path) and file == "predict":
                    shutil.copytree(file_path, save_dir + file)
                    shutil.rmtree(file_path)


    for file in os.listdir("./logs/"):
        file_path = "./logs/" + file
        if os.path.isfile(file_path):
            os.remove(file_path)

    return


def q_activations(start_step, freq, range, baseline, increment):
    for ind, val in enumerate(range):
        val = int(val)
        end_step = start_step + (ind+1)*freq
        save_dir = "./logs/" + args.class_set + "/q_act/" + str(val) + "/"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if not os.listdir(save_dir):
            command_scaffold = "python train_ssd.py --batch_size 28 --max_number_of_steps <steps_end> --qa_en --qa_bits <q> --class_set <class_set>"
            command_scaffold = command_scaffold.replace("<steps_end>", str(end_step)).replace("<q>", str(val)).replace("<class_set>", args.class_set)
            eval_scaffold = "python eval_ssd.py --qa_en --qa_bits <q> --class_set <class_set>".replace("<q>", str(val)).replace("<class_set>", args.class_set)
            voc_scaffold = "python voc_eval.py --class_set <class_set>".replace("<class_set>", args.class_set)

            if not increment or ind == 0:
                # Copy baseline weights to logs dir
                for file in os.listdir(baseline):
                    if os.path.isfile(baseline + file):
                        shutil.copy(baseline + file, "./logs/" + file)

            # Run model
            with open("./logs/command_log.txt", "w+") as f:
                f.write(command_scaffold)
                f.close()
            os.system(command_scaffold)
            os.system(eval_scaffold)
            os.system(voc_scaffold)

            # Copy output to folder
            for file in os.listdir("./logs/"):
                file_path = "./logs/" + file
                if os.path.isfile(file_path):
                    shutil.copy(file_path, save_dir + file)
                    if not increment:
                        os.remove(file_path)
                elif os.path.isdir(file_path) and file == "predict":
                    shutil.copytree(file_path, save_dir + file)
                    shutil.rmtree(file_path)

    for file in os.listdir("./logs/"):
        file_path = "./logs/" + file
        if os.path.isfile(file_path):
            os.remove(file_path)

    return


def p_weights(start_step, freq, range, baseline, increment):
    for ind, val in enumerate(range):
        end_step = start_step + (ind+1)*freq
        save_dir = "./logs/" + args.class_set + "/p_weights/" + str(val) + "/"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if not os.listdir(save_dir):
            command_scaffold = "python train_ssd.py --batch_size 28 --max_number_of_steps <steps_end> --pw_en --threshold_w <t> --begin_pruning_at_step 0 --pruning_frequency 1 --class_set <class_set>"
            command_scaffold = command_scaffold.replace("<steps_end>", str(end_step)).replace("<t>", str(val)).replace("<class_set>", args.class_set)
            eval_scaffold = "python eval_ssd.py --pw_en --threshold_w <t> --begin_pruning_at_step 0 --pruning_frequency 1 --class_set <class_set>".replace("<q>", str(val)).replace("<class_set>", args.class_set)
            voc_scaffold = "python voc_eval.py --class_set <class_set>".replace("<class_set>", args.class_set)

            if not increment or ind == 0:
                # Copy baseline weights to logs dir
                for file in os.listdir(baseline):
                    if os.path.isfile(baseline + file):
                        shutil.copy(baseline + file, "./logs/" + file)

            # Run model
            with open("./logs/command_log.txt", "w+") as f:
                f.write(command_scaffold)
                f.close()
            os.system(command_scaffold)
            os.system(eval_scaffold)
            os.system(voc_scaffold)

            # Copy output to folder
            for file in os.listdir("./logs/"):
                file_path = "./logs/" + file
                if os.path.isfile(file_path):
                    shutil.copy(file_path, save_dir + file)
                    if not increment:
                        os.remove(file_path)
                elif os.path.isdir(file_path) and file == "predict":
                    shutil.copytree(file_path, save_dir + file)
                    shutil.rmtree(file_path)

    for file in os.listdir("./logs/"):
        file_path = "./logs/" + file
        if os.path.isfile(file_path):
            os.remove(file_path)

    return


def p_activations(start_step, freq, range, baseline, increment):
    for ind, val in enumerate(range):
        end_step = start_step + (ind+1)*freq
        save_dir = "./logs/" + args.class_set + "/p_act/" + str(val) + "/"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if not os.listdir(save_dir):
            command_scaffold = "python train_ssd.py --batch_size 28 --max_number_of_steps <steps_end> --pa_en --threshold_a <t> --begin_pruning_at_step 0 --pruning_frequency 1 --class_set <class_set>"
            command_scaffold = command_scaffold.replace("<steps_end>", str(end_step)).replace("<t>", str(val)).replace("<class_set>", args.class_set)
            eval_scaffold = "python eval_ssd.py --pa_en --threshold_a <t> --begin_pruning_at_step 0 --pruning_frequency 1 --class_set <class_set>".replace("<q>", str(val)).replace("<class_set>", args.class_set)
            voc_scaffold = "python voc_eval.py --class_set <class_set>".replace("<class_set>", args.class_set)

            if not increment or ind == 0:
                # Copy baseline weights to logs dir
                for file in os.listdir(baseline):
                    if os.path.isfile(baseline + file):
                        shutil.copy(baseline + file, "./logs/" + file)

            # Run model
            with open("./logs/command_log.txt", "w+") as f:
                f.write(command_scaffold)
                f.close()
            os.system(command_scaffold)
            os.system(eval_scaffold)
            os.system(voc_scaffold)

            # Copy output to folder
            for file in os.listdir("./logs/"):
                file_path = "./logs/" + file
                if os.path.isfile(file_path):
                    shutil.copy(file_path, save_dir + file)
                    if not increment:
                        os.remove(file_path)
                elif os.path.isdir(file_path) and file == "predict":
                    shutil.copytree(file_path, save_dir + file)
                    shutil.rmtree(file_path)

    for file in os.listdir("./logs/"):
        file_path = "./logs/" + file
        if os.path.isfile(file_path):
            os.remove(file_path)

    return


if __name__ == "__main__":
    baseline = "./logs/" + args.class_set + "/baseline/"
    if not pathlib.Path(baseline).is_dir():
        os.mkdir(baseline)
    start_step = args.start_step
    freq = args.freq
    class_set = args.class_set
    logs_dir = pathlib.Path("./logs/" + class_set + "/")
    if args.qw:
        q_weights(start_step, freq, args.q_range, baseline, args.q_incremental)
    if args.qa:
        q_activations(start_step, freq, args.q_range, baseline, args.q_incremental)
    if args.pw:
        p_weights(start_step, freq, args.p_range, baseline, args.p_incremental)
    if args.pa:
        p_activations(start_step, freq, args.p_range, baseline, args.p_incremental)
    exit()
