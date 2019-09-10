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
parser.add_argument('--freq', help='frequency to change sweep condition.', default=4000)
parser.add_argument('--class_set', help='which class set to use', default='original')
parser.add_argument('--q_range', help='range to quantize over', default=[1,2,4,8,16,32])
parser.add_argument('--p_range', help='range to prune over', default=np.linspace(0, 1, 21))
parser.add_argument('--baseline', help='which model dir to use as baseline for model runs', default='./logs/baseline/')
parser.add_argument('--q_incremental', action="store_true", help='perform incremental quantization?', default=True)
parser.add_argument('--p_incremental', action="store_true", help='perform incremental pruning?', default=True)

args = parser.parse_args()

def q_weights(start_step, freq, range, baseline, increment):
    for ind, val in enumerate(range):
        start_step = start_step + ind*freq
        end_step = start_step + freq
        save_dir = "./logs/" + args.class_set + "/q_weights/" + str(val) + "/"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if not os.listdir(save_dir):
            command_scaffold = "python train_ssd.py --batch_size 28 --max_number_of_steps <steps_end> --qw_en --qw_bits <q> --class_set <class_set>"
            command_scaffold = command_scaffold.replace("<steps_end>", str(end_step)).replace("<q>", str(val)).replace("<class_set>", args.class_set)

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

            # Copy output to folder
            for file in os.listdir("./logs/"):
                file_path = "./logs/" + file
                if os.path.isfile(file_path):
                    shutil.copy(file_path, save_dir + file)
                    if not increment:
                        os.remove(file_path)

    for file in os.listdir("./logs/"):
        file_path = "./logs/" + file
        if os.path.isfile(file_path):
            os.remove(file_path)

    return start_step


def q_activations(start_step, freq, range, baseline, increment):
    for ind, val in enumerate(range):
        start_step = start_step + ind*freq
        end_step = start_step + freq
        save_dir = "./logs/" + args.class_set + "/q_act/" + str(val) + "/"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if not os.listdir(save_dir):
            command_scaffold = "python train_ssd.py --batch_size 28 --max_number_of_steps <steps_end> --qa_en --qa_bits <q> --class_set <class_set>"
            command_scaffold = command_scaffold.replace("<steps_end>", str(end_step)).replace("<q>", str(val)).replace("<class_set>", args.class_set)

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

            # Copy output to folder
            for file in os.listdir("./logs/"):
                file_path = "./logs/" + file
                if os.path.isfile(file_path):
                    shutil.copy(file_path, save_dir + file)
                    if not increment:
                        os.remove(file_path)

    for file in os.listdir("./logs/"):
        file_path = "./logs/" + file
        if os.path.isfile(file_path):
            os.remove(file_path)

    return start_step


def p_weights(start_step, freq, range, baseline, increment):
    for ind, val in enumerate(range):
        start_step = start_step + ind*freq
        end_step = start_step + freq
        save_dir = "./logs/" + args.class_set + "/p_weights/" + str(val) + "/"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if not os.listdir(save_dir):
            command_scaffold = "python train_ssd.py --batch_size 28 --max_number_of_steps <steps_end> --pw_en --threshold_w <t> --class_set <class_set>"
            command_scaffold = command_scaffold.replace("<steps_end>", str(end_step)).replace("<t>", str(val)).replace("<class_set>", args.class_set)

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

            # Copy output to folder
            for file in os.listdir("./logs/"):
                file_path = "./logs/" + file
                if os.path.isfile(file_path):
                    shutil.copy(file_path, save_dir + file)
                    if not increment:
                        os.remove(file_path)

    for file in os.listdir("./logs/"):
        file_path = "./logs/" + file
        if os.path.isfile(file_path):
            os.remove(file_path)

    return start_step


def p_activations(start_step, freq, range, baseline, increment):
    for ind, val in enumerate(range):
        start_step = start_step + ind*freq
        end_step = start_step + freq
        save_dir = "./logs/" + args.class_set + "/p_act/" + str(val) + "/"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if not os.listdir(save_dir):
            command_scaffold = "python train_ssd.py --batch_size 28 --max_number_of_steps <steps_end> --pa_en --threshold_a <t> --class_set <class_set>"
            command_scaffold = command_scaffold.replace("<steps_end>", str(end_step)).replace("<t>", str(val)).replace("<class_set>", args.class_set)

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

            # Copy output to folder
            for file in os.listdir("./logs/"):
                file_path = "./logs/" + file
                if os.path.isfile(file_path):
                    shutil.copy(file_path, save_dir + file)
                    if not increment:
                        os.remove(file_path)

    for file in os.listdir("./logs/"):
        file_path = "./logs/" + file
        if os.path.isfile(file_path):
            os.remove(file_path)

    return start_step


if __name__ == "__main__":
    if not pathlib.Path(args.baseline).is_dir():
        os.mkdir(args.baseline)
    start_step = args.start_step
    freq = args.freq
    class_set = args.class_set
    logs_dir = pathlib.Path("./logs/" + class_set + "/")
    if args.qw:
        start_step = q_weights(start_step, freq, args.q_range, args.baseline, args.q_incremental)
    if args.qa:
        start_step = q_activations(start_step, freq, args.q_range, args.baseline, args.q_incremental)
    if args.pw:
        start_step = p_weights(start_step, freq, args.p_range, args.baseline, args.p_incremental)
    if args.pa:
        start_step = p_activations(start_step, freq, args.p_range, args.baseline, args.p_incremental)
    exit()
