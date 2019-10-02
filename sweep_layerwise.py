import pathlib

# import train_ssd
import numpy as np
import os
import shutil
import argparse
from argparse import ArgumentParser

parser = argparse.ArgumentParser(description="Which sort of sweep to run?")
parser.add_argument("--start_step", help="step to begin sweep on.", default=138000)
parser.add_argument("--freq", help="frequency to change sweep condition.", default=6000)
parser.add_argument("--class_set", help="which class set to use", default="original")
parser.add_argument(
    "--p_range", help="range to prune over", default=np.linspace(0, 0.5, 11)
)
parser.add_argument(
    "--layers",
    help="comma separated list of layers to prune",
    default="conv1_2,conv2_1,conv2_2,conv3_1,conv3_2,conv3_3,conv4_1,conv4_2,conv4_3,conv5_1,conv5_2,conv5_3,fc6,fc7,conv8_1,conv8_2,conv9_1,conv9_2,conv10_1,conv10_2,conv11_1,conv11_2",
)

parser.add_argument("--specify_gpu", default=None, help="Which GPU to use?")

args = parser.parse_args()


def sweep(
    layer, start_step, freq, range, baseline, train_cmd, eval_cmd, voc_cmd, save_dir
):

    for file in os.listdir("./logs/"):
        fname = pathlib.Path("./logs/" + file)
        if fname.is_file():
            os.remove(str(fname))

    for file in os.listdir(baseline):
        fname = pathlib.Path(baseline) / file
        if fname.is_file():
            shutil.copy(str(fname), pathlib.Path("./logs/" + file))

    for ind, val in enumerate(range):
        end_step = start_step + (ind + 1) * freq
        save_dir_val = save_dir / (layer + "/" + str(val) + "/")
        os.makedirs(save_dir_val, exist_ok=True)

        dir_files = os.listdir(save_dir_val)

        if "predict" in dir_files:
            continue

        elif dir_files:

            for file in os.listdir(save_dir_val):
                fname = pathlib.Path(save_dir_val / file)
                shutil.copy(str(fname), "./logs/" + file)

            os.system(eval_cmd.replace("<val>", str(val)))
            os.system(voc_cmd)

            shutil.copytree("./logs/predict", str(save_dir_val / "predict"))

        else:
            cmd1 = train_cmd.replace("<steps_end>", str(end_step)).replace(
                "<val>", str(val)
            )
            cmd2 = eval_cmd.replace("<val>", str(val))

            if not increment and ind > 0:
                # Copy baseline weights to logs dir
                for file in os.listdir("./logs/"):
                    fname = pathlib.Path("./logs/" + file)
                    if fname.is_file():
                        os.remove(str(fname))

                for file in os.listdir(baseline):
                    fname = pathlib.Path(baseline) / file
                    if fname.is_file():
                        shutil.copy(str(fname), pathlib.Path("./logs/" + file))

            # Run model
            with open("./logs/command_log.txt", "w+") as f:
                f.write(cmd1)
                f.write(cmd2)
                f.write(voc_cmd)
                f.close()

            os.system(cmd1)
            os.system(cmd2)
            os.system(voc_cmd)

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

    for file in os.listdir("./logs/"):
        fname = pathlib.Path("./logs/" + file)
        if fname.is_file():
            os.remove(str(fname))

    return


if __name__ == "__main__":
    layers = args.layers.split(",")
    baseline = "./logs/" + args.class_set + "/baseline/"
    if not pathlib.Path(baseline).is_dir():
        os.mkdir(baseline)
    start_step = int(args.start_step)
    freq = int(args.freq)
    class_set = args.class_set
    logs_dir = pathlib.Path("./logs/" + class_set + "/layerwise/")
    pw_save_dir = logs_dir / "p_weights"
    pa_save_dir = logs_dir / "p_act"
    p_range = [float(round(p, 2)) for p in args.p_range]
    eval_cmd = "python eval_ssd.py --class_set " + class_set
    voc_cmd = "python voc_eval.py --class_set " + class_set

    train_cmd = (
        "python train_ssd.py --max_number_of_steps <steps_end> --batch_size 28 --class_set "
        + class_set
    )

    if args.specify_gpu is not None:
        train_cmd = train_cmd + " --specify_gpu " + args.specify_gpu
        eval_cmd = eval_cmd + " --specify_gpu " + args.specify_gpu

    for layer in layers:
        train_cmd_specific_w = (
            train_cmd + " --pw_<layer> True --tw_<layer> <val>"
        ).replace("<layer>", layer)
        train_cmd_specific_a = (
            train_cmd + " --pa_<layer> True --ta_<layer> <val>"
        ).replace("<layer>", layer)
        eval_cmd_specific_w = (
            eval_cmd + " --pw_<layer> True --tw_<layer> <val>"
        ).replace("<layer>", layer)
        eval_cmd_specific_a = (
            eval_cmd + " --pa_<layer> True --ta_<layer> <val>"
        ).replace("<layer>", layer)
        sweep(
            layer,
            start_step,
            freq,
            p_range,
            baseline,
            train_cmd_specific_w,
            eval_cmd_specific_w,
            voc_cmd,
            pw_save_dir,
        )
        sweep(
            layer,
            start_step,
            freq,
            p_range,
            baseline,
            train_cmd_specific_a,
            eval_cmd_specific_a,
            voc_cmd,
            pa_save_dir,
        )
    exit()
