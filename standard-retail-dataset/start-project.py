#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from cookiecutter.main import cookiecutter

this_dir = Path(__file__).resolve().parent

AVAILABLE_TEMPLATES = ["flask", "djangorestframework", "base-python"]


def copy_files(src_dir, dst_dir):
    with tempfile.NamedTemporaryFile() as fp:
        tarfile = f"{fp.name}.tar"
        shutil.make_archive(fp.name, "tar", src_dir)
        shutil.unpack_archive(tarfile, dst_dir)


def copy_cookiecutter_dir(src_dir, dst_dir, config, pre_gen_hook, post_gen_hook):
    config_json_file = src_dir / "cookiecutter.json"
    pre_gen_hook_file = src_dir / "hooks" / "pre_gen_hook.sh"
    post_gen_hook_file = src_dir / "hooks" / "post_gen_project.sh"
    if config_json_file.exists():
        with config_json_file.open() as fp:
            current_config = json.load(fp)
        if pre_gen_hook_file.exists():
            with pre_gen_hook_file.open() as fp:
                for l in reversed(fp.readlines()):
                    pre_gen_hook.insert(0, l)
        if post_gen_hook_file.exists():
            with post_gen_hook_file.open() as fp:
                for l in reversed(fp.readlines()):
                    post_gen_hook.insert(0, l)
        parent = current_config.pop("_parent", None)
        if parent is not None:
            copy_cookiecutter_dir(
                this_dir / parent, dst_dir, config, pre_gen_hook, post_gen_hook
            )
        config.update(current_config)

    copy_files(src_dir, dst_dir)

    with (dst_dir / "cookiecutter.json").open("w") as fp:
        json.dump(config, fp)
    pre_gen_hook_file = dst_dir / "hooks" / "pre_gen_hook.sh"
    if pre_gen_hook:
        with pre_gen_hook_file.open("w") as fp:
            fp.writelines(pre_gen_hook)
    elif pre_gen_hook_file.exists():
        os.remove(pre_gen_hook_file)
    post_gen_hook_file = dst_dir / "hooks" / "post_gen_project.sh"
    if post_gen_hook:
        with post_gen_hook_file.open("w") as fp:
            fp.writelines(post_gen_hook)
    elif post_gen_hook_file.exists():
        os.remove(post_gen_hook_file)


def render_cookiecutter(cookiecutter_dir, no_input=False, overwrite_if_exists=False):
    with tempfile.TemporaryDirectory() as tempdir:
        dst_dir = Path(tempdir) / "cookiecutter"
        copy_cookiecutter_dir(Path(cookiecutter_dir), dst_dir, {}, [], [])
        cookiecutter(
            str(dst_dir), no_input=no_input, overwrite_if_exists=overwrite_if_exists
        )


def main():
    parser = argparse.ArgumentParser()
    available_templates = ", ".join(AVAILABLE_TEMPLATES)
    parser.add_argument(
        "cookiecutter_dir",
        help=f"cookiecutter template to use. Available options: {available_templates}",
    )
    parser.add_argument(
        "--no-input",
        action="store_true",
        help="don't prompt for values and use defaults",
    )
    parser.add_argument(
        "--overwrite-if-exists",
        action="store_true",
        help="don't prompt for values and use defaults",
    )

    args = parser.parse_args()

    dir_arg = args.cookiecutter_dir
    if dir_arg in AVAILABLE_TEMPLATES:
        cookiecutter_dir = Path(__file__).parent.joinpath(f"cookiecutters/{dir_arg}")
    else:
        cookiecutter_dir = dir_arg

    render_cookiecutter(
        cookiecutter_dir,
        no_input=args.no_input,
        overwrite_if_exists=args.overwrite_if_exists,
    )


if __name__ == "__main__":
    main()
