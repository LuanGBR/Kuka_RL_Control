#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

~/.mujoco/mujoco-2.3.0/bin/simulate ./environment.xml
