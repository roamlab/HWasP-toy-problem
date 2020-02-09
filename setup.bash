# This file needs to be at the root of project dir
rl_env #defined in my.bashrc
export PROJECTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=${PROJECTDIR}
source ${PROJECTDIR}/venv_mass_spring/bin/activate
echo "Mass-spring toy problem environment ready."
